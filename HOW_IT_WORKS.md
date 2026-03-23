# How It Works

> A plain-language walkthrough of the RagBench Phase 1 pipeline; what happens from raw text to cited answer.

This is only really my second foray into LLMs, so I wanted to make a little document for myself explaining what exactly is going on just due to the volume of new concepts.  Then I looked at my document and said, "Hey, this would be useful for other people as well", so I'm putting it in a markdown file and throwing it in my repo.

If you couldn't tell, this project also comes from a very different side of my life from compsci; my faith.  Two things I know really well, so it kind of works well.  Only issue (which is beneficial in the long run) is I have to be very careful on how the LLM works just to not accidently create the first homebrewed AI heretic (probably happened already now that I think about it).

---

## The One-Sentence Version

RagBench takes two large religious texts, breaks them into searchable pieces, and uses them as a factual "cheat sheet" so that when you ask a theological question, the language model answers **only** from those sources (with citations) instead of making things up.

---

## The Four Steps

### 1. Preprocessing | Turning Raw Text into Clean Pieces

The pipeline starts with two messy source files: the Douay-Rheims Bible (downloaded from Project Gutenberg, complete with boilerplate headers and footers, just because RSV is copyrighted) and the full text of the Catechism of the Catholic Church (CCC).

Before the system can do anything useful with these, they need to be cleaned up and organized.

**For the Bible** (`preprocess_bible.py`):
- Strip the Project Gutenberg header/footer
- Split the single massive file into individual books (Genesis, Exodus, etc.)
- Save each book as its own text file with a consistent naming convention

**For the Catechism** (`preprocess_ccc.py`):
- Parse the text into its ~2,865 numbered paragraphs (the CCC's natural unit of organization)
- Extract scripture cross-references embedded in each paragraph (e.g., "cf. Rom 5:12")
- Output structured JSON where each entry has a paragraph number, text, and list of cross-references

The main idea here is that the CCC already has a natural chunking unit (numbered paragraphs) which makes it much easier to produce meaningful citations later. The Bible gets split into books, which are then chunked further in the next step.  Due to the natural separations, it makes it much easier on the chunking.

**Output:** ~47 clean text files (Bible books) and one JSON file with ~2,865 structured paragraphs (CCC).

---

### 2. Ingestion | Chunking, Embedding, and Storing

Once the texts are clean, the ingestion pipeline converts them into a searchable vector database. This is a three-part process:

**Loading** (`loader.py`): A unified file loader reads `.txt`, `.pdf`, `.md`, and `.json` files into a common `Document` format. Each document carries metadata (source filename, document type, paragraph reference) so that retrieval results can be traced back to their origin.

**Chunking** (`chunker.py`): Most documents are too long to embed as a single unit; a full book of Genesis wouldn't be a useful search result for a question about a specific verse. The chunker breaks documents into smaller pieces. I implemented two strategies behind a common interface (with a third planned for Phase 3 way down the line from time of writing):

- **Fixed-size**: Split every N characters with overlap. Simple and fast, but can cut mid-sentence.
- **Sentence-level**: Split on sentence boundaries, then group sentences until the chunk reaches a size limit. Preserves meaning better than fixed-size since it doesn't break thoughts in half.
- **Semantic** *(Phase 3)*: Will use embedding similarity between consecutive sentences to find natural topic boundaries.

The strategy is configurable via environment variable, which enables controlled A/B testing in the evaluation phase.

**Embedding and storage** (`embedder.py`): Each chunk gets converted into a numerical vector (a list of numbers that captures its meaning) using a sentence-transformer model (`all-MiniLM-L6-v2`). These vectors are stored in ChromaDB, a local vector database, using cosine similarity as the distance metric. Chunk IDs are deterministic (based on source + index), so re-running ingestion doesn't create duplicates.

**Output:** A ChromaDB collection with ~7,500 vectors, each linked back to its source text and metadata.

---

### 3. Retrieval | Finding the Right Passages

When a user asks a question, the retrieval layer finds the most relevant chunks:

1. The question is embedded into a vector using the same model that embedded the chunks
2. ChromaDB performs a cosine similarity search, returning the top-K most similar chunks
3. Any chunk below a **similarity threshold** (default: 0.3) is filtered out

That threshold is an intentional design decision. If no chunk scores above it, the system returns zero results, which triggers the generation layer to **abstain** rather than answer. This trades recall (sometimes declining to answer questions it could handle) for faithfulness (never fabricating an answer from thin air). In a theological context, inventing doctrine is worse than saying "I'm not sure", you don't want to accidentally recreate some ancient heresy (plus, something like a Nestorian AI just sounds like a black mirror episode).

**Output:** A ranked list of 0–5 text chunks, each with a similarity score and full metadata.

---

### 4. Generation | Assembling a Cited Answer

The retrieved chunks are passed to an LLM (Mistral 7B via Ollama locally, or GPT-4o-mini via OpenAI) along with the user's question. But the model doesn't get free rein, it's constrained by a carefully designed system prompt that enforces several rules:

- **"Curator, not theologian"**: The model's job is to organize and present what the sources say, not to interpret or synthesize independently. Basically became a motto for this project.
- **Citation enforcement**: Every claim must reference a specific CCC paragraph (e.g., CCC §1234) or Scripture verse (e.g., Romans 5:8).
- **No external knowledge**: The model is explicitly told to use only the retrieved passages, not its training data.
- **Graceful abstention**: If the retrieved context is insufficient, the model says so and suggests where to look.

If zero chunks were retrieved (because nothing scored above the similarity threshold), the generation layer short-circuits entirely and returns a canned abstention message without calling the LLM at all.

**Output:** A grounded cited answer OR an honest "I dunno."

---

## Hallucination Defense in Depth

A single guardrail isn't enough. The pipeline uses multiple overlapping layers:

| Layer | Where | What It Does |
|-------|-------|--------------|
| System prompt | Generation | Constrains the LLM to act as a curator, not a theologian |
| Citation rules | Generation | Requires CCC § or verse references for every claim |
| Retrieval-heavy design | Generation | Responses are mostly organized quotation, not free generation |
| Similarity threshold | Retrieval | Abstains when no chunk is relevant enough |
| Eval framework | End-to-end | Quantifies faithfulness across configurations *(Phase 2)* |

---

## Design Decisions Worth Noting

**Why local embeddings + local LLM?** The default configuration (sentence-transformers + Ollama) runs entirely on the user's machine with zero API costs. OpenAI is available as a drop-in alternative for higher quality, but the local-first approach means anyone can run the full pipeline without an API key.  Plus, I'm personally pretty cheap, so that factored in too.

**Why ChromaDB?** It's lightweight, runs in-process, persists to disk, and handles the scale of this corpus (thousands of chunks, not millions) without needing a separate database server.

**Why pluggable chunking?** Different chunking strategies produce measurably different retrieval quality. By putting them behind a common interface and making the choice configurable, Phase 3 can run controlled ablation experiments when that time comes: same questions, same embeddings, same LLM, only the chunking changes. This isolates the variable and produces clean comparisons.

**Why Pydantic Settings?** Every tunable parameter (chunk size, overlap, similarity threshold, model choice, etc.) lives in one `Settings` class that reads from environment variables. This means configuration changes don't require code changes, and the `.env` file documents every knob in one place.

---

## What's Next

- **Phase 2** adds a formal evaluation framework: Precision@K, Recall@K, MRR for retrieval quality; Faithfulness, BLEU, ROUGE for generation quality; all measured against a hand-curated test set of 10 question/answer/source triples.
- **Phase 3** runs ablation experiments across chunking strategies and measures the impact on eval metrics, plus deployment.