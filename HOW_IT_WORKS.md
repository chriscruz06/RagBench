# How It Works

> A plain-language walkthrough of the RagBench pipeline; what happens from raw text to cited answer, plus how I measured whether it actually works.

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

**Chunking** (`chunker.py`): Most documents are too long to embed as a single unit; a full book of Genesis wouldn't be a useful search result for a question about a specific verse. The chunker breaks documents into smaller pieces. I implemented three strategies behind a common interface:

- **Fixed-size**: Split every N characters with overlap. Simple and fast, but can cut mid-sentence.
- **Sentence-level**: Split on sentence boundaries, then group sentences until the chunk reaches a size limit. Preserves meaning better than fixed-size since it doesn't break thoughts in half.
- **Semantic**: Embeds each sentence, computes cosine similarity between adjacent sentences, and splits where similarity drops below a threshold. The idea is to find natural topic boundaries instead of arbitrary size cutoffs.

The strategy is configurable via environment variable, which enabled controlled A/B testing in Phase 3 (more on that later, spoiler: the "smartest" strategy wasn't the best).

**Embedding and storage** (`embedder.py`): Each chunk gets converted into a numerical vector (a list of numbers that captures its meaning) using a sentence-transformer model. I started with `all-MiniLM-L6-v2` as a fast baseline but upgraded to `BAAI/bge-base-en-v1.5` after realizing the MiniLM embeddings were too weak for theological text; the language is archaic and domain-specific enough that a stronger model made a noticeable difference. These vectors are stored in ChromaDB, a local vector database, using cosine similarity as the distance metric. Chunk IDs are deterministic (based on source + index), so re-running ingestion doesn't create duplicates.

**Output:** A ChromaDB collection with ~7,500 vectors, each linked back to its source text and metadata.

---

### 3. Retrieval | Finding the Right Passages

When a user asks a question, the retrieval layer finds the most relevant chunks using a **two-stage pipeline**:

1. **Bi-encoder search (fast, wide net)**: The question is embedded into a vector using the same BGE model that embedded the chunks, and ChromaDB performs a cosine similarity search to return the top 50 candidates. Bi-encoders are fast because they embed the question and the corpus independently, but they're less precise, they might miss nuance in how the question relates to a specific chunk.

2. **Cross-encoder reranking (slow, precise)**: A second model (`ms-marco-MiniLM-L-6-v2`) takes each of those 50 candidates and scores them *together with* the question. This is slower because it runs 50 separate model calls, but much more accurate because it can model the interaction between question and chunk directly. The top results (after a threshold filter) go to the LLM.

I tuned the reranker threshold down to -4.0 after finding that abstract relational questions ("How does faith relate to works?") scored lower than direct factual ones ("What is the Eucharist?"), which was causing good candidates to get filtered out. Watching the scores on real queries taught me more than any blog post could.

If no chunk scores above the threshold after reranking, the system returns zero results, which triggers the generation layer to **abstain** rather than answer. This trades recall (sometimes declining to answer questions it could handle) for faithfulness (never fabricating an answer from thin air). In a theological context, inventing doctrine is worse than saying "I'm not sure", you don't want to accidentally recreate some ancient heresy (plus, something like a Nestorian AI just sounds like a black mirror episode).

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
| Eval framework | End-to-end | Quantifies faithfulness across configurations |

---

## Design Decisions Worth Noting

**Why local embeddings + local LLM?** The default configuration (sentence-transformers + Ollama) runs entirely on the user's machine with zero API costs. OpenAI is available as a drop-in alternative for higher quality, but the local-first approach means anyone can run the full pipeline without an API key.  Plus, I'm personally pretty cheap, so that factored in too.

**Why ChromaDB?** It's lightweight, runs in-process, persists to disk, and handles the scale of this corpus (thousands of chunks, not millions) without needing a separate database server.

**Why pluggable chunking?** Different chunking strategies produce measurably different retrieval quality. By putting them behind a common interface and making the choice configurable, Phase 3 ran controlled ablation experiments: same questions, same embeddings, same LLM, only the chunking changes. This isolates the variable and produces clean comparisons. (The results section below has the payoff.)

**Why Pydantic Settings?** Every tunable parameter (chunk size, overlap, similarity threshold, model choice, etc.) lives in one `Settings` class that reads from environment variables. This means configuration changes don't require code changes, and the `.env` file documents every knob in one place.

---

## Phase 2: Measuring Whether It Actually Works

Building the pipeline is one thing; knowing whether it works is another. Phase 2 added an evaluation framework that scores the pipeline against a hand-crafted test set of 10 theological questions across 6 topics (sacraments, creed, morality, prayer, eschatology, ecclesiology). Each test case includes a gold-standard answer and a list of expected sources (CCC paragraphs or Scripture verses).

The framework measures two dimensions:

**Retrieval quality** - did we find the right chunks?
- **Precision@K**: Of the chunks we retrieved, how many were in the expected source list?
- **Recall@K**: Of the expected sources, how many did we find?
- **Mean Reciprocal Rank (MRR)**: Where in the ranked list did the first correct source show up? Higher is better.

**Generation quality** - did we write a good answer?
- **Token F1**: Word-level overlap between generated and expected answers.
- **ROUGE-L**: Longest common subsequence, rewards preserving order.
- **BLEU**: N-gram overlap; stricter, penalizes paraphrasing.
- **Source Coverage**: Did the generated answer cite the CCC paragraphs that were retrieved?

The runner (`eval/runner.py`) pipes each test question through the full pipeline, scores the result, and writes a timestamped JSON report to `eval/results/`. Each report includes a config snapshot (embedding model, chunk strategy, top-K, etc.) so I can tell later exactly what produced the numbers. A companion module (`eval/report.py`) loads multiple reports and renders comparison tables with delta arrows, per-topic breakdowns, and Plotly charts; this became invaluable in Phase 3.

One finding from the baseline numbers: my retrieval metrics looked low (P@K=0.14, R@K=0.15), but the retrieved chunks were almost always topically relevant, they just had different CCC paragraph numbers than my test set's "expected" list. The CCC covers most topics in multiple places, so a correct answer can be assembled from several valid starting points. The metric was penalizing the retriever for finding CCC §1213 when my test set expected §1265, even though both are about Baptism and both are correct. This is a test set limitation, not a retrieval failure, and it's a good reminder that metrics without interpretation can mislead you.

---

## Phase 3: Ablation - Does Fancy Always Win?

Phase 3's central question: **does the chunking strategy actually matter, and which one is best for this corpus?**

I built an ablation runner (`experiments/ablation.py`) that:
1. Resets the vector store
2. Re-ingests the full corpus (~2,800 documents, ~7,500+ chunks) using a specific chunking strategy
3. Runs the full Phase 2 eval suite with a tagged report
4. Repeats for each strategy

Everything else was held constant: embedding model, retrieval top-K, reranker, LLM, test set. The only variable was chunking.

### The Results

| Strategy | P@K | R@K | MRR | Token F1 | ROUGE-L | BLEU | Chunks |
|----------|------|------|------|----------|---------|------|--------|
| Fixed | 0.100 | 0.150 | 0.245 | **0.263** | 0.170 | 0.021 | 7,744 |
| **Sentence** | **0.140** | **0.150** | **0.270** | 0.252 | **0.182** | **0.036** | 7,376 |
| Semantic | 0.080 | 0.125 | 0.175 | 0.258 | 0.171 | 0.029 | 11,751 |

### What I Actually Learned

**Sentence chunking won on retrieval by a meaningful margin.** P@K improved 40% over fixed and 75% over semantic. Sentence-level boundaries respect how the corpus is actually organized without fighting it.

**Semantic chunking (the "smartest" strategy) came in last.** This was the surprise, and it's the finding I'm most glad I measured instead of assumed. I expected semantic to win because it uses the same embedding model that does retrieval to find topic boundaries. It felt principled. But in practice, it produced **59% more chunks than the other strategies** (11,751 vs ~7,500) without improving anything.

The reason, in hindsight, is obvious: **CCC paragraphs are already self-contained doctrinal units.** Someone spent many, many hours carefully crafting each paragraph to express exactly one coherent idea. When the semantic chunker looks inside a paragraph and sees a "topic shift" (say, doctrine transitioning into a scriptural citation) it splits the paragraph in half. But that citation is *supporting* the doctrine; they belong together. The chunker was fighting the document's structure, not respecting it.

This is the kind of finding ablation experiments exist to surface. Complexity for its own sake lost to something that just respects the corpus.

**Generation metrics were almost flat across strategies.** Token F1 ranged only 0.252–0.263 (~4% spread) despite P@K ranging 0.080–0.140 (a 75% spread). What I think this means is that Mistral is doing a lot of paraphrasing work regardless of which chunks it gets, it produces similar-quality answers whether the retrieved paragraphs are perfect hits or just topically adjacent. The LLM's paraphrasing smooths over retrieval differences downstream.

The practical takeaway: **retrieval metrics are the more sensitive indicator of pipeline quality on this corpus.** If I'd only looked at Token F1, I might have concluded chunking barely matters. The retrieval numbers told a much clearer story.

---

## What's Next

- **A frontend.** Right now the only way to query RagBench is through the terminal or a raw API call. A simple web UI would make it actually usable.
- **Expanding the test set.** 10 questions is enough to compare configurations, but a few topics scored 0.0 on retrieval (morality, prayer, ecclesiology), I suspect that's because those topics are covered across many paragraphs and my "expected sources" list is too narrow. More questions and more generous gold-standard matching would give a truer picture.
- **Maybe deployment.** The "zero-cost local tooling" constraint means traditional cloud deployment would mean adding API costs, which I'd rather not. But a Docker image that bundles everything for someone to run locally is a reasonable middle ground.