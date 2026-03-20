# RagBench

> A RAG (Retrieval-Augmented Generation) pipeline with a built-in evaluation framework for benchmarking retrieval and generation quality over Catholic theological texts.

RagBench ingests the **Catechism of the Catholic Church (CCC)** and **Sacred Scripture (Douay-Rheims Bible)**, embeds them into a local vector store, and answers theological questions with grounded, cited responses. An evaluation framework measures retrieval precision, generation faithfulness, and end-to-end accuracy across different pipeline configurations.

The system is designed around a core principle: **the LLM is a curator, not a theologian.** Every response must be grounded in retrieved source material, every claim must cite a specific CCC paragraph or Scripture verse, and the system explicitly abstains when retrieved context is insufficient, trading recall for faithfulness.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Raw Corpus     │────▶│    Ingestion     │────▶│    ChromaDB     │
│   Bible + CCC   │     │  Preprocess →    │     │   7,500+ vecs   │
│                  │     │  Chunk → Embed   │     │   (cosine)      │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          │ retrieve top-K
                                                          │ + threshold filter
┌─────────────────┐     ┌──────────────────┐     ┌────────▼────────┐
│   Answer +       │◀────│   Generation     │◀────│   Retrieval     │
│   CCC §1234      │     │   System prompt  │     │   Similarity    │
│   Romans 5:8     │     │   guardrails     │     │   scoring       │
└────────┬────────┘     └──────────────────┘     └─────────────────┘
         │
         ▼
┌──────────────────┐
│   Eval Framework  │  Precision@K · Recall@K · MRR · Faithfulness
│   (Phase 2)       │  Answer Relevance · BLEU · ROUGE · F1
└──────────────────┘
```

## Example

```
❯ What does the Church teach about the Eucharist?

The retrieved sources provide information about the Eucharist as a sacrament
in the Catholic Church. According to the Catechism of the Catholic Church
(CCC §2754), the Sacraments are efficacious signs of grace, instituted by
Christ and entrusted to the Church, by which divine life is dispensed to us.
The Eucharist, specifically, is one of the seven sacraments of the Church.

In CCC §2772, it is stated that the Sacrament of the Eucharist is also known
as the "Sacrament of the Body and Blood of Christ."
...
Sources: 5 chunks | Abstained: False
```

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/chriscruz06/ragbench.git
cd ragbench
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env               # defaults work out of the box

# 3. Pull a local LLM via Ollama
ollama pull mistral

# 4. Download the corpus
#    - Douay-Rheims Bible → data/raw/bible/douay_rheims_raw.txt
#      (available from Project Gutenberg: gutenberg.org/ebooks/8300)
#    - Catechism of the Catholic Church → data/raw/catechism/ccc_raw.txt
#      (available from vatican.va)

# 5. Preprocess the raw texts
python -m ingestion.preprocess_bible
python -m ingestion.preprocess_ccc

# 6. Ingest (chunk → embed → store)
python pipeline.py ingest

# 7. Query
python pipeline.py
```

## Project Structure

```
ragbench/
├── ingestion/              # Document loading, preprocessing, chunking, embedding
│   ├── loader.py           # Unified loader for txt, pdf, md, json
│   ├── chunker.py          # Fixed, sentence, and semantic chunking strategies
│   ├── embedder.py         # Sentence-transformer embedding + ChromaDB storage
│   ├── preprocess_bible.py # Gutenberg cleanup → per-book text files
│   └── preprocess_ccc.py   # CCC parser → numbered paragraphs with scripture refs
├── retrieval/
│   └── search.py           # Vector search with similarity threshold abstention
├── generation/
│   └── generate.py         # LLM prompting with faithfulness guardrails
├── eval/                   # Metrics, test sets, scoring pipeline (Phase 2)
├── api/
│   └── app.py              # FastAPI backend
├── frontend/               # Web UI
├── notebooks/              # Ablation experiments (Phase 3)
├── config/
│   └── settings.py         # Centralized config via Pydantic
├── data/
│   ├── raw/                # Source documents (not tracked)
│   └── processed/          # Cleaned + structured corpus (not tracked)
├── pipeline.py             # End-to-end orchestrator
├── requirements.txt
└── .env.example
```

## Key Design Decisions

**Similarity threshold abstention**  If no retrieved chunk scores above the configured threshold (default: 0.3), the system declines to answer rather than hallucinating. This is a deliberate tradeoff: lower recall in exchange for higher faithfulness to the text (don't want to accidently recreate some ancient heresy).

**"Curator, not theologian" prompt design**  The system prompt constrains the LLM to organize and present retrieved content, not to synthesize or interpret independently. This is the primary defense against unfaithful generation - the model cites CCC paragraphs and Scripture, it doesn't do theology.

**Citation enforcement**  Every claim in a response must reference a specific CCC paragraph (e.g., CCC §1234) or Scripture verse (e.g., Romans 5:8). Responses without citations indicate retrieval or generation failure.

**Pluggable chunking strategies**  Fixed-size, sentence-level, and semantic chunking are implemented behind a common interface, enabling controlled ablation experiments in Phase 3.

**Pluggable LLM backend**  The generation layer supports both local models via Ollama (zero cost) and cloud models via OpenAI, configurable through environment variables.

## Hallucination Defense Layers

| Layer | Mechanism | Description |
|-------|-----------|-------------|
| 1 | System prompt | "Curator, not theologian", answer only from context |
| 2 | Citation enforcement | Every claim must cite CCC § or Scripture |
| 3 | Retrieval-heavy generation | Responses are mostly direct quotation with connective tissue |
| 4 | Similarity threshold | Abstain when context is insufficient |
| 5 | Eval framework | Faithfulness metric quantifies grounding (Phase 2) |

## Corpus

| Source | Documents | Chunks | Type |
|--------|-----------|--------|------|
| Douay-Rheims Bible | 47 books | 2,777 | Scripture |
| Catechism of the Catholic Church | 2,934 paragraphs | 4,754 | Doctrine |
| **Total** | **2,981** | **7,531** | |

## Roadmap

- [x] **Phase 1**  RAG core pipeline (ingest → retrieve → generate)
- [ ] **Phase 2**  Evaluation framework (Precision@K, Recall@K, MRR, Faithfulness, BLEU/ROUGE)
- [ ] **Phase 3**  Ablation experiments (chunking strategy comparison) + deployment

## Tech Stack

| Component | Tool |
|-----------|------|
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector Store | ChromaDB |
| LLM | Ollama (Mistral 7B) / OpenAI (gpt-4o-mini) |
| API | FastAPI |
| Eval | RAGAS, ROUGE, BLEU |
| Config | Pydantic Settings |
| Visualization | Plotly, Pandas |

## License

MIT