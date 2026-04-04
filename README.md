# RagBench

> A RAG (Retrieval-Augmented Generation) pipeline with a built-in evaluation framework for benchmarking retrieval and generation quality over Catholic theological texts.

RagBench ingests the **Catechism of the Catholic Church (CCC)** and **Sacred Scripture (Douay-Rheims Bible)**, embeds them into a local vector store, and answers theological questions with grounded, cited responses. An evaluation framework measures retrieval precision, generation faithfulness, and end-to-end accuracy across different pipeline configurations.

The system is designed around a core principle: **the LLM is a curator, not a theologian.** Every response must be grounded in retrieved source material, every claim must cite a specific CCC paragraph or Scripture verse, and the system explicitly abstains when retrieved context is insufficient, trading recall for faithfulness. If the goal was making a model to spout some random heresies, it would've been made to do so explicitly.

## Architecture
(This diagram has slowly become more messy and messy, hopefully it will survive the project)

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Raw Corpus     │────▶│    Ingestion     │────▶│    ChromaDB     │
│   Bible + CCC   │     │  Preprocess →    │     │   7,500+ vecs   │
│                  │     │  Chunk → Embed   │     │   (cosine)      │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          │ retrieve top-50
                                                          │ + cross-encoder rerank
┌─────────────────┐     ┌──────────────────┐     ┌────────▼────────┐
│   Answer +       │◀────│   Generation     │◀────│   Retrieval     │
│   CCC §1234      │     │   System prompt  │     │   Two-stage:    │
│   Romans 5:8     │     │   guardrails     │     │   bi-encoder +  │
└────────┬────────┘     └──────────────────┘     │   cross-encoder │
         │                                        └─────────────────┘
         ▼
┌──────────────────┐
│   Eval Framework  │  Precision@K · Recall@K · MRR · Token F1
│                   │  BLEU · ROUGE-L · Source Coverage
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

# 8. Evaluate
python -m eval.runner --tag baseline           # full eval (needs Ollama)
python -m eval.runner --retrieval-only --tag baseline  # retrieval only (fast)
python -m eval.runner --dry-run                # test the harness
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
│   └── search.py           # Two-stage retrieval: bi-encoder → cross-encoder rerank
├── generation/
│   └── generate.py         # LLM prompting with faithfulness guardrails
├── eval/                   # Evaluation framework
│   ├── metrics.py          # Precision@K, Recall@K, MRR, BLEU, ROUGE-L, Token F1
│   ├── runner.py           # Test set runner with CLI, outputs timestamped JSON reports
│   ├── test_metrics.py     # Unit tests for metrics (11 tests)
│   ├── test_set.json       # 10 gold-standard Q&A pairs across 6 topics
│   └── results/            # Timestamped eval reports (JSON)
├── api/
│   └── app.py              # FastAPI backend
├── config/
│   └── settings.py         # Centralized config via Pydantic
├── data/
│   ├── raw/                # Source documents (not tracked)
│   └── processed/          # Cleaned + structured corpus (not tracked)
├── pipeline.py             # End-to-end orchestrator
├── test_pipeline.py        # Smoke tests for each pipeline component
├── requirements.txt
└── .env.example
```

## Key Design Decisions

**Two-stage retrieval**  The retrieval pipeline uses a bi-encoder (BAAI/bge-base-en-v1.5) to pull 50 candidates from ChromaDB, then a cross-encoder (ms-marco-MiniLM-L-6-v2) reranks them by query relevance. This gives both speed (bi-encoder) and precision (cross-encoder). The reranker threshold filters out low-confidence chunks before they reach the LLM.

**Similarity threshold abstention**  If no retrieved chunk scores above the configured threshold, the system declines to answer rather than hallucinating. This is a deliberate tradeoff: lower recall in exchange for higher faithfulness to the text (don't want to accidentally recreate some ancient heresy).

**"Curator, not theologian" prompt design**  The system prompt constrains the LLM to organize and present retrieved content, and not to synthesize or interpret independently. This is the most active barrier against hallucinating its way to some heresy; it doesn't do any actual theology.

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
| 5 | Eval framework | Faithfulness metrics quantify grounding |

## Baseline Eval Results

First full pipeline evaluation (10 questions, 6 topics):

| Metric | Score | Notes |
|--------|-------|-------|
| **Retrieval** | | |
| Precision@K | 0.140 | Strict match against gold-standard paragraphs |
| Recall@K | 0.150 | Retrieval finds relevant content, often different §'s than gold set |
| MRR | 0.270 | First gold-standard hit typically at rank 2–4 |
| **Generation** | | |
| BLEU | 0.035 | Expected low - Mistral paraphrases rather than echoing gold answers |
| ROUGE-L F1 | 0.190 | Moderate subsequence overlap with expected answers |
| Token F1 | 0.261 | ~26% keyword overlap, reasonable given different source paragraphs |
| Source Coverage | 0.117 | Model cites retrieved paragraphs, not the specific gold-standard ones |

These scores reflect a strict evaluation; the retriever surfaces topically relevant CCC paragraphs, but often different sections than the gold-standard expected sources. The generation metrics are bounded by this retrieval gap. The model answers from what it retrieves, which is on-topic but not the exact paragraphs the test set was written against. Expanding the test set's acceptable sources and adding more test questions are planned improvements.

## Corpus

| Source | Documents | Chunks | Type |
|--------|-----------|--------|------|
| Douay-Rheims Bible | 47 books | 2,777 | Scripture |
| Catechism of the Catholic Church | 2,789 paragraphs | 4,754 | Doctrine |
| **Total** | **2,836** | **7,531** | |

## Roadmap

- [x] **Phase 1** - RAG core pipeline (ingest → retrieve → generate)
- [x] **Phase 2** - Evaluation framework (Precision@K, Recall@K, MRR, BLEU, ROUGE-L, Token F1, Source Coverage)
- [X] **Phase 2.5** - Eval report & comparison tooling
- [ ] **Phase 3** - Ablation experiments (chunking strategy comparison) + deployment

## Tech Stack

| Component | Tool |
|-----------|------|
| Embeddings | `BAAI/bge-base-en-v1.5` via sentence-transformers |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Vector Store | ChromaDB (cosine similarity) |
| LLM | Ollama (Mistral 7B) / OpenAI (gpt-4o-mini) |
| API | FastAPI |
| Eval | Custom metrics (Precision@K, Recall@K, MRR, BLEU, ROUGE-L, Token F1) |
| Config | Pydantic Settings |

## License

MIT
