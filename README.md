# RAG-Eval

A production-style RAG pipeline built layer by layer, from PDF ingestion to answer evaluation.

It runs on real-world PDF data and compares 3 chunking strategies, 5 indexes (FAISS flat, IVF and HNSW, BM25 and Qdrant), dense, BM25 and hybrid retrieval, 4 query types (original, rewritten, multi-query and HyDE) and cross-encoder reranking. Both the retrieved chunks and the generated answers are scored. All LLM calls go through one LLM layer, and Claude, Gemini, ChatGPT and a local Ollama model are each used for a different task.

### Key results

5,000 documents, 200 questions: 150 for validation (tuning) and 50 for test (used once).

| Set | Chunk recall | Faithfulness | Correctness | Relevance |
|---|---|---|---|---|
| Validation (150) | 0.93 | 0.89 | 0.61 | 0.57 |
| Test (50) | 0.88 | 0.89 | 0.55 | 0.64 |

- **Best setup:** sentence chunks, hybrid retrieval (FAISS + BM25, fused with RRF), top 20, no reranking.
- **Prompt tuning** raised correctness from 0.55 to 0.61 and halved "I don't know" answers.
- **Cost:** $4.88 to generate the 200-question eval set, $9.16 for the three tuning runs on the validation set, and $0.78 for the test run. Cached re-runs cost nothing.
## Architecture

The pipeline has two phases. The build phase turns the documents into searchable indexes. The eval phase asks questions, retrieves chunks, generates answers and scores them. Each layer is its own package under `src/` and saves its output, so a re-run loads what has not changed instead of computing it again.

```mermaid
flowchart LR
    subgraph BUILD["Build phase"]
        direction TB
        L1["l1_ingestion<br/>parse PDFs"] --> L2["l2_chunking<br/>fixed / sentence / semantic"]
        L2 --> L3["l3_embedding<br/>chunk embeddings"]
        L3 --> L4["l4_indexing<br/>FAISS / BM25 / Qdrant"]
    end

    subgraph EVAL["Eval phase"]
        direction TB
        QNA["qna<br/>eval question set"] --> L5["l5_query<br/>rewritten / multi / HyDE"]
        L5 --> L6["l6_retrieval<br/>dense / BM25 / hybrid + rerank"]
        L6 --> RE["l10_evaluation<br/>score retrieval, keep best setups"]
        RE --> L7["l7_context<br/>build context"]
        L7 --> L8["l8_prompt<br/>build prompts"]
        L8 --> L9["l9_generation<br/>answer in one batch"]
        L9 --> AE["l10_evaluation<br/>score answers with RAGAS"]
    end

    AN["analysis<br/>summaries, diagnosis,<br/>run comparison, report"]
    LLM(["llm<br/>Claude / Gemini / ChatGPT / Ollama"])

    L4 --> L6
    AE --> AN
    LLM -.-> QNA
    LLM -.-> L5
    LLM -.-> L9
    LLM -.-> AE

    classDef build fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef eval fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef out fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef llm fill:#f3e8ff,stroke:#9333ea,color:#581c87
    class L1,L2,L3,L4 build
    class QNA,L5,L6,RE,L7,L8,L9,AE eval
    class AN out
    class LLM llm
```

There are three entry points, all reading the same `config.yaml`:

| Script | Runs |
|---|---|
| `scripts/run.sh` | the whole pipeline (`src/pipeline.py`) |
| `scripts/build.sh` | the build phase only (`src/build.py`) |
| `scripts/eval.sh` | the eval phase only (`src/eval.py`) |
