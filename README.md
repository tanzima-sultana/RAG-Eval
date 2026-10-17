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

The build phase turns the documents into searchable indexes. In parallel, the question phase creates the eval questions and the search queries made from them, with the LLM layer. The eval phase retrieves chunks for each query, generates answers and scores both, and the analysis phase turns the scores into a report. Each layer saves its output, so a re-run loads what has not changed instead of computing it again.

```mermaid
flowchart TB
    subgraph BUILD["Build phase"]
        direction TB
        L1["Ingestion<br/>parse PDFs"] --> L2["Chunking<br/>fixed / sentence / semantic"]
        L2 --> L3["Embedding<br/>chunk embeddings"]
        L3 --> L4["Indexing<br/>FAISS / BM25 / Qdrant"]
    end

    subgraph PREP["Question phase"]
        direction TB
        QNA["Eval question set<br/>questions and reference answers"] --> L5["Query transformation<br/>rewritten / multi-query / HyDE"]
        LLM(["LLM layer<br/>Claude / Gemini / ChatGPT / Ollama"])
    end

    subgraph EVAL["Eval phase"]
        direction TB
        L6["Retrieval<br/>dense / BM25 / hybrid + rerank"] --> RE["Retrieval scoring<br/>keep the best setups"]
        RE --> L7["Context<br/>join the retrieved chunks"]
        L7 --> L8["Prompt<br/>system and user prompt"]
        L8 --> L9["Generation<br/>answer in one batch"]
        L9 --> AE["Answer scoring<br/>RAGAS with a judge LLM"]
    end

    subgraph ANALYSIS["Analysis phase"]
        AN["Summaries, diagnosis,<br/>run comparison and report"]
    end

    L4 --> L6
    L5 --> L6
    AE --> AN
    LLM -.-> QNA
    LLM -.-> L5
    LLM -.-> L9
    LLM -.-> AE

    classDef build fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef prep fill:#f3e8ff,stroke:#9333ea,color:#581c87
    classDef eval fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef out fill:#fef3c7,stroke:#d97706,color:#78350f
    class L1,L2,L3,L4 build
    class QNA,L5,LLM prep
    class L6,RE,L7,L8,L9,AE eval
    class AN out

    style BUILD fill:#eff6ff,stroke:#2563eb,stroke-width:2px,color:#1d4ed8,font-weight:bold
    style PREP fill:#faf5ff,stroke:#9333ea,stroke-width:2px,color:#7e22ce,font-weight:bold
    style EVAL fill:#f0fdf4,stroke:#16a34a,stroke-width:2px,color:#15803d,font-weight:bold
    style ANALYSIS fill:#fffbeb,stroke:#d97706,stroke-width:2px,color:#b45309,font-weight:bold
```

There are three entry points, all reading the same `config.yaml`:

| Script | Runs |
|---|---|
| `scripts/run.sh` | the whole pipeline (`src/pipeline.py`) |
| `scripts/build.sh` | the build phase only (`src/build.py`) |
| `scripts/eval.sh` | the eval phase only (`src/eval.py`) |
