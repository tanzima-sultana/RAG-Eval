# RAG-Eval

A production-style RAG pipeline built layer by layer, from PDF ingestion to answer evaluation.

It runs on real-world PDF data. It compares 3 chunking strategies, 5 indexes (FAISS flat, IVF and HNSW, BM25 and Qdrant), 3 retrieval types (dense, BM25 and hybrid), 4 query types (original, rewritten, multi-query and HyDE) and cross-encoder reranking. Both the retrieved chunks and the generated answers are scored. All LLM calls go through one LLM layer. Claude, Gemini, ChatGPT and a local Ollama model each handle a different task.

### Key results

5,000 documents, 200 questions: 150 for validation (tuning) and 50 for test (used once).

| Set | Chunk recall | Faithfulness | Correctness | Relevance |
|---|---|---|---|---|
| Validation (150) | 0.93 | 0.89 | 0.61 | 0.57 |
| Test (50) | 0.88 | 0.89 | 0.55 | 0.64 |

- **Best setup:** sentence chunks, hybrid retrieval (FAISS + BM25, fused with RRF), top 20, no reranking.
- **Prompt tuning** raised correctness from 0.55 to 0.61 and halved "I don't know" answers.
- **Cost:** The 200-question eval set cost $4.88 to generate. The three tuning runs on the validation set cost $9.16. The test run cost $0.78. Cached re-runs cost nothing.

## Architecture

<p align="center"><img src="docs/architecture.svg" alt="Architecture: build, question, eval and analysis phases" width="820"></p>

## Tech stack

| Layer | Tools |
|---|---|
| **Core** | Python 3.10, PyYAML config, Pydantic, NumPy, PyTorch, pytest (436 tests) |
| **Ingestion** | `doc_ingest` (PDF parsing with pypdfium2, MinHash deduplication), Hugging Face `datasets` |
| **Chunking** | Fixed, sentence and semantic chunkers implemented, NLTK for sentence splitting |
| **Embedding** | sentence-transformers (`all-MiniLM-L6-v2`) |
| **Indexing** | FAISS (flat, IVF, HNSW), `rank-bm25` for BM25, Qdrant in Docker |
| **Eval question set** | RAGAS test set generation with Claude Sonnet and Haiku |
| **Query transformation** | Rewritten, multi-query and HyDE implemented, with Llama 3.1 8B on Ollama (local) |
| **Retrieval** | Dense, BM25 and hybrid (RRF) retrieval implemented, cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) |
| **Generation** | Gemini Flash-Lite through its batch API |
| **Evaluation** | Retrieval metrics implemented (hit, precision, recall, MRR, nDCG), RAGAS answer scores with GPT-4.1-mini as judge |
| **LLM layer** | Implemented over the Anthropic, OpenAI and Google SDKs and Ollama, which lets each task use any of the four |
| **Comparison** | LangChain, as an optional framework for chunking, embedding, indexing and retrieval |

**Abbreviations:**
NLTK: Natural Language Toolkit ·
FAISS: Facebook AI Similarity Search ·
IVF: inverted file index ·
HNSW: Hierarchical Navigable Small World ·
BM25: Best Matching 25 ·
HyDE: Hypothetical Document Embeddings ·
RRF: reciprocal rank fusion ·
MRR: mean reciprocal rank ·
nDCG: normalized discounted cumulative gain ·
RAGAS: Retrieval Augmented Generation Assessment ·
LLM: large language model ·
SDK: software development kit

## Design decisions

**1. One config file:**
Every setting lives in `config.yaml`. The build stage reads the build part of the file. The eval stage reads both
the build and eval parts, so that it always uses the settings the indexes were built with. Any setting can be
overridden on the command line for one run. A missing or misspelled setting stops the run.
*Trade-off:* Every setting must be in the file, even if it rarely changes.

**2. Separate stages that hand off through files:**
Build, question, evaluation and analysis each have their own entry script. Each stage saves its output for the next
stage: documents, chunks, embeddings, indexes, eval questions, queries, answers, scores and eval results. Because of
this, any stage can be rerun alone. For example, a new report can be made without rerunning the evaluation. The LLM
layer and the ingestion library are kept apart from the rest of the code, so that either one can be swapped or reused.
*Trade-off:* More files are kept on disk.

**3. Cache keys built from the settings:**
Each layer saves its output under a name that is made from the settings that change it. For example, chunk files are
named by chunk type, model and sizes. Answers are keyed by their prompt. Judge scores are keyed by the question,
answer, context and reference. When nothing has changed, a rerun loads everything and makes no LLM call. When one
setting changes, only the layers that depend on it are computed again. Each file is first written to a temporary file
and then renamed, so that a crash never leaves a half-written cache.
*Trade-off:* If a code change alters the output without changing a setting, its cache must be deleted by hand.

**4. A different LLM provider for each task:**
Claude writes the eval questions. A local Llama model transforms the queries. Gemini writes the answers.
GPT-4.1-mini judges the answers. Because each task uses a different provider, no model grades its own work. All
providers go through one LLM layer, which lets a task switch provider in the config. Paid calls only run when
`ALLOW_PAID_API=1` is set.
*Trade-off:* Four providers must be set up, with four API keys.

**5. A cheaper model for the knowledge graph:**
RAGAS first builds a knowledge graph from the documents. It then writes the questions from that graph. Building the
graph needs a lot of tokens, because whole documents and chunks are sent to the model for simple extraction. Claude
Haiku does this step, because it costs 20 times less per token than Claude Sonnet. Because the questions and
reference answers set the quality of the benchmark, Claude Sonnet writes them. For the 200-question set, Haiku made
75% of the calls and read 84% of the input tokens. It cost $0.89 of the $4.88 total. The same run on Sonnet alone
would have cost about $21.86.
*Trade-off:* Two models must be configured for one step.

**6. Batch API for generation:**
Answers are sent to Gemini as one batch, which costs half the normal price. The batch id is saved. If a run stops,
the next run picks up the same batch instead of paying again.
*Trade-off:* A batch can take minutes to hours instead of seconds.

**7. A validation set and a test set:**
The 200 eval questions are split with a fixed seed into 150 for validation and 50 for test. Every setting was chosen
on the validation set. The test set was run only once, at the end, for the final numbers.
*Trade-off:* With only 50 test questions, the final numbers are rough (about ±0.07 on correctness).

## Evaluation methodology

The numbers and models below are the ones used for the results in this README. All of them are settings in
`config.yaml` and can be changed for each run.

**Eval set:**
The eval set is generated once by RAGAS and then cached. RAGAS needs documents with more than 500 tokens, so 100
such documents are picked with a fixed seed. Claude Haiku builds a knowledge graph from them. Claude Sonnet then
writes 200 questions, each with a reference answer and its source passages. Half the questions are single-hop, while
the other half are multi-hop and need several passages. RAGAS writes them in the voice of made-up users, so some are
casual or misspelled like real questions. The 200 questions came from 54 of the 100 documents. The set is split with
a fixed seed into 150 questions for validation and 50 for test.

**Retrieval metrics:**
Each retrieval result is scored on its top 20 chunks, or its top 5 after reranking. The scores are given at two
levels:
- *Chunk level:* A chunk is relevant if it comes from a source document and shares at least half its words with a
  source passage. Recall is the share of source passages found.
- *Doc level:* A chunk is relevant if it comes from a source document. Recall is the share of source documents found.

Both levels report hit, precision, recall, MRR and nDCG. The chunk level is the main measure, because a chunk from
the right document can still miss the passage that the answer needs.

**Answer metrics:**
RAGAS scores each answer from 0 to 1 on three measures:
- *Correctness:* how well the answer matches the reference answer.
- *Faithfulness:* how much of the answer the context supports.
- *Relevance:* how well the answer addresses the question.

An "I don't know" answer scores 0 on all three without a judge call. When the judge cannot give a score, for example
because a content filter blocks its reply, the score is left out of the average and counted as unscored.

**Judge:**
GPT-4.1-mini is the judge. It comes from a different provider than the question writer (Claude) and the answer
writer (Gemini), so no model grades its own work. The runs showed three limits of the judge:
- *One reference answer:* Some questions have more than one valid answer, although the reference gives only one. A
  question about a university schedule, for example, names no year. The answer gave the 2020 dates while the
  reference gave the 2019 dates, so a faithful answer scored only 0.21.
- *Partial answers:* An answer that says the context does not cover part of the question gets relevance 0.
- *Noise:* Small differences between runs, like 0.612 and 0.613, mean nothing.

All runs use the same judge. The report marks any run that was judged by a different model.


<!--
README plan, agreed 2026-10-16. Sections 1 to 5 are done; write the rest in this order, then remove this note.

3. Tech stack, grouped:
   - Retrieval and embeddings: FAISS, BM25, Qdrant, sentence-transformers, the cross-encoder
   - LLMs: Claude, Gemini, ChatGPT and Ollama through the project's own llm layer
   - Evaluation: RAGAS
   - Core: Python, PyYAML, pytest, Docker, doc_ingest
4. Design decisions: caching keys, batch APIs, provider separation, validation/test discipline, config file,
   layer isolation
5. Evaluation methodology:
   - Eval set: RAGAS-generated, 200 questions, 150 validation / 50 test
   - Retrieval metrics: chunk and doc level
   - Answer metrics
   - Judge choice and bias
6. Results and analysis:
   - Retrieval comparison: chunk types x indexes x retrieval types, and reranking
   - Experiment table: baseline -> exp 1 -> exp 2 -> test, with what changed and its effect
   - Diagnosis: hit vs miss, multi-hop vs single-hop, worst answers
   - Cost and time per run, including build times and sizes
7. Problems faced and fixes: issue -> cause -> fix
8. Setup and how to run: prerequisites, .env, Qdrant in Docker, config.yaml, the scripts, ALLOW_PAID_API
   (optionally a short VHS terminal GIF)
9. Known issues and limitations: synthetic eval set, single reference, small test set, judge noise
10. Future work / what production would add: FastAPI serving with load tests, auth, monitoring, guardrails,
    the red-teaming project

Dropped on purpose: project structure (the architecture diagram covers it) and a link to the notebook
(docs/notebook.md stays local only).
-->
