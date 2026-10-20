# RAG-Eval

A production-style RAG pipeline built layer by layer, from PDF ingestion to answer evaluation.

It runs on real-world PDF data and compares 3 chunking strategies, 5 indexes (FAISS flat, IVF and HNSW, BM25 and Qdrant), 3 retrieval types (dense, BM25 and hybrid), 4 query types (original, rewritten, multi-query and HyDE) and cross-encoder reranking. Both the retrieved chunks and the generated answers are scored. All LLM calls go through one LLM layer, where Claude, Gemini, ChatGPT and a local Ollama model each handle a different task.

### Key results

5,000 documents, 200 questions: 150 for validation (tuning) and 50 for test (used once).

| Set | Chunk recall | Faithfulness | Correctness | Relevance |
|---|---|---|---|---|
| Validation (150) | 0.93 | 0.89 | 0.61 | 0.57 |
| Test (50) | 0.88 | 0.89 | 0.55 | 0.64 |

- **Best setup:** sentence chunks, hybrid retrieval (FAISS + BM25, fused with RRF), top 20, no reranking.
- **Prompt tuning** raised correctness from 0.55 to 0.61 and halved "I don't know" answers.
- **Cost:** The 200-question eval set cost $4.88 to generate. The three tuning runs on the validation set cost $9.16 in total, while the test run cost $0.78. Cached re-runs cost nothing.

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
Every setting lives in `config.yaml`. The build stage reads only the build part of the file. The eval stage reads
both the build and eval parts, so that it always uses the settings the indexes were built with. Any setting can be
overridden on the command line for one run, while a missing or misspelled setting stops the run.
*Trade-off:* Every setting must be in the file, even if it rarely changes.

**2. Separate stages that hand off through files:**
Build, question, evaluation and analysis each have their own entry script. Each stage saves its output for the next
stage: documents, chunks, embeddings, indexes, eval questions, queries, answers, scores and eval results. Because of
this, any stage can be rerun alone, for example to make a new report without rerunning the evaluation. The LLM
layer and the ingestion library are kept apart from the rest of the code, so that either one can be swapped or reused.
*Trade-off:* More files are kept on disk.

**3. Cache keys built from the settings:**
Each layer saves its output under a name that is made from the settings that change it. Chunk files, for example,
are named by chunk type, model and sizes, while answers are keyed by their prompt and judge scores by the question,
answer, context and reference. When nothing has changed, a rerun loads everything and makes no LLM call. When one
setting changes, only the layers that depend on it are computed again. Each file is first written to a temporary file
and then renamed, so that a crash never leaves a half-written cache.
*Trade-off:* If a code change alters the output without changing a setting, its cache must be deleted by hand.

**4. A different LLM provider for each task:**
Each task uses a different provider: Claude for the eval questions, a local Llama model for query transformation,
Gemini for the answers and GPT-4.1-mini for judging. Because of this split, no model grades its own work. All
providers go through one LLM layer, which lets a task switch provider in the config. Paid calls only run when
`ALLOW_PAID_API=1` is set.
*Trade-off:* Four providers must be set up, with four API keys.

**5. A cheaper model for the knowledge graph:**
RAGAS first builds a knowledge graph from the documents and then writes the questions from that graph. Building the
graph needs a lot of tokens, because whole documents and chunks are sent to the model for simple extraction. Claude
Haiku does this step, since it costs 20 times less per token than Claude Sonnet. Because the questions and
reference answers set the quality of the benchmark, Claude Sonnet writes them. For the 200-question set, Haiku made
75% of the calls and read 84% of the input tokens but cost only $0.89 of the $4.88 total. The same run on Sonnet
alone would have cost about $21.86.
*Trade-off:* Two models must be configured for one step.

**6. Batch API for generation:**
Answers are sent to Gemini as one batch, which costs half the normal price. The batch id is saved, so that a run
which stops can pick up the same batch later instead of paying again.
*Trade-off:* A batch can take minutes to hours instead of seconds.

**7. A validation set and a test set:**
The 200 eval questions are split with a fixed seed into 150 for validation and 50 for test. Every setting was chosen
on the validation set, while the test set was run only once at the end for the final numbers.
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


## Results

### Retrieval

The top setups on the 150 validation questions, each scored on its top 20 chunks:

| Chunk | Retrieval | Index | Query | Chunk hit | Chunk recall | Chunk MRR | Doc hit |
|---|---|---|---|---|---|---|---|
| sentence | hybrid | FAISS flat + BM25 | original | 0.940 | 0.927 | 0.718 | 0.987 |
| sentence | hybrid | Qdrant + BM25 | original | 0.940 | 0.927 | 0.716 | 0.987 |
| sentence | hybrid | FAISS HNSW + BM25 | original | 0.940 | 0.923 | 0.705 | 0.980 |
| fixed | hybrid | FAISS flat + BM25 | original | 0.913 | 0.900 | 0.729 | 0.967 |
| semantic | hybrid | FAISS flat + BM25 | original | 0.913 | 0.903 | 0.723 | 0.973 |

Hybrid retrieval with the original question takes all five top places. The best setup finds a source passage for
94% of the questions and a source document for almost 99%. FAISS flat and Qdrant give almost the same results,
since both do exact search over the same vectors. The best setup was chosen for the answer runs.

Each table below changes one thing. Everything else stays at the best setup: sentence chunks, hybrid retrieval,
the original question and the top 20 chunks.

**Chunk type:**

| Chunk type | Chunk hit | Chunk recall | Chunk MRR | Doc hit |
|---|---|---|---|---|
| Sentence | 0.940 | 0.927 | 0.718 | 0.987 |
| Fixed | 0.913 | 0.900 | 0.729 | 0.967 |
| Semantic | 0.913 | 0.903 | 0.723 | 0.973 |

The three chunk types are close. Sentence chunks lead by about 4 of the 150 questions.

**Retrieval type:**

| Retrieval type | Chunk hit | Chunk recall | Chunk MRR | Doc hit |
|---|---|---|---|---|
| Hybrid | 0.940 | 0.927 | 0.718 | 0.987 |
| Dense | 0.853 | 0.837 | 0.630 | 0.900 |
| BM25 | 0.853 | 0.820 | 0.702 | 0.907 |

Hybrid finds about 9 points more than either retriever alone, because dense and BM25 find different passages.

**Query type:**

| Query type | Chunk hit | Chunk recall | Chunk MRR | Doc hit |
|---|---|---|---|---|
| Original | 0.940 | 0.927 | 0.718 | 0.987 |
| Rewritten | 0.907 | 0.887 | 0.681 | 0.967 |
| Multi-query | 0.907 | 0.900 | 0.681 | 0.953 |

The original question works best for hybrid retrieval. HyDE is not shown, since it can only be searched by dense
retrieval.

**Reranking:**

| Reranking | Chunks kept | Chunk hit | Chunk recall | Chunk MRR | Doc hit |
|---|---|---|---|---|---|
| No | 20 | 0.940 | 0.927 | 0.718 | 0.987 |
| Yes | 5 | 0.853 | 0.837 | 0.768 | 0.947 |

Reranking puts the right chunk higher. Because it keeps only 5 chunks, it misses more passages.

**From retrieval to generation:**
Retrieval gave 27,900 results: 186 setups for each of the 150 validation questions. Each result sent on costs LLM
calls for generation and judging, so only a filtered list goes to the generation stage:
- Each setup is scored by its average chunk recall, with MRR breaking ties.
- The best plain setup and the best reranked setup are kept. They are ranked apart, because a plain setup keeps 20
  chunks and a reranked one only 5.
- The 150 results of each kept setup go on, which are 300 results instead of 27,900.

### Generated answers

| Run | Questions | What changed | I don't know | Correctness | Faithfulness | Relevance | Cost |
|---|---|---|---|---|---|---|---|
| Baseline | 150 validation | First prompt, Gemini Flash-Lite | 0.127 | 0.546 | 0.853 | 0.561 | $3.61 |
| Experiment 1 | 150 validation | New prompt, no reranking | 0.067 | 0.612 | 0.894 | 0.573 | $2.56 |
| Experiment 2 | 150 validation | Stricter prompt, Gemini Flash | 0.073 | 0.613 | 0.889 | 0.532 | $2.99 |
| **Test** | **50 test** | **Experiment 1 setup** | **0.080** | **0.550** | **0.891** | **0.637** | **$0.78** |

- **Baseline:** It answered 300 questions, 150 for each of the two setups. The table shows the plain setup. The
  reranked setup with the top 5 chunks scored lower on every measure, for example 0.463 correctness against 0.546.
- **Experiment 1:** Reranking was dropped, so only the plain setup's 150 questions were answered. The new prompt asks
  for complete answers that cover each part of the question. It also asks for a partial answer instead of "I don't
  know" when the context holds part of the answer. Correctness rose from 0.546 to 0.612, while "I don't know" answers
  fell by half.
- **Experiment 2:** A stronger model that thinks before it answers kept the same scores, although generation cost
  about 6 times as much. So experiment 1 was kept as the final setup.
- **Test:** The 50 test questions were run once with the experiment 1 setup. Correctness is lower than on validation,
  while faithfulness is the same and relevance is higher. With only 50 questions, these differences are within noise.

### Cost and time

**Build:** The build runs once on 5,000 documents. It runs locally, so it costs nothing.

| Stage | Time | Disk |
|---|---|---|
| Ingestion | about 4 min | 2.3 GB of PDFs, 99 MB of parsed text |
| Chunking | 42 s (fixed), 46 s (sentence), 5.4 min (semantic) | 317 MB |
| Embedding | about 2.2 to 2.5 min per chunk type | 486 MB |
| Indexing | under 1 min per chunk type, for all five indexes | 1.4 GB for FAISS, 316 MB for BM25 |

**Paid runs:**

| Run | Cost | Time |
|---|---|---|
| Eval set (200 questions) | $4.88 | about 55 min |
| Baseline (300 answers) | $3.61 | 33 min |
| Experiment 1 (150 answers) | $2.56 | not recorded |
| Experiment 2 (150 answers) | $2.99 | 23 min |
| Test (50 answers) | $0.78 | 10 min |

- Most of the cost of each answer run is the judge. In experiment 1, judging cost $2.44 while generation cost only
  $0.11.
- Semantic chunking is the slowest build step, because it embeds every sentence to find where topics change.
- Every stage is cached, so a rerun with the same settings makes no LLM call and costs nothing.

## Analysis

<!--
README plan, agreed 2026-10-16. Sections 1 to 6 are done; write the rest in this order, then remove this note.

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
6. Results (agreed 2026-10-20), written one part at a time:
   - Retrieval: top 5-10 setups table, then small comparisons by chunk type, retrieval type, query type and
     reranking, then a short description. No full 186-row table.
   - Generated answers: baseline -> exp 1 -> exp 2 -> test table, with what changed and its effect
   - Cost and time: cost per run, build times and sizes
7. Analysis: hit vs miss, single-hop vs multi-hop, worst answers and the reference-answer ceiling (link to the
   judge limits in the methodology), why the stronger model gave no gain
8. Problems faced and fixes: issue -> cause -> fix
9. Setup and how to run: prerequisites, .env, Qdrant in Docker, config.yaml, the scripts, ALLOW_PAID_API
   (optionally a short VHS terminal GIF)
10. Known issues and limitations: synthetic eval set, single reference, small test set, judge noise
11. Future work / what production would add: FastAPI serving with load tests, auth, monitoring, guardrails,
    the red-teaming project

Dropped on purpose: project structure (the architecture diagram covers it) and a link to the notebook
(docs/notebook.md stays local only).
-->
