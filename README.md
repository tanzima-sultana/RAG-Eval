# RAG-Eval

A production-style RAG pipeline built layer by layer, from document ingestion to answer evaluation.

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
- **LangChain comparison:** The same pipeline was also run with LangChain on this dataset of clean, text-based PDFs.
  Both scored about the same. Built-in was slightly higher on retrieval quality and answer correctness. With
  only 5,000 documents, a gap of 2% to 3% is too small to call one system better.

## Architecture

<p align="center"><img src="docs/architecture.svg" alt="Architecture: build, question, eval and analysis phases" width="820"></p>

## Tech stack

| Layer | Tools |
|---|---|
| **Core** | Python 3.10, PyYAML config, Pydantic, NumPy, PyTorch, pytest (436 tests) |
| **Ingestion** | [`doc_ingest`](https://github.com/tanzima-sultana/Document_Ingestion_Engine) (built on LangChain's document parsers, text cleanup, deduplication) |
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

**8. Built-in code with LangChain as an option:**
Chunking, embedding, indexing and retrieval are implemented in the project's own code. Each of them also has a
LangChain path, which one setting turns on. The LangChain path converts its output into the project's own records,
so every later step works the same way on both paths. Each path keeps its own cache, so their outputs never mix.
This setup was used to compare the built-in code with LangChain on the same documents.
*Trade-off:* Two code paths must be kept in step.

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

### Built-in vs LangChain

Both paths were run end to end on the same 5,000 documents with the same settings. LangChain reads the raw PDFs with
its own loader, so each path has its own documents, eval questions and queries. Each path is scored on its own 150
validation questions.

**Retrieval**

The table shows the setups that are in the top 5 of both paths. All use sentence chunks, hybrid search, the original
question and the top 20 chunks, without reranking.

| Setup | Chunk hit (B / L) | Chunk recall (B / L) | Chunk MRR (B / L) | Doc hit (B / L) |
|---|---|---|---|---|
| FAISS flat + BM25 | 0.940 / 0.927 | 0.927 / 0.903 | 0.718 / 0.706 | 0.987 / 0.960 |
| Qdrant + BM25 | 0.940 / 0.927 | 0.927 / 0.903 | 0.716 / 0.706 | 0.987 / 0.960 |
| FAISS HNSW + BM25 | 0.940 / 0.920 | 0.923 / 0.897 | 0.705 / 0.672 | 0.980 / 0.960 |

B is built-in and L is LangChain.

- **The same kind of setup wins on both paths.** Hybrid search with the original question and the top 20 chunks
  scores best, and reranking lowers the scores on both.
- **The best chunking differs.** Built-in scores best with sentence chunks. LangChain scores best with fixed chunks,
  with a chunk recall of 0.913 against built-in's 0.927.
- **Built-in is 1.4% to 2.9% higher on every measure** on these shared setups. Chunk hit is 1.4% to 2.2% higher, and
  chunk recall is 2.7% to 2.9% higher. Doc hit is 2.1% to 2.8% higher. MRR is up to 4.9% higher with HNSW.

**Generated answers**

Each path answered its own 150 validation questions with its best setup. Both used Gemini Flash-Lite, the same prompt
and the same judge.

| | Built-in | LangChain |
|---|---|---|
| Setup | sentence, FAISS flat + BM25 | fixed, Qdrant + BM25 |
| I don't know | 0.067 | 0.053 |
| Correctness | 0.612 | 0.600 |
| Faithfulness | 0.894 | 0.898 |
| Relevance | 0.573 | 0.567 |
| Cost | $2.56 | $2.45 |

- **The answer scores are close.** Built-in is 2.0% higher on correctness and 1.1% higher on relevance. LangChain is
  0.4% higher on faithfulness.
- **LangChain says "I don't know" less often.** It gave 8 such answers against built-in's 10.
- **The small gap follows retrieval.** Built-in retrieval is 1.4% to 2.9% higher, and its answers are 1% to 2% higher.
- **One LangChain answer has no score.** The judge's reply ran over its output limit.

## Analysis

### Hit vs miss

A hit means that a source passage was among the retrieved chunks. A miss means that none was.

| Set | Source found | Questions | I don't know | Correctness | Faithfulness | Relevance |
|---|---|---|---|---|---|---|
| Validation | Yes | 141 | 0.043 | 0.626 | 0.916 | 0.582 |
| Validation | No | 9 | 0.444 | 0.391 | 0.556 | 0.429 |
| Test | Yes | 45 | 0.044 | 0.597 | 0.926 | 0.661 |
| Test | No | 5 | 0.400 | 0.122 | 0.570 | 0.419 |

- **Most lost points come after retrieval.** The source was found for 141 of 150 validation questions and 45 of 50
  test questions. Even with the source in the context, correctness is only about 0.6.
- **On a miss, "I don't know" is the right answer.** The model gives it for almost half of the misses.
- **The model was too cautious before the new prompt.** With the first prompt, 10.6% of the hits got "I don't know",
  although the source was in the context. The new prompt cut this to 4.3%.
- **So tuning focused on generation, not on the build.** Retrieval already finds the source for 94% of the
  questions. Even perfect retrieval would lift correctness only on the 9 misses, which is at most about 0.04 on the
  average. The prompt and the generation model were therefore the settings worth changing.

### Single-hop vs multi-hop

| Set | Question type | Questions | I don't know | Correctness | Faithfulness | Relevance |
|---|---|---|---|---|---|---|
| Validation | Single-hop | 73 | 0.068 | 0.643 | 0.892 | 0.606 |
| Validation | Multi-hop | 77 | 0.065 | 0.583 | 0.896 | 0.541 |
| Test | Single-hop | 27 | 0.148 | 0.467 | 0.809 | 0.626 |
| Test | Multi-hop | 23 | 0.000 | 0.647 | 0.986 | 0.650 |

- **On validation, multi-hop questions score lower**, because they need passages from several places.
- **The new prompt helped multi-hop questions the most.** It asks the model to answer each part of the question.
  Multi-hop correctness rose from 0.506 to 0.583, while single-hop correctness rose from 0.588 to 0.643.
- **On test, the order flips.** With only 23 and 27 questions per type, this is within the noise of a small set.

### Worst answers

Each run's report lists the 10 answers with the lowest correctness, next to their reference answers.

- **Many of the worst answers are not wrong.** In the baseline, 9 of the 10 worst answers had faithfulness 1.000, so
  everything they said was backed by the context. Their correctness was still only 0.21 to 0.27. The reason is that
  some questions have more than one valid answer in the documents, while the reference answer comes from a single
  passage. For example, one question asks which bills a child care center sponsored. The reference answer named
  AB 92. The model found a different passage and named AB 393 and SB 246. The documents support both answers, but the
  judge compares only with the reference, so it marked the model's answer wrong.
- **Correctness therefore has a ceiling below 1** on this eval set, even for perfect answers.
- **Some low scores come from partial answers.** In experiment 1, 4 of the 10 worst answers said that the context did
  not cover all or part of the question. All four got relevance 0, as the judge treats them as non-answers.

### Why the stronger model gave no gain

Experiment 2 replaced Gemini Flash-Lite with Gemini Flash, a stronger model that thinks before it answers.

| Generator | Correctness | Correctness on hits | Faithfulness | Relevance | Generation cost |
|---|---|---|---|---|---|
| Gemini Flash-Lite (experiment 1) | 0.612 | 0.626 | 0.894 | 0.573 | $0.11 |
| Gemini Flash (experiment 2) | 0.613 | 0.629 | 0.889 | 0.532 | $0.64 |

- **The scores stayed the same.** The small differences are within the noise between runs.
- **The cost rose about 6 times**, mostly for thinking tokens. The stronger model wrote 184,531 output tokens for the
  150 prompts, while Flash-Lite wrote only 19,305.
- **The limit is in the eval set and the judge, not in the generator.** The answers were already faithful to their
  context. The points they lose come from the single reference answer and from partial answers, which a better
  model cannot fix.
- **The stricter prompt was tested in the same run.** So the effects of the new prompt and the new model cannot be
  told apart.

## Problems faced and fixes

**1. Eval questions linked to no document:**
*Problem:* 11 of the 200 eval questions were linked to no source document.
*Cause:* RAGAS sometimes joins sections from different parts of a document into one passage. The full passage is
then not found in the document text.
*Fix:* Only the first 80 characters of each passage are matched. The saved eval set was repaired without any new API
call, and all 200 questions are now linked.

**2. A large and slow embedding cache:**
*Problem:* The first embedding cache was about 5 times larger than the vectors it held and slow to load. Its vectors
also came back as float64 instead of float32.
*Cause:* The cache was JSON, which writes every number as text.
*Fix:* The cache now uses NumPy's binary `.npz` format. It is 5.1 times smaller, loads in about half the time and
keeps the float32 type.

**3. A blocked judge reply stopped the run:**
*Problem:* The judge's content filter blocked its reply for one answer, which stopped the whole run before the
analysis.
*Cause:* The code treated a missing score as an error.
*Fix:* A score the judge cannot give is now stored as empty, counted as unscored and left out of the average. The
other scores were already cached, so the rerun sent only that one answer.

**4. Tiny extra chunks:**
*Problem:* The chunkers made extra chunks, some only 1 token long.
*Cause:* The splitter started one more window after the last token, so the last tokens were chunked twice.
*Fix:* The splitter now stops at the last token. On 5,000 documents, fixed chunks went from 92,366 to 91,670, and the
smallest fixed chunk went from 1 token to 17.

**5. Hybrid retrieval with nothing to fuse:**
*Problem:* The first test run stopped with an empty result.
*Cause:* Hybrid retrieval was asked for alone, so the dense and BM25 searches it fuses did not run.
*Fix:* Hybrid retrieval now runs its own dense and BM25 searches. An empty retrieval also stops the run right after
retrieval, before any paid call, so the failed try cost nothing.

**6. Thinking tokens missing from the cost:**
*Problem:* The cost of the stronger Gemini model was too low.
*Cause:* Gemini reports thinking tokens apart from the answer tokens, although it bills them as output.
*Fix:* The cost now adds thinking tokens to the output tokens.

## Setup and how to run

**Prerequisites:** Python 3.10, Docker for the Qdrant server, and Ollama for query transformation. API keys for
Claude, Gemini and ChatGPT are needed only for runs that call them.

1. **Install the document ingestion library**, `doc_ingest`, from its repository:

   ```bash
   pip install git+https://github.com/tanzima-sultana/Document_Ingestion_Engine.git
   ```

2. **Install the project's other requirements** and the NLTK sentence splitter data:

   ```bash
   pip install -r requirements.txt
   python -m nltk.downloader punkt punkt_tab
   ```

3. **Start Qdrant and Ollama:**

   ```bash
   docker run -d --name qdrant -p 6333:6333 -v "$(pwd)/qdrant:/qdrant/storage" qdrant/qdrant
   ollama pull llama3.1:8b
   ```

4. **Add the API keys** to a `.env` file at the repository root. The scripts load it on every run. `HF_TOKEN` is
   optional and only makes Hugging Face downloads faster.

   ```bash
   export HF_TOKEN=...
   export CLAUDE_API_KEY=...
   export GEMINI_API_KEY=...
   export CHATGPT_API_KEY=...
   ```

5. **Run a script.** Every setting is in `config.yaml`. Any setting can also be changed for one run by adding it
   after the script name.

   | Script | What it runs |
   |---|---|
   | `bash scripts/pipeline.sh` | The whole pipeline: build, question, evaluation and analysis |
   | `bash scripts/build.sh` | Ingestion, chunking, embedding and indexing |
   | `bash scripts/question.sh` | The eval question set and query transformation |
   | `bash scripts/eval.sh` | Retrieval, generation and scoring |
   | `bash scripts/analysis.sh` | The report and the comparison with earlier runs |

   The documents come from the `input_path` setting, a local file or folder. The `document_type` setting picks the
   type of documents to read, e.g. `pdf`. For example, this builds a small test set from the first 20 documents:

   ```bash
   bash scripts/build.sh --input_path outputs/pdfa-eng-wds/data/raw --dataset_size 20
   ```

**Paid calls:** The project never calls a paid API by accident. Claude, Gemini and ChatGPT are only called when the
run is started with `ALLOW_PAID_API=1`. Without it, a run that needs a paid call stops with a message. Cached runs
make no paid call, so they need neither the switch nor any API key.

```bash
ALLOW_PAID_API=1 bash scripts/pipeline.sh
```

**Tests:** `python -m pytest tests`

## Known issues and limitations

- **Synthetic eval set:** The questions and reference answers were written by an LLM, not by real users. They come
  from only 54 of the 5,000 documents, so retrieval is tested on a small part of the corpus.
- **One reference answer per question:** Some questions have more than one valid answer, which keeps correctness
  below 1 even for good answers (see [Worst answers](#worst-answers)).
- **Small test set:** With 50 test questions, the final numbers are rough, at about ±0.07 on correctness.
- **Judge noise:** The scores come from one LLM judge. They were not checked against human ratings, and small
  differences between runs mean nothing.
- **IVF index:** The IVF index searches only 1 of its 256 clusters, which keeps its recall low. Searching more
  clusters was not tried.

## Future work

**Next steps, by stage:**
- **Build:** Scale to 50,000 and 100,000 documents, and try other chunking settings.
- **Question:** Write more than one reference answer for questions that have several valid answers.

**What a production version would add:**
- **An API:** a FastAPI service that answers questions, with load tests for speed under many users.
- **An answer cache:** good answers to repeated questions are reused, so they cost nothing the second time.
- **Access control:** user login, and limits on which documents each user can search.
- **Monitoring:** logs of live questions, answer times and costs.
- **Guardrails:** checks on what goes into and comes out of the model.
- **Red-teaming:** tests that try to make the pipeline leak data or follow injected instructions.
