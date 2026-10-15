## Claude Code project setup

This project is developed with Claude Code. `CLAUDE.md` holds the code style, environment and project structure rules. The project's own skills are in `.claude/skills/` and its hook is in `.claude/settings.json`; both are loaded by every Claude Code session started in this repository.

### Skills

A skill is a set of instructions Claude follows for one kind of task. Claude uses it when a request matches the skill's description, or when it is called by name (`/<skill-name>`).

- **`git-commit`** — Writes a commit message, commits and pushes to the remote. Used whenever a commit is asked for.
- **`layer-readme-section`** — Writes or updates one pipeline layer's section of this README: an overview, one line per module from the structure in `CLAUDE.md`, and run results taken only from the logs.
- **`qdrant-docker`** — Checks that the Qdrant server is up on `localhost:6333` and, if not, starts Docker Desktop and the Qdrant container. Used before `scripts/run.sh` or anything that builds or queries the l4 Qdrant index. It never removes a container, image, volume or collection.
- **`paid-api-run-approval`** — Requires the user's explicit approval before every run that can call a paid LLM API. Paid runs through the project code must be started with `ALLOW_PAID_API=1` (e.g. `ALLOW_PAID_API=1 bash scripts/eval.sh`), because `llm.get_llm()` refuses to build a Claude LLM without it; direct API requests also count. Before the run Claude shows the command, the model, the settings that drive the cost, whether a cache file already exists and a cost estimate if one is known. One approval covers one run: a retry or a changed setting needs a new approval. After the run Claude reports the cost it printed.

### Hooks

- **Paid API guard** (`.claude/settings.json`, `.claude/hooks/paid-api-guard.sh`) — A `PreToolUse` hook on every Bash command Claude runs. If the command sets `ALLOW_PAID_API=1`, or reaches a paid API without `get_llm()` (the Anthropic, OpenAI or Google API hosts, the `anthropic` SDK, `langchain_anthropic` or `ChatAnthropic`), the hook returns `ask`, so Claude Code shows a permission prompt and the command runs only if the user approves it. Every other command runs without a prompt, including local Ollama runs and `eval.sh` with a cached eval set: without `ALLOW_PAID_API=1`, `get_llm()` stops with an error before any Claude call. The hook is set to `onFailure: block`, so if the hook itself fails the command is blocked.

## l2_chunking

### Overview

`l2_chunking` is the second pipeline layer. `src/rag.py` runs it right after `l1_ingestion` and passes it the ingested documents in memory, as a dict of `doc_id` to `Document`, together with an output folder `{--output_path}/{dataset_name}`, where `dataset_name` is the last part of `--input_path` (for example `outputs/pdfa-eng-wds`). The chunking settings travel as a `ChunkingArgs`: `--framework`, `--model_name`, `--chunk_type` (`fixed`, `sentence` or `semantic`), `--min_chunk_size`, `--max_chunk_size`, `--fixed_chunk_overlap` and `--semantic_threshold`. `Chunking.compute_chunks()` first validates these settings, then splits each document into `Chunk` records (`id`, `text`, `size` in tokens, `type`). Token counts come from the tokenizer of the SentenceTransformer model named by `--model_name`; `semantic` chunking also uses that model to embed every sentence. All chunks of one document are saved together as a JSON list in `{output folder}/chunks/{chunk_type}/{model_name}/{min}_{max}_{overlap}_{threshold}/{doc_id}.json`. The file is written to a temporary file first and then renamed into place, so it is either complete or absent. A document whose file already exists is loaded from it instead of being chunked again, and for `semantic` chunking only the documents that are not cached have their sentences embedded. `compute_chunks()` returns all chunks as a dict of `chunk_id` to `Chunk` and prints the chunk folder and how many documents and chunks came from cache or were newly computed.

With `--framework ""` the built-in chunkers are used. With `--framework langchain`, `compute_chunks()` hands the documents to the `Langchain` class instead, which chunks with LangChain's splitters for all three chunk types, using the same settings and the same per-document cache, saved under `{output folder}/chunks/langchain/...` so the two never share a cache. LangChain fixed and sentence chunking follow the same rules as the built-in ones; LangChain semantic chunking uses LangChain's own `SemanticChunker` rule and does not use `--semantic_threshold` or `--min_chunk_size`. The SentenceTransformer model is loaded by `Chunking` only for the built-in chunkers, because the `Langchain` class loads its own copy. LangChain chunking needs the `langchain-text-splitters` and `langchain-experimental` packages.

### Architecture

- **`chunking.py`** — `Chunking`, the orchestrator: validates the settings and, for `--framework langchain`, hands the work to `Langchain`. For the built-in chunkers it loads the SentenceTransformer model, loads cached chunks, embeds sentences for semantic chunking, chunks the remaining documents, saves them and prints the run summary.
- **`langchain.py`** — `Langchain`: loads cached chunks, chunks the remaining documents with one of `fixed_chunking` (LangChain's `SentenceTransformersTokenTextSplitter`), `sentence_chunking` (`NLTKTextSplitter`) or `semantic_chunking` (`SemanticChunker` from `langchain-experimental`), saves them and prints the run summary. Sentence and semantic pieces longer than `max_chunk_size` are cut with LangChain's `split_text_on_tokens`. Also holds `SentenceTransformerEmbeddings`, which lets `SemanticChunker` embed with the already-loaded model.
- **`chunker.py`** — The abstract `Chunker` base class shared by all built-in strategies (chunk id and size handling, and a sliding token-window splitter that stops once a window reaches the last token), and `get_chunker()`, the registry that maps a `chunk_type` to its strategy.
- **`chunkers/fixed.py`** — `FixedChunking`: splits a document's tokens into windows of `max_chunk_size` tokens that overlap by `fixed_chunk_overlap` tokens.
- **`chunkers/sentence.py`** — `SentenceChunking`: packs whole sentences into chunks of up to `max_chunk_size` tokens; a sentence longer than that is split with the fixed window.
- **`chunkers/semantic.py`** — `SemanticChunking`: like sentence chunking, but also starts a new chunk where the cosine similarity between neighbouring sentence embeddings falls below `semantic_threshold`, then merges chunks smaller than `min_chunk_size` into the previous chunk when they fit.
- **`cache.py`** — Saves and loads one JSON file per document in the chunk folder.
- **`validate.py`** — Checks the `ChunkingArgs` values before any chunking starts.
- **`schema.py`** — Defines `Chunk`, the layer's output record.

### Run results

All runs below use `pixparse/pdfa-eng-wds` with `--model_name all-MiniLM-L6-v2 --min_chunk_size 128 --max_chunk_size 256 --fixed_chunk_overlap 32 --semantic_threshold 0.2`.

**5000 documents, built-in vs LangChain** (from `log/l2_chunking/log_5k_builtin_{fixed,sentence,semantic}.txt`, `log/l2_chunking/log_5k_langchain_{fixed,sentence,semantic}.txt`, and the chunk statistics in `log/l2_chunking/chunk_stats_5k_builtin_vs_langchain.txt`). All six runs chunk the same 5000 documents, loaded from the `l1_ingestion` cache, with nothing in the chunk cache. The times are the `Chunking time` printed by `src/rag.py`, which covers the whole `Chunking(...).compute_chunks(...)` call, including loading the model.

| | built-in fixed | LangChain fixed | built-in sentence | LangChain sentence | built-in semantic | LangChain semantic |
|---|---|---|---|---|---|---|
| Chunks | 91670 | 91670 | 94053 | 94053 | 104900 | 112218 |
| Chunks per document | 18.3 | 18.3 | 18.8 | 18.8 | 21.0 | 22.4 |
| Mean / median size (tokens) | 249.8 / 256.0 | 250.0 / 256.0 | 218.2 / 239 | 218.3 / 239 | 195.6 / 220.0 | 199.5 / 256.0 |
| Min / max size (tokens) | 17 / 256 | 17 / 260 | 1 / 256 | 1 / 260 | 1 / 256 | 0 / 260 |
| Chunks below 128 tokens | 2166 (2.4%) | 2163 (2.4%) | 8253 (8.8%) | 8247 (8.8%) | 17368 (16.6%) | 26434 (23.6%) |
| Chunks of exactly 256 tokens | 86701 (94.6%) | 77822 (84.9%) | 15015 (16.0%) | 13951 (14.8%) | 13909 (13.3%) | 64290 (57.3%) |
| Chunks counted above 256 tokens | 0 | 8873 | 0 | 1067 | 0 | 5811 |
| Chunks whose text contains `[UNK]` | 0 | 4252 | 0 | 1048 | 0 | 3998 |
| Total tokens in all chunks | 22900712 | 22920513 | 20523464 | 20527069 | 20523464 | 22382959 |
| Chunking time | 41.90636873245239s | 67.74775099754333s | 46.04506301879883s | 123.76853036880493s | 325.39638090133667s | 567.8647112846375s |

The built-in semantic run spends 267.174s embedding 639475 sentences (`Embedded 639475 sentences in 267.174s` in its log).

**Fixed.** Both cut each document into windows of 256 tokens from the `all-MiniLM-L6-v2` tokenizer, moving 224 tokens forward each time (256 minus the 32-token overlap) and stopping once a window reaches the last token, so both produce the same 91670 chunks. Almost every window is a full 256 tokens; only the last window of each document is shorter. The fixed runs have about 2.4 million more tokens in total than the sentence runs because neighbouring windows repeat 32 tokens.

**Sentence.** Both cut the text into sentences with NLTK's `sent_tokenize` and pack whole sentences into a chunk while it stays within 256 tokens, with no overlap; a single sentence longer than 256 tokens is cut into overlapping 256-token windows. LangChain's `NLTKTextSplitter` keeps such a sentence whole on its own, so `Langchain` cuts it afterwards with LangChain's `split_text_on_tokens`. Both produce the same 94053 chunks, and they cover the text exactly once, so built-in sentence and semantic chunking have the same total token count (20523464).

**Where built-in and LangChain fixed/sentence chunks differ.** The chunks are cut at the same places; the differences come from turning token ids back into text and counting them:

- LangChain keeps the text `[UNK]`, the tokenizer's placeholder for a character it does not know, in the chunk text; the built-in chunkers decode with `skip_special_tokens=True`, which leaves it out. This happens only where token ids are decoded back into text: in every fixed chunk, and in sentence chunking only in windows cut from long sentences.
- LangChain's splitters only return text, so `Langchain` gets each chunk's `size` by tokenizing its text again, while the built-in chunkers take the size from the window or from the sentences' token counts. Tokenizing decoded text does not always give back the same number of tokens, which is why some LangChain chunks are counted at 257 to 260 tokens although they were cut from 256-token windows.

LangChain is slower for the same chunks because of that second tokenizing pass over every chunk; LangChain sentence also tokenizes every sentence while packing, which is why its gap to the built-in chunker is larger (123.76853036880493s vs 46.04506301879883s) than for fixed (67.74775099754333s vs 41.90636873245239s).

**Semantic.** The two use different rules, so their chunks are not expected to match. The built-in chunker starts a new chunk where the cosine similarity between neighbouring sentences falls below 0.2, never lets a chunk grow beyond 256 tokens, and merges chunks smaller than 128 tokens into the previous one. LangChain's `SemanticChunker` (run with `buffer_size=0`, so each sentence is embedded on its own) splits sentences after `.`, `?` or `!` and starts a new chunk only where the distance between neighbouring sentences is in the top 5% for that document (the 95th percentile), with no size limit. That leaves long pieces, which `Langchain` then cuts into overlapping 256-token windows: 57.3% of LangChain semantic chunks are exactly 256 tokens, its median is 256, and the window overlap adds about 1.9 million tokens compared with the built-in semantic chunks. It also leaves more chunks below 128 tokens (23.6% vs 16.6%), because it has no merge step. One LangChain semantic chunk has a size of 0: its text is not blank but contains nothing the tokenizer turns into a token.

Built-in semantic chunking is dominated by sentence embedding: 267.174s of its 325.39638090133667s go to running the model over all 639475 sentences in one batch, and the rest is the same sentence splitting and tokenizing work as sentence chunking. LangChain's `SemanticChunker` embeds one document at a time and `Langchain` tokenizes every chunk again for its size, which together make it slower (567.8647112846375s).

**100 documents, chunk-by-chunk comparison** (from `log/l2_chunking/log_100_fixed_vs_langchain.txt`, `log_100_sentence_vs_langchain.txt` and `log_100_semantic_vs_langchain.txt`). Each run chunks the same 100 documents with the built-in chunker and with LangChain and compares the chunks of each document; the times cover creating the `Chunking` object (including model loading) and the `compute_chunks()` call:

| | fixed | sentence | semantic |
|---|---|---|---|
| Chunks, built-in / LangChain | 1543 / 1543 | 1569 / 1569 | 1763 / 1841 |
| Documents with identical chunks | 83 | 98 | 8 |
| Chunks with different text | 40 (38 only by `[UNK]`) | 4 (all only by `[UNK]`) | – |
| Chunks with identical text but different size | 126 | 28 | – |
| Time, built-in / LangChain | 3.25s / 4.57s | 4.41s / 6.15s | 7.84s / 13.4s |

For fixed and sentence, every chunk whose text differs contains `[UNK]`, and the size differences are 1 to 3 tokens, for the reasons above. For semantic, 86 of the 100 documents get a different number of chunks, as expected from the different rules.

**Earlier runs, before the built-in chunkers stopped at the last token** (from `log/l2_chunking/log_100_fixed.txt`, `log_5k_fixed.txt`, `log_5k_sentence.txt`, `log_5k_semantic.txt` and `chunk_stats_5k.txt`). The built-in window splitter used to start one more window after a window had already reached the last token; that window held only the last tokens of the document, which were already inside the previous window. On 100 documents fixed chunking produced 1555 chunks instead of 1543, and on 5000 documents fixed, sentence and semantic produced 92366, 94473 and 105050 chunks instead of 91670, 94053 and 104900; in sentence and semantic chunking the extra windows came from sentences longer than 256 tokens. These extra windows were also the 1-token chunks in fixed chunking: the smallest fixed chunk was 1 token then and is 17 tokens now.

The log line `Token indices sequence length is longer than the specified maximum sequence length for this model` is printed by the tokenizer when a whole document is tokenized for counting; the model is not run on those tokens.

## l3_embedding

### Overview

`l3_embedding` is the third pipeline layer. `src/rag.py` runs it right after `l2_chunking` and passes it the chunks in memory, as a dict of `chunk_id` to `Chunk`, together with the same output folder and `ChunkingArgs` as l2. `Embedding.compute_embeddings()` embeds each chunk's text with the SentenceTransformer model named by `--model_name` and returns a dict of `chunk_id` to `Embedded` (`chunk_id`, `embeddings` as a numpy vector, `normalized`). Vectors are normalized to length 1, so a dot product between two of them is their cosine similarity. Embeddings are cached in `{output folder}/embeddings/{chunk_type}/{model_name}/{min}_{max}_{overlap}_{threshold}/`, one file per document holding the embeddings of all its chunks, written to a temporary file first and then renamed into place. Each `{doc_id}.npz` holds three arrays in chunk order: the chunk ids, the vectors as one float32 array, and the `normalized` flags. A chunk whose document file already exists is loaded from it; all other chunks are embedded in one batched call to the model, and each document that got new embeddings is saved. `compute_embeddings()` prints the embedding folder and how many embeddings came from cache or were newly computed.

With `--framework ""` the built-in `Embedding` path is used. With `--framework langchain`, `compute_embeddings()` hands the chunks to the `Langchain` class instead, which embeds with LangChain's `HuggingFaceEmbeddings` (from `langchain-huggingface`) using the same SentenceTransformer model, batch size 256 and normalization, and the same per-document `.npz` cache, saved under `{output folder}/embeddings/langchain/...` so the two never share a cache. `HuggingFaceEmbeddings` returns each vector as a list of Python floats, so each one is turned back into a float32 numpy array before it is cached. The SentenceTransformer model is loaded by `Embedding` only for the built-in path, because the `Langchain` class loads its own copy.

### Architecture

- **`embedding.py`** — `Embedding`, the orchestrator: for `--framework langchain`, hands the work to `Langchain`. For the built-in path it loads the SentenceTransformer model, loads cached embeddings, embeds the remaining chunks in one batch, saves them per document and prints the run summary.
- **`cache.py`** — Saves and loads one file per document in the embedding folder.
- **`schema.py`** — Defines `Embedded`, the layer's output record.
- **`langchain.py`** — `Langchain`: loads cached embeddings, embeds the remaining chunks in one call with LangChain's `HuggingFaceEmbeddings`, converts the vectors to float32, saves them per document and prints the run summary.

### Run results

**5000 documents, JSON cache** (from `log/l3_embedding/log_5k_{fixed,sentence,semantic}.txt`, `log/l3_embedding/log_5k_json_cached_{fixed,sentence,semantic}.txt` and `log/l3_embedding/cache_json_stats.txt`). These runs save each document's embeddings as a JSON list of `{chunk_id, embeddings, normalized}`, with the vector written as a list of numbers. All runs use the 5000 `pixparse/pdfa-eng-wds` documents, `--model_name all-MiniLM-L6-v2 --min_chunk_size 128 --max_chunk_size 256 --fixed_chunk_overlap 32 --semantic_threshold 0.2` and the built-in chunkers; l1 and l2 load everything from their caches (about 8 to 9s and 4s). The first run embeds every chunk; the second run loads every embedding from the JSON cache. The times are the `Embedding time` printed by `src/rag.py`, which includes loading the model:

| | fixed | sentence | semantic |
|---|---|---|---|
| Chunks embedded | 91670 | 94053 | 104900 |
| Embedding time, first run (all new) | 152.94002962112427s | 154.61784386634827s | 170.7557156085968s |
| Embedding time, second run (all from JSON cache) | 10.379403352737427s | 9.395085573196411s | 10.475048780441284s |
| Cache files | 5000 | 5000 | 5000 |
| Cache size on disk | 779851088 bytes | 800119736 bytes | 892390228 bytes |

Embedding time grows with the number of chunks, at about 600 chunks per second for all three chunk types, because every chunk is at most 256 tokens and the model runs over every chunk's text once. Semantic chunking has the most chunks, so it takes longest.

**Why the cache moved from JSON to `.npz`.** The JSON cache works, but it has three costs that matter for 384-number vectors:

- **Size.** Each number is stored as text of up to about 17 digits, so the cache takes about 8.5 KB per chunk and 779851088 to 892390228 bytes per chunk type, for vectors that hold 1536 bytes of float32 data each.
- **Data type.** The model returns float32 vectors (`vector returned by the model: dtype float32 ... bytes 1536` in `cache_json_stats.txt`), but a vector loaded from the JSON cache comes back as float64 (`vector loaded from JSON cache: dtype float64 ... bytes 3072`), because the numbers are read back as Python floats. Cached and new embeddings therefore have different types, and cached ones take twice the memory. The FAISS indexes in l4 take float32 vectors.
- **Load time.** Loading the embeddings back means parsing every number from text: about 10s per chunk type for the second runs above.

numpy's `.npz` format stores several arrays in one binary file, so one `{doc_id}.npz` per document can hold the chunk ids, all the document's vectors as one float32 array, and the `normalized` flag, while keeping one file per document and the temporary-file-and-rename write.

**5000 documents, `.npz` cache** (from `log/l3_embedding/log_5k_npz_{fixed,sentence,semantic}.txt`, `log/l3_embedding/log_5k_npz_cached_{fixed,sentence,semantic}.txt` and `log/l3_embedding/cache_npz_stats.txt`). Same documents, settings and two runs per chunk type as the JSON runs above, after deleting the JSON cache:

| | fixed | sentence | semantic |
|---|---|---|---|
| Embedding time, first run (all new), JSON → `.npz` | 152.94002962112427s → 131.20864152908325s | 154.61784386634827s → 134.51999473571777s | 170.7557156085968s → 148.0821304321289s |
| Embedding time, second run (all from cache), JSON → `.npz` | 10.379403352737427s → 4.5234534740448s | 9.395085573196411s → 4.807491779327393s | 10.475048780441284s → 4.921001672744751s |
| Cache size on disk, JSON → `.npz` | 779851088 → 152010282 bytes | 800119736 → 155861957 bytes | 892390228 → 173538276 bytes |
| Cache files | 5000 | 5000 | 5000 |

- **Size.** The `.npz` cache is about 5.1 times smaller (152010282 bytes for 91670 fixed chunks, about 1.66 KB per chunk), close to the 1536 bytes of float32 data in each vector; the rest is the chunk ids, the flags and the file headers.
- **Data type.** A vector loaded from the `.npz` cache is float32 (`vector loaded from npz cache: dtype float32 ... bytes 1536` in `cache_npz_stats.txt`), the same type the model returns, so cached and new embeddings now match.
- **Load time.** Loading all embeddings from the cache takes about 4.5 to 4.9s instead of about 9.4 to 10.5s, because each document's vectors are read as one binary array instead of being parsed number by number from text. These times include loading the model.
- **First run.** The first run is about 20s faster for every chunk type. Embedding the chunks is the same work in both formats, so the difference is writing the cache: converting every number to text for JSON takes longer than writing the binary arrays.

**5000 documents, built-in vs LangChain** (from `log/l3_embedding/log_5k_npz_{fixed,sentence,semantic}.txt`, `log/l3_embedding/log_5k_npz_cached_{fixed,sentence,semantic}.txt`, `log/l3_embedding/log_5k_langchain_{fixed,sentence,semantic}.txt`, `log/l3_embedding/log_5k_langchain_cached_{fixed,sentence,semantic}.txt`, `log/l3_embedding/cache_npz_stats.txt` and `log/l3_embedding/cache_npz_langchain_stats.txt`). The LangChain runs use the same 5000 documents and settings as the `.npz` runs above with `--framework langchain`, so l2 loads the LangChain chunks from the LangChain chunk cache and l3 embeds them with `HuggingFaceEmbeddings`. As before, the first run embeds every chunk and the second run loads every embedding from the cache. The times are the `Embedding time` printed by `src/rag.py`, which includes loading the model:

| | fixed | sentence | semantic |
|---|---|---|---|
| Chunks embedded, built-in / LangChain | 91670 / 91670 | 94053 / 94053 | 104900 / 112218 |
| Embedding time, first run, built-in → LangChain | 131.20864152908325s → 136.1494882106781s | 134.51999473571777s → 137.66652965545654s | 148.0821304321289s → 151.84057593345642s |
| Embedding time, second run (all from cache), built-in → LangChain | 4.5234534740448s → 4.88601016998291s | 4.807491779327393s → 4.618432521820068s | 4.921001672744751s → 4.904831886291504s |
| Cache size on disk, built-in → LangChain | 152010282 → 152010282 bytes | 155861957 → 155861957 bytes | 173538276 → 185212182 bytes |
| Cache files | 5000 / 5000 | 5000 / 5000 | 5000 / 5000 |

- **First run.** LangChain takes 3.1 to 4.9s longer for fixed and sentence, about 3% more. `HuggingFaceEmbeddings` runs the same SentenceTransformer `encode` call on the same chunks, so the model work is the same. The extra time comes from the format conversion: `HuggingFaceEmbeddings` turns the float32 output into Python lists of floats (384 per chunk, about 35 million for fixed), and `Langchain` turns each list back into a float32 array for the cache.
- **Semantic.** LangChain's `SemanticChunker` produces 112218 chunks instead of 104900 (see l2_chunking), so LangChain embeds 7318 more chunks. It still takes only 3.76s longer, because the time per chunk is about the same: 104900 chunks in 148.08s and 112218 in 151.84s.
- **Second run.** Loading from cache takes 4.6 to 4.9s for both, because both read the same `.npz` files with the same `cache.py` code; LangChain is not involved once the embeddings are cached.
- **Cache size.** Fixed and sentence caches are exactly the same size in bytes, because both hold the same number of chunks with the same chunk ids and float32 vectors of 1536 bytes. The semantic LangChain cache is larger only because it holds more chunks.

## l4_indexing

### Overview

`l4_indexing` is the fourth pipeline layer. `src/rag.py` runs it right after `l3_embedding` and passes it the chunks (`chunk_id` to `Chunk`) and the embeddings (`chunk_id` to `Embedded`) in memory, together with the dataset's output folder, `ChunkingArgs` and `IndexArgs`. `Indexing.compute_index()` validates `IndexArgs` and always builds three indexes from the same chunks, returning them as a tuple `(FaissIndex, Bm25Index, QdrantIndex)`:

- **FAISS** over the embedding vectors, of the type given by `--faiss_index_type`: `faiss_flatip` (exact inner product), `faiss_ivf` (`--ivf_nlist` clusters) or `faiss_hnsw` (`--hnsw_m` neighbours per node, `--hnsw_ef_construction`). The vectors are normalized by l3, so inner product is cosine similarity. Saved as `index.faiss` and `chunk_ids.json` in `{output folder}/indexes/faiss/{chunk_type}/{model_name}/{min}_{max}_{overlap}_{threshold}/{flatip | ivf_{nlist} | hnsw_{m}_{ef_construction}}/`.
- **BM25** (`rank_bm25.BM25Okapi` with `--bm25_k1` and `--bm25_b`) over the chunk texts, lowercased and split on whitespace. Saved as `bm25.pkl` and `chunk_ids.json` in `{output folder}/indexes/bm25/{chunk_type}/{model_name}/{min}_{max}_{overlap}_{threshold}/{k1}_{b}_{tokenizer}/`.
- **Qdrant**, a collection on the Qdrant server at `QDRANT_URL` (`http://localhost:6333`, started with the `qdrant-docker` skill) with cosine distance, uploaded in batches of `--qdrant_batch_size`. Each point stores its `chunk_id` and chunk text in its payload, so a search returns both. A collection without text in its payload (built before the text was added) is rebuilt. The collection is named `{dataset_name}_{chunk_type}_{model_name}_{min}_{max}_{overlap}_{threshold}`.

Each index is used as its cache: if the saved FAISS or BM25 files, or the existing Qdrant collection, hold exactly the same chunk ids as the current chunks, it is loaded; otherwise it is rebuilt and replaces the old one. FAISS and BM25 files are written to a temporary file first and then renamed into place. `chunk_ids` in `FaissIndex` and `Bm25Index` maps an index position back to its chunk id. Each index prints whether it was built or loaded from cache.

With `--framework ""` the built-in indexes above are used. With `--framework langchain`, `compute_index()` hands the chunks and embeddings to the `Langchain` class instead, which builds the same three indexes with LangChain's classes and returns `(LangchainFaissIndex, LangchainBm25Index, LangchainQdrantIndex)`. Every LangChain index holds one LangChain `Document` per chunk, with the chunk text as `page_content` and `{"chunk_id": chunk_id}` as metadata, so it stores the chunk texts as well as the vectors or terms. No chunk is embedded again: all indexes use the l3 vectors, and the `HuggingFaceEmbeddings` model is loaded only because the FAISS and Qdrant vector stores need it to embed queries.

- **FAISS** (`FAISS` from `langchain-community`): the FAISS index is built by the same `build_flatip_index`, `build_ivf_index` or `build_hnsw_index` functions as the built-in one and wrapped in a LangChain `FAISS` vector store with inner product distance. Saved with `save_local` as `index.faiss` and `index.pkl` (the documents and the position-to-chunk-id map).
- **BM25** (`BM25Retriever` from `langchain-community`) with the same `k1`, `b` and lowercase whitespace tokenizer. Saved with pickle as `bm25_retriever.pkl`.
- **Qdrant** (`QdrantVectorStore` from `langchain-qdrant`): the l3 vectors are uploaded directly in the payload layout `QdrantVectorStore` reads (`page_content` and `metadata`), because `add_texts` would embed every chunk again.

The LangChain indexes are saved under `{output folder}/indexes/{faiss,bm25}/langchain/...`, and the Qdrant collection name has `langchain_` after the dataset name, so the two never share a cache. The cache rule is the same as for the built-in indexes.

### Architecture

- **`indexing.py`** — `Indexing`, the orchestrator: validates the index args and, for `--framework langchain`, hands the work to `Langchain`. Otherwise gets each index's folder or collection name, and builds or loads the FAISS, BM25 and Qdrant indexes.
- **`validate.py`** — Checks the indexing parameters and raises an error for the first invalid one.
- **`schema.py`** — Defines `FaissIndex`, `Bm25Index` and `QdrantIndex`, and `LangchainFaissIndex`, `LangchainBm25Index` and `LangchainQdrantIndex`, the layer's output records.
- **`indexers/faiss.py`** — Builds a flat, IVF or HNSW FAISS index, saves it, and loads it from cache.
- **`indexers/bm25.py`** — Tokenizes the chunk texts, builds the BM25 model, saves it, and loads it from cache.
- **`indexers/qdrant.py`** — Connects to the Qdrant server, reuses a matching collection or creates one and uploads the points.
- **`indexer.py`** — Empty placeholder for a shared base class.
- **`cache.py`** — Empty placeholder; each indexer currently saves and loads its own cache.
- **`langchain.py`** — `Langchain`: loads the query embedding model and builds or loads the LangChain FAISS vector store, BM25 retriever and Qdrant vector store, one method each.

The index paths and the Qdrant collection name are built in `src/common/paths.py`.

### Run results

**5000 documents, flatip / ivf / hnsw** (from `log/l4_indexing/log_5k_{flatip,ivf,hnsw}_{fixed,sentence,semantic}.txt` and `log/l4_indexing/index_stats.txt`). All runs use the 5000 `pixparse/pdfa-eng-wds` documents, `--model_name all-MiniLM-L6-v2 --min_chunk_size 128 --max_chunk_size 256 --fixed_chunk_overlap 32 --semantic_threshold 0.2`, the built-in chunkers, and `--ivf_nlist 256 --hnsw_m 32 --hnsw_ef_construction 200 --bm25_k1 1.5 --bm25_b 0.75 --bm25_tokenizer whitespace --qdrant_batch_size 256`. l1, l2 and l3 load everything from their caches (about 11 to 14s, 5 to 6s and 6 to 8s). For each chunk type the flatip run went first with no index cache, so it built FAISS, BM25 and Qdrant; the ivf and hnsw runs then built only their FAISS index and loaded BM25 and Qdrant from cache. The times are the `Indexing time` printed by `src/rag.py`, which covers the whole `compute_index()` call:

| | fixed | sentence | semantic |
|---|---|---|---|
| Chunks indexed | 91670 | 94053 | 104900 |
| flatip run (FAISS flatip + BM25 + Qdrant built) | 28.998353481292725s | 27.878997802734375s | 34.35384702682495s |
| ivf run (FAISS ivf built, BM25 and Qdrant from cache) | 2.5223073959350586s | 2.45229434967041s | 2.518348455429077s |
| hnsw run (FAISS hnsw built, BM25 and Qdrant from cache) | 10.658159971237183s | 11.403585195541382s | 13.719996452331543s |
| Total elapsed, flatip / ivf / hnsw run | 54.40952920913696s / 24.56925320625305s / 33.112295150756836s | 52.15672993659973s / 25.17382049560547s / 34.09154915809631s | 60.032033920288086s / 27.975639820098877s / 36.53645873069763s |

Index size on disk (bytes):

| | fixed | sentence | semantic |
|---|---|---|---|
| FAISS flatip | 142904842 | 146621223 | 163558055 |
| FAISS ivf_256 | 144033560 | 147769005 | 164792613 |
| FAISS hnsw_32_200 | 167856327 | 172221336 | 192114700 |
| BM25 | 113634193 | 107784587 | 109171690 |

- **flatip run.** This is the only run per chunk type that builds BM25 and Qdrant, so it is the slowest. Building the flat FAISS index itself only copies the vectors into memory. The rest is tokenizing every chunk text and building and pickling the BM25 model, and uploading every vector to Qdrant over HTTP in batches of 256, which Qdrant also writes to disk and indexes. The log prints one time for all three, so the split between BM25 and Qdrant is not in the logs.
- **ivf run.** Building IVF runs k-means to find 256 cluster centres and then assigns each vector to its nearest centre. The 2.45 to 2.52s also includes loading the BM25 pickle and reading every chunk id back from the Qdrant collection to check the cache, so the IVF build alone is shorter than that.
- **hnsw run.** HNSW inserts the vectors one by one into a graph, and each insert searches the graph for its 32 neighbours with a candidate list of 200. That search per vector makes it the slowest FAISS index to build, and the time grows with the chunk count: 10.66s for 91670 chunks, 13.72s for 104900.
- **FAISS sizes.** A flat index stores each 384-number float32 vector, 1536 bytes per chunk (140805120 bytes for 91670 chunks), plus `chunk_ids.json`. IVF adds the 256 cluster centres and a list id for each vector, about 1.1 MB. HNSW adds the graph links, about 25 MB for fixed (a little over 256 bytes per chunk: 64 neighbour ids of 4 bytes on the bottom layer), so it is the largest.
- **Chunk count.** Semantic chunking has the most chunks, so its FAISS indexes are the largest and its flatip and hnsw runs take longest.

**5000 documents, Qdrant rebuilt with chunk text in the payload** (from `log/l4_indexing/log_5k_qdrant_text_{fixed,sentence,semantic}.txt`). The runs above stored only the `chunk_id` in each Qdrant point. After the chunk text was added to the payload, one built-in run per chunk type (`--faiss_index_type faiss_flatip`, same settings as above) rebuilt the three Qdrant collections; FAISS and BM25 were loaded from cache, so the `Indexing time` is almost all the Qdrant upload:

| | fixed | sentence | semantic |
|---|---|---|---|
| Points uploaded | 91670 | 94053 | 104900 |
| Indexing time (FAISS and BM25 from cache, Qdrant built) | 26.42556595802307s | 26.590499877929688s | 31.36462950706482s |

These times, the Qdrant upload plus loading FAISS and BM25 from cache, are close to the full built-in flatip run times above (27.878997802734375s to 34.35384702682495s), so the Qdrant upload is the largest part of building all three indexes. The time grows with the number of points, because every point is sent over HTTP in batches of 256 and written and indexed by the Qdrant server.

**5000 documents, built-in vs LangChain** (from `log/l4_indexing/log_5k_{flatip,ivf,hnsw}_{fixed,sentence,semantic}.txt`, `log/l4_indexing/log_5k_langchain_{flatip,ivf,hnsw}_{fixed,sentence,semantic}.txt`, `log/l4_indexing/index_stats.txt` and `log/l4_indexing/index_langchain_stats.txt`). The LangChain runs use the same documents and settings as the built-in runs above with `--framework langchain`, so l1, l2 and l3 load the LangChain chunks and embeddings from their caches. As before, for each chunk type the flatip run went first and built FAISS, BM25 and Qdrant, and the ivf and hnsw runs built only their FAISS index and loaded BM25 and Qdrant from cache. The times are the `Indexing time` printed by `src/rag.py`:

| | fixed | sentence | semantic |
|---|---|---|---|
| Chunks indexed, built-in / LangChain | 91670 / 91670 | 94053 / 94053 | 104900 / 112218 |
| flatip run, built-in → LangChain | 28.998353481292725s → 36.737186431884766s | 27.878997802734375s → 39.81909370422363s | 34.35384702682495s → 44.598501205444336s |
| ivf run, built-in → LangChain | 2.5223073959350586s → 7.879034519195557s | 2.45229434967041s → 9.069776058197021s | 2.518348455429077s → 9.874390602111816s |
| hnsw run, built-in → LangChain | 10.658159971237183s → 16.54848623275757s | 11.403585195541382s → 18.42800498008728s | 13.719996452331543s → 22.074488401412964s |

Index size on disk, built-in → LangChain (bytes):

| | fixed | sentence | semantic |
|---|---|---|---|
| FAISS flatip | 142904842 → 247032753 | 146621223 → 238728506 | 163558055 → 277941285 |
| FAISS ivf_256 | 144033560 → 248161471 | 147769005 → 239876288 | 164792613 → 279234387 |
| FAISS hnsw_32_200 | 167856327 → 271984238 | 172221336 → 264328619 | 192114700 → 308491922 |
| BM25 | 113634193 → 220136801 | 107784587 → 202291446 | 109171690 → 219744399 |

- **Every LangChain run is slower, by 5.4 to 11.9s.** The FAISS index itself is built by the same functions, so the difference is the LangChain work around it: loading the `HuggingFaceEmbeddings` model, creating one `Document` per chunk, and pickling or unpickling those documents with their chunk texts. The ivf and hnsw runs also load `bm25_retriever.pkl`, which holds every chunk text, instead of the smaller built-in `bm25.pkl`. The log prints one time for the whole call, so the split between these steps is not in the logs.
- **FAISS size.** `index.faiss` is the same FAISS index as the built-in one (140805165 bytes for fixed flatip, the 91670 float32 vectors). The extra size is `index.pkl`, 106223492 bytes for fixed, which holds every chunk text and the position-to-chunk-id map. The built-in index stores only `chunk_ids.json` next to `index.faiss`, without the chunk texts.
- **BM25 size.** `bm25_retriever.pkl` is about twice the built-in `bm25.pkl` because it holds the BM25 model plus every chunk as a LangChain `Document` with its text and metadata.
- **Semantic.** LangChain semantic chunking has 7318 more chunks than the built-in one (see l2_chunking), so its indexes and times are larger for that reason as well.

## llm

### Overview

`llm` gives every layer that calls an LLM the same interface, for four providers: `claude`, `chatgpt`, `gemini` and `ollama`. A layer's LLM settings come from the command line: `eval.sh` sets the provider, model and prices for each layer (`QNA_*`, `QNA_KG_*`, `QUERY_*`, `GENERATION_*`) from a models block at the top, plus one `LLM_MAX_TOKENS` and `LLM_TIMEOUT_SECONDS` shared by all layers. `eval.py` turns each layer's settings into an `LLM` object with `make_llm_args`, and passes that object to the layer. The layer calls `validate_llm(llm_args)` only when its cache is missing, and then calls `get_llm(llm_args)`, which returns the provider's LLM class: `ClaudeLLM`, `ChatGPTLLM`, `GeminiLLM` or `OllamaLLM`. Every class has the same four things: `chat_model` (a LangChain chat model), `json_chat_model` (the same model forced to reply in JSON), `generate(system_prompt, user_prompt)`, which returns the reply text, and `compute_cost()`, which prints and returns the cost of its calls. `qna` hands `json_chat_model` of its two LLMs to RAGAS. `l5_query` calls `generate("", prompt)`. No layer knows which provider or model it got.

The models block in `eval.sh`, one model name and price pair per model:

| Provider | Model | Price per million tokens (input / output) | Used for |
|---|---|---|---|
| `claude` | `claude-sonnet-5-5` | $2.00 / $10.00 | QnA questions and answers |
| `claude` | `claude-haiku-5-5` | $0.10 / $0.50 | QnA knowledge graph |
| `chatgpt` | `gpt-4.1-mini` | $0.40 / $1.60 | |
| `chatgpt` | `gpt-4.1-nano` | $0.10 / $0.40 | |
| `gemini` | `gemini-3.5-flash-lite` | $0.30 / $2.50 | |
| `gemini` | `gemini-3.1-flash-lite` | $0.25 / $1.50 | |
| `ollama` | `llama3.1:8b` | $0 / $0 | l5 query, l9 generation |

Each provider has a cheaper model for the many knowledge graph calls, which send whole documents, and a stronger one for the few question and answer calls.

Design choices and why:

- **One `LLM` object per layer.** Its fields are `llm`, `llm_model`, `input_price`, `output_price`, `max_tokens` and `timeout_seconds`. Changing a layer's model is one line in `eval.sh`, with no code change. Cache names use the model name, not the provider (e.g. `eval_qna_100_200_claude-sonnet-5-5_…`), so a cache made by one model is never loaded for another.
- **Prices come from the command line.** The providers' list-models APIs do not return prices, so a price table would have to be kept by hand in the code anyway. `compute_cost()` multiplies the counted tokens by `input_price` and `output_price`. A paid model must have both prices above 0, so the cost of every paid run is printed. A wrong price cannot be detected, only a missing one.
- **General code in `llm.py`, `schema.py` and `validate.py`, provider code in the provider's own file.** Each provider file holds its settings, its `validate_<provider>()` and its LLM class, written out in full rather than sharing a base class, so each provider's flow reads top to bottom in one place, and a fix for one provider changes only its file. The only shared piece is `TokenCounter`, because LangChain reports token usage in the same `usage_metadata` for every provider.
- **Checks run only when an LLM is needed.** `validate_llm` checks the settings every provider shares (model name, prices, `max_tokens`, `timeout_seconds`), then calls the provider's check: API key set and the model listed by the provider for the paid ones, server running and model pulled for Ollama. QnA and l5 run it only on a cache miss, so a cached run needs no API key, no `ALLOW_PAID_API` and no Ollama server. A listed model is not always usable: Google lists `gemini-2.5-flash` and `gemini-2.5-flash-lite`, but rejects them for new accounts (`404 NOT_FOUND: … no longer available to new users`), which only a real call shows. A failed call raises an error and stops the run.
- **Paid calls need `ALLOW_PAID_API=1`.** Each paid provider's check tests it, and each paid LLM class tests it again when it is built, so a paid LLM cannot be built by a caller that skipped `validate_llm`.
- **API keys are read from `CLAUDE_API_KEY`, `CHATGPT_API_KEY` and `GEMINI_API_KEY` in `.env`,** named after the providers, and each class passes its key to its LangChain client. LangChain's clients look for `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` and `GOOGLE_API_KEY` on their own, so passing the key keeps a key set under those names for something else from being used.
- **One output limit and one timeout for every provider** (16000 tokens, 180s). The replies are short JSON (QnA), one question or passage (l5) or a few sentences (l9), so one value fits all of them.
- **`json_chat_model`.** RAGAS parses every reply as JSON. Claude follows the JSON format RAGAS asks for, so its `json_chat_model` is its `chat_model`. ChatGPT uses OpenAI's JSON mode (`response_format`), Gemini its JSON output (`response_mime_type="application/json"`) and Ollama its JSON mode (`format="json"`).
- **ChatGPT keep-alive is off.** RAGAS runs each step (knowledge graph, personas, scenarios, samples) in a new asyncio event loop. `ChatOpenAI` keeps its HTTPS connections open between calls, and a connection left from the previous step's closed loop failed the scenarios step with `RuntimeError: Event loop is closed` and `openai.APIConnectionError: Connection error.` `ChatGPTLLM` gives its chat models an `httpx.AsyncClient` with `max_keepalive_connections=0`, so every call opens its own connection. Google's client has the same reuse, but there it only prints `Fatal error on SSL transport … Event loop is closed` while closing old connections; every call succeeds, so `GeminiLLM` is left unchanged.
- **Libraries:** `langchain-anthropic` (`ChatAnthropic`), `langchain-openai` (`ChatOpenAI`), `langchain-google-genai` 4.4.1 (`ChatGoogleGenerativeAI`; 4.4.2 needs `langchain-core` 1.6.10, and the project stays on 1.6.9) and `langchain-ollama` (`ChatOllama`).

Ollama choices and limits:

- **Ollama's own API through `ChatOllama`, not its OpenAI-compatible API.** Ollama's default context window is 4096 tokens, and a prompt longer than that is cut from the start, so RAGAS's instructions are lost. With 4096, llama3.1:8b answered the RAGAS headline prompt correctly for a 1,196-token prompt, but for a 15,347-token prompt it copied the document's table as JSON until the 180s timeout, and for a 77,084-token prompt it returned its input. The OpenAI-compatible API ignores a context setting, so the native API is used, where `num_ctx` is set on every call.
- **Context window 32768 (`OLLAMA_CONTEXT_LENGTH` in `ollama.py`).** At 32768, llama3.1:8b uses 9.1 GB and stays fully on the 12 GB RTX 3060 of this machine. A larger window would not fit on the GPU.
- **Temperature 0**, so the same prompt gives the same reply on every run.
- **Ollama is used for l5 queries, not for eval QnA.** l5 prompts are one question plus an instruction, far below 32768 tokens. RAGAS sends whole documents, and this dataset has documents much longer than 32768 tokens (75,450 in the 20-document check set), which do not fit. Eval QnA uses Claude.

`scripts/check_llms.sh <provider>` runs `src/llm/check_llms.py` (as `python3 -m llm.check_llms`, so `src/llm/llm.py` does not hide the `llm` package) to check the layers that use an LLM end to end with one provider. It is a developer check, not part of the pipeline: `eval.sh` and `build.sh` never run it. It builds a 20-document dataset under `outputs/llm_check/<provider>/`, keeps only documents of at most 8000 tokens so every provider gets the same documents and they fit in Ollama's context window, deletes its own QnA, knowledge graph and query caches, generates 2 QnA items from 3 documents and their queries, checks the records, the `llm` and `llm_model` fields and the cache files, and runs both layers again to check they load from the cache without writing it again. It never writes under the real `outputs/<dataset>/` caches. `ollama` is free; `claude`, `chatgpt` and `gemini` need `ALLOW_PAID_API=1`. Each provider uses its cheaper model for the knowledge graph and the queries, and its stronger one for the questions and answers. QnA and query passed with all four providers (below), so those steps are now commented out in `check_llms.py`, to not pay for them again; removing the `#` in front of steps 2 to 6 turns them back on.

### Architecture

- **`schema.py`**: defines `LLM`, the LLM settings of one layer.
- **`validate.py`**: `validate_llm`, the checks every provider shares, then the provider's own check.
- **`llm.py`**: `TokenCounter`, shared by every provider, and `get_llm`, which builds the LLM class of the provider.
- **`claude.py`**: `validate_claude` and `ClaudeLLM`, built on `ChatAnthropic`.
- **`chatgpt.py`**: `validate_chatgpt` and `ChatGPTLLM`, built on `ChatOpenAI`.
- **`gemini.py`**: `validate_gemini` and `GeminiLLM`, built on `ChatGoogleGenerativeAI`.
- **`ollama.py`**: Ollama's settings, `validate_ollama` and `OllamaLLM`, built on `ChatOllama`.

### Run results

**`check_llms.sh` for each provider** (from `log/llm/log_check_ollama.txt`, `log_check_claude.txt`, `log_check_gemini.txt` and `log_check_chatgpt.txt`). Every run: 17 of 20 documents have at most 8000 tokens, 12 of those have more than 500 tokens (RAGAS minimum), 3 are sampled, and 2 QnA items and 2 queries are generated, with 0 multi-query and 0 HyDE I don't know. Each run ends with `ALL CHECKS PASSED`.

| Provider | Knowledge graph model | Calls | Input / output tokens | Questions model | Calls | Input / output tokens | QnA cost |
|---|---|---|---|---|---|---|---|
| Ollama | `llama3.1:8b` | 31 | 42581 / 2174 | `llama3.1:8b` | 7 | 5778 / 462 | $0.0000 |
| Claude | `claude-haiku-5-5` | 31 | 72111 / 14223 | `claude-sonnet-5-5` | 7 | 8879 / 1870 | $0.0508 |
| Gemini | `gemini-3.1-flash-lite` | 29 | 42504 / 1790 | `gemini-3.5-flash-lite` | 8 | 5284 / 450 | $0.0160 |
| ChatGPT | `gpt-4.1-nano` | 25 | 36245 / 1665 | `gpt-4.1-mini` | 7 | 5259 / 617 | $0.0074 |

Query, in the same runs:

| Provider | Query model | Calls | Input / output tokens | Cost |
|---|---|---|---|---|
| Ollama | `llama3.1:8b` | 6 | 590 / 348 | $0.0000 |
| Claude | `claude-haiku-5-5` | 6 | 1039 / 1847 | $0.0010 |
| Gemini | `gemini-3.1-flash-lite` | 6 | 556 / 340 | $0.0006 |
| ChatGPT | `gpt-4.1-nano` | 6 | 594 / 425 | $0.0003 |

RAGAS step times:

| Step | Ollama | Claude | Gemini | ChatGPT |
|---|---|---|---|---|
| HeadlinesExtractor (3 documents) | 00:08 | 00:06 | 00:04 | 00:04 |
| SummaryExtractor (3) | 00:12 | 00:07 | 00:03 | 00:04 |
| CustomNodeFilter (9) | 00:11 | 00:07 | 00:07 | 00:06 |
| EmbeddingExtractor (3) | 00:01 | 00:02 | 00:01 | 00:01 |
| ThemesExtractor | 00:15 (8) | 00:08 (8) | 00:05 (7) | 00:06 (5) |
| NERExtractor | 00:11 (8) | 00:09 (8) | 00:05 (7) | 00:05 (5) |
| Generating personas (3) | 00:04 | 00:05 | 00:04 | 00:03 |
| Generating Scenarios | 00:05 (2) | 00:04 (2) | 00:05 (1) | 00:02 (2) |
| Generating Samples (2) | 00:06 | 00:09 | 00:02 | 00:03 |

The total run times are not in the logs.

The knowledge graph steps make most of the calls (25 to 31 of 32 to 38) and send most of the input tokens, because each sends a whole document or one of its chunks to the model. The question steps send only short persona, scenario and chunk texts, and the query step sends one question per prompt. The number of chunks the themes and entities steps work on differs by provider (5, 7 or 8), because the CustomNodeFilter step, which keeps or drops chunks, is itself an LLM call. Claude writes the most output tokens for the same work (14223 for the knowledge graph against 1665 to 2174 for the others), so its run costs the most even though Haiku's input price ($0.10) is as low as `gpt-4.1-nano`'s. Ollama is the slowest per step, because llama3.1:8b runs on the local GPU, while the paid models run on the providers' servers.
## qna

### Overview

`qna` builds the eval question-answer set the RAG pipeline is scored against. `src/eval.py` calls `get_eval_qna(output_path, documents, eval_qna_args, qna_llm_args, qna_kg_llm_args)` after loading the l1 documents (`doc_id` to `Document`). `EvalQnAArgs` holds `--eval_qna_model` (the local SentenceTransformer RAGAS uses to link related documents, `all-MiniLM-L6-v2`), `--n_sample_documents` and `--n_sample_evals`. The two LLMs are `LLM` objects (see `llm`) made from `eval.sh`: `qna_llm_args` from `QNA_LLM`, `QNA_LLM_MODEL` and its prices, which writes the questions and answers, and `qna_kg_llm_args` from the `QNA_KG_*` settings, which builds the knowledge graph. They can be any of the four providers; the project uses `claude-sonnet-5-5` and `claude-haiku-5-5`. It returns a list of `EvalQnA`, each with the question, the reference answer, the source passages, the RAGAS question type, the `doc_ids` and source paths of the documents the passages came from, and the `llm` and `llm_model` that wrote it.

`get_eval_qna()` first validates `EvalQnAArgs`, before any LLM call. The cache file is `{output folder}/eval_qna_{n_sample_documents}_{n_sample_evals}_{qna_llm_model}_{eval_qna_model}.json`, named by the model that writes the questions, so an eval set from one model is never loaded for another. It does not include the dataset size, so one eval set made once is reused at every corpus size (5k, 50k, 100k) as long as its source documents are in the corpus. If that file exists it is loaded and returned, even if it holds fewer items than asked for, and no LLM is checked or built, so a cached run needs no API key. Otherwise `validate_llm` checks both LLMs and `generate_eval_qna()`:

1. Keeps only documents with more than `EVAL_MIN_DOCUMENT_TOKENS` (500) tokens, counted with RAGAS's own counter (tiktoken `cl100k_base`), because RAGAS only extracts headlines and summaries from documents over 500 tokens. It raises an error if fewer documents than `n_sample_documents` are left.
2. Sorts those doc_ids and picks `n_sample_documents` of them with a fixed seed (`SAMPLE_DOCUMENTS_SEED` = 42), so the same documents are picked every run.
3. Builds the two LLMs with `llm.get_llm()`, wraps each one's `json_chat_model` for RAGAS (RAGAS parses every reply as JSON), and turns each picked document into a LangChain `Document`.
4. Builds the RAGAS knowledge graph with the knowledge graph LLM (`claude-haiku-5-5` in the project): headlines, splitting into chunks, summaries, a filter that drops chunks not useful for questions, themes, named entities, and the links between related chunks. The graph is saved to `{output folder}/eval_kg_{n_sample_documents}_{qna_kg_llm_model}_{eval_qna_model}.json` and loaded from there on later runs, so a failed or repeated run does not pay to build it again. Its name has no `n_sample_evals`, so more questions can later be generated from the same documents and only the question writing is paid.
5. Runs the RAGAS `TestsetGenerator` on the graph with the question LLM (`claude-sonnet-5-5` in the project) to write the personas, scenarios, questions and answers. After generation it prints the calls, tokens and cost of each LLM and the `QnA total cost` (see `llm`).
6. Raises an error if RAGAS generated nothing, so no empty file is saved. If it generated fewer than `n_sample_evals`, it prints a warning and keeps the ones generated.
7. Converts each RAGAS sample into an `EvalQnA`, with `llm` and `llm_model` from `qna_llm_args`. A sample's `doc_ids` are the picked documents that contain the start of one of its passages: the first `EVAL_PASSAGE_MATCH_CHARS` (80) characters, after removing the `<N-hop>` marker RAGAS puts in front of each multi-hop passage and collapsing whitespace in both texts. Only the start is matched because RAGAS can change the rest of a passage (see the 5000-document run below).

The list is then saved to the cache file, written to a temporary file first and renamed into place.

### Architecture

- **`qna.py`** — Loads the cached eval QnA, or generates it with RAGAS and the LLM generator from a sample of documents and saves it.
- **`schema.py`** — Defines `EvalQnA`, the layer's output record.
- **`validate.py`** — Checks the eval QnA parameters and raises an error for the first invalid one, before any LLM call.

The LLMs come from the `llm` module (see below). The cache file path is built in `src/common/paths.py`. QnA works with all four providers; `scripts/check_llms.sh` generates a small eval set with each one (see `llm`). Ollama only works on documents that fit its 32768-token context window, so the project's eval set, from documents up to tens of thousands of tokens, is made with Claude.

### Run results

**5000 documents, 100 sample documents, 200 eval QnA: the eval set used for the project** (from `log/qna/qna_5k_100_200_log.txt`). This is the eval set the retrieval and generation layers are scored against. It was made from the same 5000 `pixparse/pdfa-eng-wds` documents the build phase used, through a one-off script that calls `get_eval_qna()` with `--eval_qna_llm claude --eval_qna_model all-MiniLM-L6-v2 --n_sample_documents 100 --n_sample_evals 200` (the arguments at the time; the models were then fixed in the code to `claude-haiku-5-5` for the knowledge graph and `claude-sonnet-5-5` for the questions, the same as `QNA_KG_LLM_MODEL` and `QNA_LLM_MODEL` in `eval.sh` now). l1 loaded all 5000 documents from its cache in 8.529310777000319s.

| | |
|---|---|
| Documents with more than 500 tokens | 4028 of 5000 |
| Documents sampled for RAGAS | 100 of 4028 |
| Eval QnA generated | 200 of 200 (100 `single_hop_specific_query_synthesizer`, 100 `multi_hop_specific_query_synthesizer`) |
| Duplicate questions | 0 |
| Source documents the questions come from | 54 of the 100 sampled |
| Questions linked to 1 document / 2 documents | 190 / 10 (all 10 are multi-hop) |
| Question length in words, min / median / max | 6 / 28 / 144 |
| Saved to | `outputs/pdfa-eng-wds/eval_qna_100_200_claude_all-MiniLM-L6-v2.json`, renamed to `eval_qna_100_200_claude-sonnet-5-5_all-MiniLM-L6-v2.json` when cache names moved to the model name. The items got `llm` `claude` and `llm_model` `claude-sonnet-5-5` at the same time, added in place with no LLM call |

Cost, as printed by `compute_cost()`:

| Model | Role | Calls | Input tokens | Output tokens | Cost |
|---|---|---|---|---|---|
| `claude-haiku-5-5` | Knowledge graph | 1976 | 4677367 | 851692 | $0.8936 |
| `claude-sonnet-5-5` | Personas, scenarios, questions and answers | 645 | 866331 | 225362 | $3.9863 |
| **Total** | | 2621 | 5543698 | 1077054 | **$4.8799** |

RAGAS stage times, as printed by its progress bars:

| Stage | Model | Items | Time |
|---|---|---|---|
| HeadlinesExtractor | Haiku | 100 documents | 02:01 |
| HeadlineSplitter | none | 100 documents | 00:00 |
| SummaryExtractor | Haiku | 108 nodes | 02:01 |
| CustomNodeFilter | Haiku | 621 chunks | 09:51 |
| EmbeddingExtractor | local `all-MiniLM-L6-v2` | 108 summaries | 00:54 |
| ThemesExtractor | Haiku | 572 chunks | 09:39 |
| NERExtractor | Haiku | 572 chunks | 09:54 |
| CosineSimilarityBuilder | none | 1 | 00:00 |
| OverlapScoreBuilder | none | 1 | 00:08 |
| Generating personas | Sonnet | 3 | 00:04 |
| Generating Scenarios | Sonnet | 2 question types | 17:16 |
| Generating Samples | Sonnet | 200 questions | 03:15 |

The stage times add up to about 55 minutes; the log does not print a total.

- **Why two models.** In the earlier 10-document run (below), where every call used Sonnet, building the knowledge graph made almost all of the 101 calls and most of the cost: each call sends a whole document or chunk and gets back a short JSON answer, so the work is large inputs and simple extraction. That work does not need the strongest model, so it was moved to Claude Haiku 5.5, which costs $0.10 / $0.50 per million input / output tokens against $2.00 / $10.00 for Sonnet. The questions and answers, which set the quality of the benchmark, stay on Sonnet. In this run Haiku made 1976 of the 2621 calls (75%) and read 4677367 of the 5543698 input tokens (84%), for $0.8936 of the $4.8799. At Sonnet prices the same Haiku token counts would have cost $17.8717 ($9.3547 input and $8.5169 output), so the whole run would have been about $21.86 on Sonnet alone instead of $4.88.
- **Where the time goes.** The 100 documents were split into 621 chunks, and the filter kept 572 of them. The filter, themes and entity steps each send every chunk to Haiku, so together they take about 29 minutes. Generating Scenarios is the longest single stage at 17:16: Sonnet plans 100 single-hop and 100 multi-hop questions, choosing chunks, themes and personas for each, and its progress bar only counts the 2 question types. Writing the 200 questions and answers then took 03:15.
- **Where the money goes.** Sonnet is $3.9863 of the $4.8799 even with only 645 calls, and $2.2536 of that is output: Sonnet 5.5 thinks before answering, and thinking is billed as output tokens.
- **Skipped nodes.** RAGAS printed `Property 'summary' already exists in node ...` (and the same for `summary_embedding`) 17 times; those nodes already had the property from an earlier step, so it was not computed again.
- **Questions come from 54 of the 100 documents.** RAGAS builds questions from chunks that are well linked by shared themes and entities, so some documents get several questions and others none.
- **Questions imitate the personas.** RAGAS writes each question in the voice of one of 3 generated personas, so some are casual and include spelling mistakes (for example *"Is Tampa Bay Intergroup takeing 7th Tradition donashuns thru PayPal, and why do online groops use it?"*) and some are keyword lists (for example *"SPC-4 buildings inpatient beds by service and unit type Section 130061(e) BLD-01212 BLD-01214 bed counts"*).

**11 items repaired locally after the run.** As generated, 11 of the 200 items (`eval_qna_1`, `6`, `12`, `61`, `68`, `84`, `115`, `176`, `178`, `184`, `197`) had empty `doc_ids`. Their passages matched a sampled document exactly for the first 89 to 2380 characters and then continued with text from another part of the same document. RAGAS's `HeadlineSplitter` builds chunks by cutting the text at headline positions and then merging chunks that are too short, which can join sections that are not next to each other, so the whole passage is not a substring of the document even with whitespace collapsed. The matching was changed to use only the first 80 characters of each passage (`EVAL_PASSAGE_MATCH_CHARS`), and the saved file was repaired without any API call: the same 100-document sample was rebuilt from the l1 cache (same token filter, sort and seed) and the new matching was run on the saved passages. Exactly those 11 items changed, each now linked to one document, and the other 189 kept the same `doc_ids`, so the shorter match added no wrong documents. All 200 items now have `doc_ids`. The file as generated is kept as `eval_qna_100_200_claude_all-MiniLM-L6-v2.json.bak`.

This run was made before the knowledge graph was saved to disk, so its graph was not kept.

**Planned use.** The 200 questions will be split once, with a fixed seed, into 150 for tuning and 50 kept back for one final check per phase: the retrieval phase (query, retrieval, retrieval quality) with no API calls, then the generation phase (context, prompt, generation, answer quality) with paid calls. Because the cache file name has no dataset size, the same 200 questions will be reused for the 50k and 100k document runs, as long as their source documents are in those corpora.

**Earlier test run: 100 documents, 10 sample documents, 2 eval QnA, every call on Sonnet** (from `log/qna_test_log.txt`). This run was made before the Haiku / Sonnet split and before the dataset size was dropped from the cache file name: every Claude call, including the knowledge graph steps, used `claude-sonnet-5-5`, and its cache file has since been deleted. It called `get_eval_qna()` from a one-off script, not through `src/eval.py`, with the `scripts/eval.sh` dataset settings (`--dataset_size 100`, `pixparse/pdfa-eng-wds`) and `--eval_llm claude-sonnet-5-5 --eval_qna_model all-MiniLM-L6-v2 --n_sample_documents 10 --n_sample_evals 2`. l1 loaded all 100 documents from its cache in 10.647662741000204s. There was no eval QnA cache file, so the QnA was generated:

| | |
|---|---|
| Documents with more than 500 tokens | 81 of 100 |
| Documents sampled for RAGAS | 10 of 81 |
| Claude calls | 101 |
| Claude input tokens, cost | 219268, $0.4385 |
| Claude output tokens, cost | 14055, $0.1406 |
| Claude total cost | $0.5791 |
| Eval QnA generated | 2 of 2 |

RAGAS stage times, as printed by its progress bars:

| Stage | Items | Time |
|---|---|---|
| HeadlinesExtractor | 10 | 00:15 |
| HeadlineSplitter | 10 | 00:00 |
| SummaryExtractor | 14 | 00:17 |
| CustomNodeFilter | 24 | 00:24 |
| EmbeddingExtractor | 14 | 00:07 |
| ThemesExtractor | 23 | 01:06 |
| NERExtractor | 23 | 00:23 |
| CosineSimilarityBuilder | 1 | 00:00 |
| OverlapScoreBuilder | 1 | 00:00 |
| Generating personas | 3 | 00:05 |
| Generating Scenarios | 2 | 00:04 |
| Generating Samples | 2 | 00:07 |

The two generated items are one `single_hop_specific_query_synthesizer` question (a share option scheme field in a GEM monthly return) and one `multi_hop_specific_query_synthesizer` question (an LED and fluorescent fixture warranty and return policy). Each was matched to one source document. Both passages of the multi-hop question come from the same document. They were saved to `outputs/pdfa-eng-wds/eval_qna_100_10_2_claude-sonnet-5-5_all-MiniLM-L6-v2.json`. The log does not print a total elapsed time for the run.

- **Building the knowledge graph takes most of the time and calls.** Before writing any question, RAGAS sends each document or chunk to Claude several times: to extract headlines, then (after `HeadlineSplitter` splits the 10 documents into chunks) to summarize, to filter out chunks that are not useful for questions, and to extract themes and named entities. ThemesExtractor is the longest stage at 01:06 over 23 chunks. Writing the questions themselves (personas, scenarios and samples) took 00:05, 00:04 and 00:07. The log prints only the total of 101 Claude calls, not the calls per stage.
- **SummaryExtractor skipped 4 of its 14 nodes.** RAGAS printed `Property 'summary' already exists in node ...` for 4 nodes, which already had a summary from an earlier step.
- **Input tokens are most of the cost.** Each call sends a full document or chunk text (219268 input tokens in total) and gets back a short JSON answer (14055 output tokens). So input is $0.4385 of the $0.5791 even though input tokens cost a fifth as much as output tokens for `claude-sonnet-5-5` ($2 and $10 per million).
- **Stages that make no Claude call are fast.** HeadlineSplitter, CosineSimilarityBuilder and OverlapScoreBuilder do only local text splitting and score calculations, so they show 00:00. EmbeddingExtractor runs `all-MiniLM-L6-v2` locally on the 14 summaries and took 00:07.

## l5_query

### Overview

`l5_query` turns each eval question into the queries the retrieval layer searches with. `src/eval.py` calls `get_queries(output_path, eval_qna, eval_qna_args, query_args, query_llm_args)` after loading the eval QnA. `QueryArgs` holds the build's embedding model (`embedding_model`, taken from `--model_name` so it always matches the chunk embeddings) and `--n_multi_qus` (3). `query_llm_args` is an `LLM` object (see `llm`) made from `QUERY_LLM`, `QUERY_LLM_MODEL` and its prices in `eval.sh`; it can be any of the four providers, and the project uses `llama3.1:8b` on Ollama. It returns one `Query` per eval question, with the same `id` as its `EvalQnA` and the `llm` and `llm_model` that wrote it, holding four ways of searching for it: the original question (`original_qus`), an LLM rewrite (`rewritten_qus`), `n_multi_qus` LLM phrasings (`multi_qus`) and a HyDE hypothetical answer passage (`hyde_ans`). Each question (`Qus`) has its text, its BM25 tokens (lowercased and split on whitespace, the same as the l4 BM25 chunks) and its embedding; the HyDE answer (`Ans`) has its text and embedding only, because it is searched by embedding alone.

`get_queries()` first validates `QueryArgs`. The cache folder is `{output folder}/queries/{n_sample_documents}_{n_sample_evals}_{qna_llm_model}_{eval_qna_model}_{embedding_model}_{query_llm_model}_{n_multi_qus}`, with `/` and `:` in model names replaced by `-`. `qna_llm_model` is read from the eval QnA records, which hold the model that wrote them, so queries made for one eval set are never loaded for another. The name does not include the dataset size, so one set of queries is reused at every corpus size. If the folder holds both cache files they are loaded and returned, and no LLM is checked or built, so a cached run needs no Ollama server or API key. Otherwise `validate_llm` checks the query LLM and `generate_queries()`:

1. Builds the query LLM with `llm.get_llm()`. With Ollama it runs locally at temperature 0, so the transformations cost nothing.
2. Makes three LLM calls per question with `generate("", prompt)`: the rewrite, the `n_multi_qus` phrasings (one per line) and the HyDE passage. If the phrasings reply does not have exactly `n_multi_qus` non-empty lines, the question gets no phrasings; if the HyDE reply still starts with "I don't know", it is stored as exactly `I don't know`. Both are printed and counted, and the run goes on. It prints the LLM cost ($0 for Ollama).
3. Embeds every question and answer with `embedding_model` (a SentenceTransformer, normalized to length 1 like the l3 chunk embeddings).
4. Builds one `Query` per question.

The queries are then saved as `queries.json` (the texts and tokens, readable) and `embeddings.npz` (the float32 vectors), each written to a temporary file first and renamed into place.

**HyDE and hallucination.** The HyDE answer is written by the LLM without seeing the documents, and the prompt asks for a plausible passage even when the LLM does not know the answer, so it can hold invented facts. It is only used as a search probe: it is embedded and searched with, and is never shown or used as an answer. The risk is in retrieval, not in the answer: invented names, numbers or claims can pull in wrong chunks, which the retrieval evaluation measures. If the HyDE text is ever passed to a later layer (for example into the l8 prompt or l9 generation), it can bring those invented facts into the generated answer and cause hallucination, so later layers must use only the retrieved chunks, never `hyde_ans`.

### Architecture

- **`query.py`** — Loads the cached queries, or transforms each eval question with the query LLM, embeds the results and saves them.
- **`schema.py`** — Defines `Qus` (a searchable question), `Ans` (the HyDE answer) and `Query`, the layer's output record.
- **`validate.py`** — Checks `embedding_model` and `n_multi_qus` and raises an error for the first invalid one, before any LLM call. The query LLM is checked by `llm.validate.validate_llm`.
- **`cache.py`** — Saves and loads the queries of one run as `queries.json` and `embeddings.npz`.
- **`prompt.py`** — The rewrite, multi-query and HyDE prompt templates.

The query LLM comes from the `llm` module. The cache folder path is built in `src/common/paths.py`. Query works with all four providers; `scripts/check_llms.sh` generated a small set of queries with each one (see `llm`).

### Run results

**200 eval questions, llama3.1:8b, 3 phrasings each** (from `log/l5_query/query_5k_100_200_log.txt`). The input is the 200-question eval set made from the 5000 `pixparse/pdfa-eng-wds` documents (see qna), loaded from its cache file `eval_qna_100_200_claude_all-MiniLM-L6-v2.json` (now `eval_qna_100_200_claude-sonnet-5-5_all-MiniLM-L6-v2.json`). Settings: `embedding_model` `all-MiniLM-L6-v2`, `query_llm` `ollama`, `query_llm_model` `llama3.1:8b`, `n_multi_qus` 3. There was no query cache, so the queries were generated. At the time, Ollama was called through its `/api/generate` endpoint with replies cut at 400 tokens; it now runs through `ChatOllama` with the shared 16000-token limit (see `llm`). These 200 queries are kept as they are and loaded from the cache, never generated again.

| | Value |
|---|---|
| Eval QnA loaded | 200 |
| Questions transformed | 200 of 200 |
| Multi-query I don't know | 0 of 200 |
| HyDE I don't know | 0 of 200 |
| LLM cost | $0.0000 |
| Queries saved | 200, to `outputs/pdfa-eng-wds/queries/100_200_claude_all-MiniLM-L6-v2_all-MiniLM-L6-v2_ollama_llama3.1-8b_3`, renamed to `100_200_claude-sonnet-5-5_all-MiniLM-L6-v2_all-MiniLM-L6-v2_llama3.1-8b_3` when cache names moved to the model name. The queries got `llm` `ollama` at the same time, added in place with no LLM call |
| Query time | 763.0530879497528s |

The log has no per-stage timing, only the total. Nearly all of it is the LLM: each question makes three calls to `llama3.1:8b`, one after another, so 600 calls in total, each generating its reply token by token on the GPU. Embedding the results is a single batched SentenceTransformer pass per query type over short texts, and loading the eval set and saving the two cache files are file reads and writes, so they are small next to the 600 generations.

## l6_retrieval

### Overview

`l6_retrieval` finds the chunks for each eval question. `src/eval.py` splits the 200 queries with a fixed seed (`TUNING_SPLIT_SEED` 42) into `--tuning_queries_size` (150) tuning queries and 50 held-out queries, and only the tuning queries are retrieved. `RetrievalArgs` holds `--framework`, `--retrieval_types` (any of `dense`, `bm25`, `hybrid`), `--top_k` (20), `--reranking` (`true` or `false`) and `--rerank_top_k` (5). It then loops one item at a time:

1. For each chunk type in `--chunk_types`, it loads that chunk type's l2 chunks and creates one `Retrieval(retrieval_args, chunking_args, index_args, tuning_queries_size, output_path)` for that chunk type. `index_args` holds the run's index settings with an empty `index_type`; each method sets its own index type when it builds its cache folder.
2. For each index type in `--index_types`, it loads that one l4 index with `index_cache.load_index()` (load-only, never builds) and runs:
   - **Dense** (`compute_dense_retrieval`) on `faiss_flatip`, `faiss_ivf`, `faiss_hnsw` and `qdrant`. It searches with each query type's embedding and keeps the top_k chunks. The score is cosine similarity. The multi-query phrasings are searched one by one and merged, keeping each chunk's best score. A HyDE answer stored as `I don't know` is not searched.
   - **BM25** (`compute_bm25_retrieval`) on `bm25`. It scores every chunk with the query tokens and keeps the top_k chunks with a score above 0. HyDE has no tokens, so BM25 has no HyDE record.
   - Each method returns `[]` when its type is not in `--retrieval_types` or the index is the wrong kind, so the loop needs no type checks.
3. After the index loop, **hybrid** (`compute_hybrid_retrieval`) fuses each dense index's outputs with the BM25 outputs of the same question and query type, using reciprocal rank fusion. A chunk gets `1 / (RRF_K + rank)` from each list it is in, with `RRF_K` 60, and the top_k by fused score are kept. Ranks are used, not scores, because dense and BM25 scores are on different scales. `index_type` is written as e.g. `faiss_flatip+bm25`.
4. With `--reranking true`, each plain output list is reranked separately (`get_reranked_dense_retrieval`, `get_reranked_bm25_retrieval`, `get_reranked_hybrid_retrieval`). The cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` (`CROSS_ENCODER_MODEL`, loaded in `Retrieval.__init__`, so once per chunk type) scores each chunk's text against the original eval question, and the best `rerank_top_k` are kept. Every query type is reranked against the original question, because that is what the answer must match. The reranked records are a separate list from the plain ones, and hybrid always fuses the plain top_k lists.

Each output is a list of `Retrieved`, one per eval question and query type:
- **Setup fields:** `id` (the `EvalQnA` id), `query_type`, `chunk_type`, `retrieval_type`, `index_type`, `top_k`, `reranking`, `rerank_top_k`.
- **Result fields:** the ranked `chunk_ids`, their `chunk_scores` and `doc_ids`. The doc id is the part of the chunk id before the last `_`.

Records per index, for 150 tuning queries:

| Retrieval | Query types | Records |
|---|---|---|
| Dense | original, rewritten, multi, hyde | 600 |
| BM25 | original, rewritten, multi | 450 |
| Hybrid | original, rewritten, multi (no BM25 for HyDE) | 450 |

A query type with nothing to search (no multi-query phrasings, or a HyDE `I don't know`) still gets a `Retrieved` with no chunks, so it counts as a miss.

**Cache.** Every method first looks for its saved outputs and returns them if found. Otherwise it computes them, saves them and returns them. `eval.py` passes only `output_path`; each method builds its own folder:
- **Dense or BM25 index:** `{output folder}/retrieved/{chunk_type}/{model_name}/{min}_{max}_{overlap}_{threshold}/{index}/{tuning_queries_size}/`. `{index}` is `flatip`, `ivf_{nlist}`, `hnsw_{m}_{ef_construction}`, `bm25_{k1}_{b}_{tokenizer}` or `qdrant`.
- **Hybrid pair:** the same, with the two index folders joined by `+`, e.g. `flatip+bm25_1.5_0.75_whitespace`.
- **LangChain:** a `langchain/` folder after `retrieved/`.
- **Files:** `{retrieval_type}_top{top_k}.json` holds the plain outputs. `{retrieval_type}_top{top_k}_rerank{rerank_top_k}_{cross-encoder}.json` holds the reranked ones. Each is written to a temporary file first and then renamed into place.

The folder name does not include the l5 queries or the split seed, so delete `retrieved/` after regenerating the queries or changing `TUNING_SPLIT_SEED`. The indexes are still loaded on a cached run, so the loop keeps the same shape.

With `--framework langchain`, `Retrieval` hands dense and BM25 to the `Langchain` class, which searches the LangChain indexes with the same l5 query embeddings and tokens:
- **FAISS and Qdrant:** `similarity_search_with_score_by_vector`.
- **BM25:** the retriever's rank_bm25 model directly (`vectorizer.get_scores`), because `BM25Retriever.invoke` returns no scores.

The records are built by the same functions as the built-in ones, so hybrid, reranking and evaluation work unchanged, and only the chunking and indexing differ between the two frameworks. LangChain's own query transformers (`MultiQueryRetriever`, `HypotheticalDocumentEmbedder`) are not used: they would call an LLM again and give each framework different queries. `Retrieved` does not record the framework, because one run uses one framework.

### Architecture

- **`retrieval.py`**: `Retrieval`, the orchestrator. It validates the args, creates `Langchain` for `--framework langchain` and loads the cross-encoder for `--reranking true`. It has one method per retrieval type (`compute_dense_retrieval`, `compute_bm25_retrieval`, `compute_hybrid_retrieval`) and one per reranked list.
- **`validate.py`**: checks `retrieval_types`, `top_k` ≥ 1, `rerank_top_k` ≥ 1 and `rerank_top_k` ≤ `top_k`.
- **`schema.py`**: defines `Retrieved`, the layer's output record.
- **`retrievars/dense.py`**: searches the built-in FAISS index or the Qdrant collection with each query type's embeddings and builds the dense `Retrieved` records.
- **`retrievars/bm25.py`**: scores the built-in BM25 model with each query type's tokens and builds the BM25 `Retrieved` records.
- **`retrievars/hybrid.py`**: matches dense and BM25 records by question and query type and fuses them with reciprocal rank fusion.
- **`rerank.py`**: scores every (question, chunk text) pair with the cross-encoder in one batched call and keeps the best `rerank_top_k` per record.
- **`langchain.py`**: `Langchain`, dense and BM25 search on the LangChain indexes.
- **`cache.py`**: saves and loads one retrieval's `Retrieved` list as one JSON file.

The retrieved folder paths are built in `src/common/paths.py`. Retrieval quality (hit, precision, recall, MRR, nDCG at chunk and doc level) is measured in `l10_evaluation`, not here.

### Run results

**5000 documents, fixed chunks, 150 tuning queries, no reranking.** These come from two test runs whose logs were not saved under `log/`. Both used `--index_types faiss_flatip bm25 qdrant` and `--retrieval_types dense bm25 hybrid`. The times are those printed by `src/eval.py`.

| | Built-in | LangChain |
|---|---|---|
| Dense on faiss_flatip | 600 records, 7s | 600 records, 10s |
| BM25 | 450 records, 361s | 450 records, 385s |
| Dense on qdrant | 600 records, 5s | 600 records, 14s |
| Hybrid faiss_flatip+bm25 | 450 records, 0.02s | 450 records, 0.01s |
| Hybrid qdrant+bm25 | 450 records, 0.01s | 450 records, 0.01s |

- **BM25 is nearly all of the time.** rank_bm25's `get_scores` scores every one of the 91670 chunks for every token list in Python, 750 lists for 150 questions: 150 original + 150 rewritten + 450 multi-query phrasings. FAISS and Qdrant search are vector lookups, so dense takes seconds.
- **Hybrid** only merges two lists of 20 chunk ids per record, so it takes no measurable time.

**5000 documents, all chunk types and built-in indexes, 150 tuning queries** (from `log/l6_retrieval/log_5k_retrieval_quality.txt`, `log_5k_retrieval_quality_rerank.txt` and `log_5k_retrieval_quality_rerank_cached.txt`). All runs used `--chunk_types fixed sentence semantic`, `--index_types faiss_flatip faiss_ivf faiss_hnsw bm25 qdrant` and `--retrieval_types dense bm25 hybrid`. The first run was made before the cache existed. The second filled it. The third loaded everything from it.

| Run | Reranking | Total time |
|---|---|---|
| No cache | off | 1273.65s |
| Fills the cache | on | 1868.99s |
| From the cache | on | 33.31s |

- **BM25 is still nearly all of the retrieval time:** 362.30s, 399.13s and 456.13s for fixed, sentence and semantic in the first run. It grows with the chunk count (91670, 94053, 104900).
- **Reranking added about 595s**, the difference between the first two runs. The cross-encoder scores 20 chunks for every plain record: 4650 records per chunk type.
- **From the cache**, all 54 files load (plain and reranked dense on 4 indexes, BM25, and hybrid on 4 pairs, for 3 chunk types). The run takes 33.31s, most of it loading the l1 documents and the 15 indexes, and it prints the same quality table.

**5000 documents, LangChain, all chunk types and indexes, reranking on, 150 tuning queries** (from `log/l6_retrieval/log_5k_langchain_retrieval_quality_rerank.txt`). Same settings as the built-in runs, with `--framework langchain`. There was no LangChain cache, so this run computed and saved all 54 files under `retrieved/langchain/`.

| | fixed | sentence | semantic |
|---|---|---|---|
| Chunks, built-in / LangChain | 91670 / 91670 | 94053 / 94053 | 104900 / 112218 |
| BM25 retrieval time, built-in → LangChain | 359.73s → 378.92s | 404.23s → 405.31s | 466.89s → 474.48s |

- **Total: 1997.63s**, against 1868.99s for the built-in run with reranking.
- **BM25 still takes most of the time.** The LangChain BM25 search scores the retriever's own rank_bm25 model the same way the built-in one does, so the times are close. Semantic has 7318 more LangChain chunks, which is part of why it is the slowest.

## l7_context

### Overview

`l7_context` turns each filtered `Retrieved` into the text the LLM gets as context. It runs in `src/eval.py` right after the retrieval filter (see l10_evaluation), one chunk type at a time, so only that chunk type's chunks are in memory:

1. For each chunk type in `--chunk_types`, `eval.py` collects the filtered `Retrieved` of that chunk type and skips the type if there are none.
2. It loads that chunk type's l2 chunks with `load_chunks(output_path, chunking_args, doc_ids)`.
3. It calls `get_context_list(retrieved_outputs, chunks)`, which returns one `Context` per `Retrieved`, in the same order.

`eval.py` adds the records and their contexts to `filtered_retrieved_list` and `context_list` in the same loop, so `context_list[i]` always belongs to `filtered_retrieved_list[i]`. From l7 to l10 every layer keeps this rule: one output per input, in the same order. Where a layer has nothing to produce, it puts an empty placeholder in that place instead of dropping it, so the lists never shift.

`Context` holds:
- **`id`:** the eval question id. It is the one field kept from `Retrieved`. It lets the prompt layer find the question, and it can check that two lists line up.
- **`context`:** the text sent to the LLM.
- **`token_count`:** the sum of the chunks' l2 `size`. This is an estimate in embedding-model tokens (all-MiniLM-L6-v2), not the generation LLM's own count.

The setup (chunk type, retrieval type, index, query type, reranking) is not copied. It is in the `Retrieved` at the same position.

`get_context_list` joins the chunk texts in rank order, best first, with one blank line between chunks. It uses all of a `Retrieved`'s chunks: 20 for a plain one, 5 for a reranked one. A `Retrieved` with no chunks gets `Context(id, "", 0)`, so it keeps its place.

- **Not cached.** Building a context only looks up chunk texts that are already loaded and joins strings, so it takes a fraction of a second. A cache would only add files that go stale when the format changes.
- **Fixed-chunk overlap is not removed.** Fixed chunks overlap by `--fixed_chunk_overlap` (32) tokens with the next chunk of the same document. If both are retrieved, those tokens appear twice in the context. That is about 12% of one chunk, and the current filtered setups use sentence chunks, which have no overlap. Trimming the overlap or merging neighbouring chunks can be added if fixed setups are ever chosen.

### Architecture

- **`context.py`**: `get_context_list`, builds one `Context` per `Retrieved` from the chunk texts.
- **`schema.py`**: defines `Context`.

### Run results

**5000 documents, built-in, filtered retrieval** (from `log/l7_context/log_5k_context.txt`). The run loads all retrieved outputs from the cache. The filter keeps the plain sentence hybrid faiss_flatip+bm25 setup and the reranked sentence hybrid qdrant+bm25 setup, both on the original question: 300 `Retrieved`.

| | Value |
|---|---|
| Chunk types loaded | sentence (both kept setups) |
| Contexts built | 300 |
| Empty contexts | 0 |
| Context time (loading the sentence chunks and building all 300) | 0.30s |
| Total run time | 37.13s |

`token_count` of the 150 contexts per setup:

| Setup | Chunks | Mean | Min | Max |
|---|---|---|---|---|
| Plain (k=20) | 20 | 4412 | 2740 | 4997 |
| Reranked (k=5) | 5 | 1107 | 391 | 1249 |

A plain context is about 4 times the size of a reranked one, so each plain prompt will cost about 4 times as many input tokens in generation.

## l8_prompt

### Overview

`l8_prompt` turns each l7 `Context` into the prompt the generation LLM gets in l9. It runs in `src/eval.py` as `run_prompt(context_list, tuning_queries)`, right after l7, and calls `get_prompt_list(context_list, tuning_queries)`, which returns one `Prompt` per `Context`, in the same order, so `prompt_list[i]` belongs to `context_list[i]` and `retrieved_list[i]`.

1. It stores each tuning query's original question by its id, in one loop over `tuning_queries`.
2. For each `Context`, it finds its question by `context.id`. The l5 `original_qus` is used for every query type, because the rewritten, multi and HyDE forms are only for searching. The LLM answers the question as asked.
3. It fills `USER_PROMPT_TEMPLATE` with the context text and the question, and builds the `Prompt`.

The eval QnA is not used here. Its ground truth is only needed again in l10. The `Retrieved` is not needed either: `Context` already holds the id, and the setup stays in `retrieved_list` at the same position.

`Prompt` holds:
- **`id`:** the eval question id, the same as the `Retrieved`, `Context`, `Query` and `EvalQnA` id.
- **`qus`:** the original question, so l9 does not need the queries again.
- **`system_prompt`:** the instructions, the same for every prompt.
- **`user_prompt`:** the context and the question.
- **`token_count`:** the context's `token_count` plus the word count of the system prompt and the question. This is an estimate: the context part is in embedding-model tokens (all-MiniLM-L6-v2) and the rest in words, not the generation LLM's own count.

The system and user prompts are kept apart because the chat APIs of Claude, OpenAI and Gemini take a system prompt separately from the user message.

The system prompt tells the LLM to answer only from the context and to reply with exactly `I don't know` (`constants.I_DONT_KNOW`, the same constant l5 uses) when the context does not hold the answer, so l10 can find those answers by an exact match. The user prompt is:

```
Context:
{context}

Question: {question}
```

- **Every question appears twice.** The 300 prompts are the 150 tuning questions, once for each kept setup. The question is the same, but the context differs: 20 chunks for the plain setup, 5 for the reranked one.
- **An empty context gets an empty user prompt.** A `Context` with no text still gets a `Prompt`, so the lists stay lined up, but its `user_prompt` is `""` and its `token_count` is 0, so l9 can skip the LLM call. None of the current 300 are empty, because both kept setups are hybrid, and dense retrieval always returns `top_k` chunks.
- **Not cached.** Building a prompt only fills a template, so it takes a few milliseconds for all 300.
- **No provider prompt caching.** The only part that repeats across calls is the short system prompt, which is below the minimum cacheable size. The context, which is the long part, is different in every prompt.

### Architecture

- **`prompt.py`**: `get_prompt_list`, builds one `Prompt` per `Context` with its original question.
- **`prompt_template.py`**: `SYSTEM_PROMPT` and `USER_PROMPT_TEMPLATE`.
- **`schema.py`**: defines `Prompt`.

### Run results

**5000 documents, built-in, filtered retrieval** (from `log/l8_prompt/log_5k_prompt.txt`). The run loads all retrieved outputs from the cache. The filter keeps the same two sentence setups as in l7: 300 `Retrieved`, 0 duplicates skipped.

| | Value |
|---|---|
| Tuning queries | 150 |
| Contexts in | 300 |
| Prompts built | 300 |
| Prompt time | 0.0019s |
| Total run time | 38.40s |

Prompt time is about 2 ms for 300 prompts, because each prompt is one dict lookup and one template fill. The 38.40s total goes to the steps before l8. The largest timed ones are loading the l1 documents (9.37s) and loading the l2 chunks and l4 indexes for each chunk type (7.19s together). Retrieval from the cache takes 0.36s and l7 context takes 0.26s. The remaining time is not broken down in the log.

## l10_evaluation

### Overview

`l10_evaluation` measures the pipeline's quality. So far it has retrieval quality; answer quality comes after generation.

`src/eval.py` creates one `RetrievalEvaluation(eval_qna)` before the loops. It stores each eval question by its id. For every plain and reranked `Retrieved`, `compute_retrieval_quality(retrieved, chunks)` finds the question by `retrieved.id` and returns a `RetrievalQuality`. `eval.py` keeps the `(retrieved, quality)` pairs in `retrieval_qualities`. After the loops, `print_retrieval_summary()` groups them by setup (chunk type, retrieval type, index, query type, reranking) and prints the mean of each metric.

`RetrievalQuality` holds `k` and two `Quality` records:
- **`k`:** `top_k` (20) for a plain `Retrieved`, `rerank_top_k` (5) for a reranked one. Chunks missing from a shorter list count as not relevant.
- **`chunk_quality`:** a chunk is relevant if it comes from one of the question's `doc_ids` and at least `CHUNK_CONTEXT_OVERLAP_THRESHOLD` (0.5) of its distinct lowercase words are in one of the question's `context` passages. The passages are 34 to 875 words, with a median of 560, about three chunks long, so a relevant chunk sits inside a passage. Recall is the share of context passages found.
- **`doc_quality`:** a chunk is relevant if it comes from one of the question's `doc_ids`. Recall is the share of `doc_ids` found.

Each `Quality` has these fields:
- `hit`: 1 if any chunk is relevant.
- `precision`: relevant chunks ÷ k.
- `recall`.
- `mrr`: 1 ÷ rank of the first relevant chunk.
- `ndcg`: binary relevance, with the ideal order putting every relevant chunk found at the top.
- `first_relevant_rank`: 0 for a miss.

### Architecture

- **`retrieval_evaluation.py`**: `RetrievalEvaluation`, with `compute_retrieval_quality`, `compute_chunk_quality`, `compute_doc_quality` and `get_filtered_retrieval_qualities`.
- **`schema.py`**: defines `Quality` and `RetrievalQuality`.
- **`summary.py`**: `print_retrieval_summary`, the mean of each metric per setup.
- **`answer_quality.py`**: empty placeholder.

### Run results

**Retrieval quality, 5000 documents, built-in indexes, 150 tuning queries** (from `log/l6_retrieval/log_5k_retrieval_quality_rerank_cached.txt`, which has all 186 rows: 93 plain and 93 reranked). These are the best plain rows (k=20):

| Chunk | Retrieval | Index | Query | c_hit | c_recall | c_mrr | d_hit |
|---|---|---|---|---|---|---|---|
| sentence | hybrid | faiss_flatip+bm25 | original | 0.940 | 0.927 | 0.718 | 0.987 |
| sentence | hybrid | qdrant+bm25 | original | 0.940 | 0.927 | 0.716 | 0.987 |
| sentence | hybrid | faiss_hnsw+bm25 | original | 0.940 | 0.923 | 0.705 | 0.980 |
| fixed | hybrid | faiss_flatip+bm25 | original | 0.913 | 0.900 | 0.729 | 0.967 |
| semantic | hybrid | faiss_flatip+bm25 | original | 0.913 | 0.903 | 0.723 | 0.973 |

These are the best reranked rows (k=5):

| Chunk | Retrieval | Index | Query | c_hit | c_recall | c_mrr | d_hit |
|---|---|---|---|---|---|---|---|
| sentence | hybrid | faiss_flatip+bm25 | original | 0.853 | 0.837 | 0.768 | 0.947 |
| sentence | hybrid | faiss_flatip+bm25 | rewritten | 0.853 | 0.823 | 0.784 | 0.940 |
| semantic | hybrid | faiss_flatip+bm25 | rewritten | 0.853 | 0.827 | 0.778 | 0.920 |
| semantic | hybrid | faiss_hnsw+bm25 | rewritten | 0.853 | 0.827 | 0.779 | 0.920 |
| fixed | hybrid | faiss_flatip+bm25 | original | 0.840 | 0.817 | 0.771 | 0.927 |

- **Hybrid is best at every chunk type, plain or reranked.** Dense alone reaches about 0.87 chunk hit at k=20, and BM25 alone about 0.87.
- **faiss_flatip and qdrant give the same results.** Both are exact cosine search over the same vectors. faiss_hnsw is 0 to 5 points lower, because its search is approximate.
- **faiss_ivf is much lower:** 0.52 to 0.66 chunk hit for dense. The search uses FAISS's default `nprobe` of 1, so it looks in only 1 of the 256 clusters.
- **Query type.** Plain hybrid and BM25 do best with the original question. Plain dense does best with HyDE (fixed faiss_flatip: 0.873 vs 0.840 original). The rewritten question is often the worst plain, but after reranking it is at or near the top.
- **Chunk types are close.** At k=20 sentence leads by 2.7 points (0.940 vs 0.913), about 4 of 150 questions. After reranking, all three tie at 0.84 to 0.853.
- **Reranking to 5 chunks.** It keeps most of the hits: sentence hybrid original goes from 0.940 at k=20 to 0.853 at k=5. It puts the right chunk higher: MRR 0.718 → 0.768. Precision goes from 0.154 to 0.360, because only 5 chunks are kept.
- **Chunk level vs doc level.** Chunk-level hit is 4 to 9 points below doc-level hit. That is how often retrieval found the right document but not the part holding the answer.

<details>
<summary>All 93 plain rows (k=20)</summary>

| Chunk | Retrieval | Index | Query | k | n | c_hit | c_prec | c_rec | c_mrr | c_ndcg | d_hit | d_prec | d_rec | d_mrr | d_ndcg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed | bm25 | bm25 | multi | 20 | 150 | 0.820 | 0.108 | 0.797 | 0.612 | 0.624 | 0.880 | 0.280 | 0.870 | 0.744 | 0.738 |
| fixed | bm25 | bm25 | original | 20 | 150 | 0.867 | 0.119 | 0.843 | 0.699 | 0.703 | 0.913 | 0.320 | 0.903 | 0.811 | 0.799 |
| fixed | bm25 | bm25 | rewritten | 20 | 150 | 0.780 | 0.103 | 0.757 | 0.596 | 0.614 | 0.887 | 0.264 | 0.877 | 0.743 | 0.734 |
| fixed | dense | faiss_flatip | hyde | 20 | 150 | 0.873 | 0.129 | 0.847 | 0.670 | 0.685 | 0.947 | 0.378 | 0.930 | 0.827 | 0.817 |
| fixed | dense | faiss_flatip | multi | 20 | 150 | 0.860 | 0.125 | 0.833 | 0.662 | 0.664 | 0.933 | 0.354 | 0.920 | 0.787 | 0.777 |
| fixed | dense | faiss_flatip | original | 20 | 150 | 0.840 | 0.124 | 0.813 | 0.650 | 0.661 | 0.920 | 0.356 | 0.903 | 0.785 | 0.778 |
| fixed | dense | faiss_flatip | rewritten | 20 | 150 | 0.820 | 0.122 | 0.800 | 0.660 | 0.647 | 0.913 | 0.342 | 0.900 | 0.784 | 0.763 |
| fixed | dense | faiss_hnsw | hyde | 20 | 150 | 0.867 | 0.126 | 0.837 | 0.669 | 0.684 | 0.940 | 0.378 | 0.923 | 0.826 | 0.818 |
| fixed | dense | faiss_hnsw | multi | 20 | 150 | 0.853 | 0.125 | 0.820 | 0.662 | 0.660 | 0.927 | 0.360 | 0.910 | 0.787 | 0.773 |
| fixed | dense | faiss_hnsw | original | 20 | 150 | 0.800 | 0.122 | 0.780 | 0.623 | 0.630 | 0.893 | 0.356 | 0.870 | 0.754 | 0.753 |
| fixed | dense | faiss_hnsw | rewritten | 20 | 150 | 0.787 | 0.118 | 0.760 | 0.634 | 0.621 | 0.860 | 0.336 | 0.847 | 0.751 | 0.734 |
| fixed | dense | faiss_ivf | hyde | 20 | 150 | 0.600 | 0.086 | 0.580 | 0.470 | 0.475 | 0.720 | 0.258 | 0.710 | 0.641 | 0.630 |
| fixed | dense | faiss_ivf | multi | 20 | 150 | 0.607 | 0.091 | 0.583 | 0.476 | 0.478 | 0.753 | 0.265 | 0.743 | 0.634 | 0.629 |
| fixed | dense | faiss_ivf | original | 20 | 150 | 0.540 | 0.083 | 0.520 | 0.424 | 0.429 | 0.667 | 0.262 | 0.653 | 0.582 | 0.574 |
| fixed | dense | faiss_ivf | rewritten | 20 | 150 | 0.520 | 0.075 | 0.507 | 0.426 | 0.422 | 0.653 | 0.230 | 0.647 | 0.573 | 0.558 |
| fixed | dense | qdrant | hyde | 20 | 150 | 0.873 | 0.129 | 0.847 | 0.670 | 0.685 | 0.947 | 0.378 | 0.930 | 0.827 | 0.817 |
| fixed | dense | qdrant | multi | 20 | 150 | 0.860 | 0.125 | 0.833 | 0.662 | 0.664 | 0.933 | 0.354 | 0.920 | 0.787 | 0.776 |
| fixed | dense | qdrant | original | 20 | 150 | 0.840 | 0.123 | 0.810 | 0.651 | 0.661 | 0.920 | 0.356 | 0.900 | 0.785 | 0.778 |
| fixed | dense | qdrant | rewritten | 20 | 150 | 0.820 | 0.121 | 0.797 | 0.654 | 0.644 | 0.913 | 0.341 | 0.900 | 0.778 | 0.761 |
| fixed | hybrid | faiss_flatip+bm25 | multi | 20 | 150 | 0.880 | 0.131 | 0.863 | 0.703 | 0.688 | 0.947 | 0.351 | 0.943 | 0.812 | 0.798 |
| fixed | hybrid | faiss_flatip+bm25 | original | 20 | 150 | 0.913 | 0.138 | 0.900 | 0.729 | 0.721 | 0.967 | 0.371 | 0.963 | 0.856 | 0.834 |
| fixed | hybrid | faiss_flatip+bm25 | rewritten | 20 | 150 | 0.880 | 0.131 | 0.863 | 0.686 | 0.684 | 0.960 | 0.338 | 0.950 | 0.806 | 0.806 |
| fixed | hybrid | faiss_hnsw+bm25 | multi | 20 | 150 | 0.880 | 0.131 | 0.863 | 0.702 | 0.686 | 0.947 | 0.353 | 0.943 | 0.809 | 0.796 |
| fixed | hybrid | faiss_hnsw+bm25 | original | 20 | 150 | 0.913 | 0.136 | 0.897 | 0.709 | 0.710 | 0.960 | 0.370 | 0.957 | 0.834 | 0.822 |
| fixed | hybrid | faiss_hnsw+bm25 | rewritten | 20 | 150 | 0.873 | 0.130 | 0.857 | 0.667 | 0.668 | 0.953 | 0.336 | 0.943 | 0.781 | 0.791 |
| fixed | hybrid | faiss_ivf+bm25 | multi | 20 | 150 | 0.840 | 0.123 | 0.823 | 0.560 | 0.593 | 0.907 | 0.317 | 0.897 | 0.702 | 0.721 |
| fixed | hybrid | faiss_ivf+bm25 | original | 20 | 150 | 0.867 | 0.128 | 0.840 | 0.565 | 0.604 | 0.933 | 0.340 | 0.923 | 0.727 | 0.739 |
| fixed | hybrid | faiss_ivf+bm25 | rewritten | 20 | 150 | 0.833 | 0.117 | 0.810 | 0.512 | 0.567 | 0.913 | 0.296 | 0.900 | 0.673 | 0.692 |
| fixed | hybrid | qdrant+bm25 | multi | 20 | 150 | 0.880 | 0.131 | 0.863 | 0.703 | 0.688 | 0.947 | 0.351 | 0.943 | 0.812 | 0.798 |
| fixed | hybrid | qdrant+bm25 | original | 20 | 150 | 0.913 | 0.138 | 0.900 | 0.729 | 0.721 | 0.967 | 0.371 | 0.963 | 0.856 | 0.834 |
| fixed | hybrid | qdrant+bm25 | rewritten | 20 | 150 | 0.880 | 0.131 | 0.863 | 0.686 | 0.683 | 0.960 | 0.338 | 0.950 | 0.806 | 0.806 |
| semantic | bm25 | bm25 | multi | 20 | 150 | 0.820 | 0.120 | 0.793 | 0.620 | 0.626 | 0.893 | 0.261 | 0.887 | 0.741 | 0.733 |
| semantic | bm25 | bm25 | original | 20 | 150 | 0.860 | 0.138 | 0.830 | 0.686 | 0.679 | 0.907 | 0.307 | 0.900 | 0.803 | 0.784 |
| semantic | bm25 | bm25 | rewritten | 20 | 150 | 0.807 | 0.122 | 0.770 | 0.626 | 0.632 | 0.893 | 0.251 | 0.883 | 0.753 | 0.735 |
| semantic | dense | faiss_flatip | hyde | 20 | 150 | 0.867 | 0.152 | 0.843 | 0.701 | 0.696 | 0.953 | 0.372 | 0.933 | 0.827 | 0.807 |
| semantic | dense | faiss_flatip | multi | 20 | 150 | 0.847 | 0.150 | 0.833 | 0.655 | 0.654 | 0.913 | 0.354 | 0.907 | 0.781 | 0.765 |
| semantic | dense | faiss_flatip | original | 20 | 150 | 0.853 | 0.152 | 0.833 | 0.648 | 0.653 | 0.927 | 0.353 | 0.917 | 0.769 | 0.762 |
| semantic | dense | faiss_flatip | rewritten | 20 | 150 | 0.840 | 0.141 | 0.817 | 0.633 | 0.640 | 0.920 | 0.336 | 0.907 | 0.776 | 0.757 |
| semantic | dense | faiss_hnsw | hyde | 20 | 150 | 0.833 | 0.150 | 0.807 | 0.678 | 0.671 | 0.927 | 0.373 | 0.907 | 0.806 | 0.784 |
| semantic | dense | faiss_hnsw | multi | 20 | 150 | 0.820 | 0.150 | 0.803 | 0.654 | 0.646 | 0.907 | 0.355 | 0.900 | 0.780 | 0.765 |
| semantic | dense | faiss_hnsw | original | 20 | 150 | 0.800 | 0.147 | 0.780 | 0.610 | 0.613 | 0.873 | 0.351 | 0.857 | 0.728 | 0.724 |
| semantic | dense | faiss_hnsw | rewritten | 20 | 150 | 0.800 | 0.140 | 0.780 | 0.614 | 0.616 | 0.880 | 0.339 | 0.870 | 0.750 | 0.728 |
| semantic | dense | faiss_ivf | hyde | 20 | 150 | 0.607 | 0.094 | 0.560 | 0.503 | 0.496 | 0.747 | 0.248 | 0.723 | 0.657 | 0.638 |
| semantic | dense | faiss_ivf | multi | 20 | 150 | 0.660 | 0.103 | 0.633 | 0.535 | 0.528 | 0.787 | 0.246 | 0.773 | 0.648 | 0.643 |
| semantic | dense | faiss_ivf | original | 20 | 150 | 0.567 | 0.086 | 0.537 | 0.466 | 0.453 | 0.713 | 0.219 | 0.700 | 0.598 | 0.587 |
| semantic | dense | faiss_ivf | rewritten | 20 | 150 | 0.573 | 0.083 | 0.543 | 0.464 | 0.456 | 0.713 | 0.212 | 0.697 | 0.626 | 0.604 |
| semantic | dense | qdrant | hyde | 20 | 150 | 0.867 | 0.152 | 0.843 | 0.701 | 0.696 | 0.953 | 0.372 | 0.933 | 0.827 | 0.806 |
| semantic | dense | qdrant | multi | 20 | 150 | 0.847 | 0.150 | 0.833 | 0.655 | 0.654 | 0.913 | 0.354 | 0.907 | 0.781 | 0.765 |
| semantic | dense | qdrant | original | 20 | 150 | 0.847 | 0.151 | 0.827 | 0.647 | 0.650 | 0.920 | 0.352 | 0.907 | 0.768 | 0.760 |
| semantic | dense | qdrant | rewritten | 20 | 150 | 0.840 | 0.141 | 0.817 | 0.633 | 0.640 | 0.920 | 0.336 | 0.907 | 0.776 | 0.757 |
| semantic | hybrid | faiss_flatip+bm25 | multi | 20 | 150 | 0.893 | 0.154 | 0.887 | 0.693 | 0.685 | 0.953 | 0.343 | 0.953 | 0.832 | 0.801 |
| semantic | hybrid | faiss_flatip+bm25 | original | 20 | 150 | 0.913 | 0.167 | 0.903 | 0.723 | 0.708 | 0.973 | 0.363 | 0.970 | 0.835 | 0.817 |
| semantic | hybrid | faiss_flatip+bm25 | rewritten | 20 | 150 | 0.873 | 0.153 | 0.857 | 0.687 | 0.680 | 0.940 | 0.332 | 0.933 | 0.818 | 0.790 |
| semantic | hybrid | faiss_hnsw+bm25 | multi | 20 | 150 | 0.887 | 0.153 | 0.877 | 0.684 | 0.679 | 0.953 | 0.342 | 0.950 | 0.829 | 0.800 |
| semantic | hybrid | faiss_hnsw+bm25 | original | 20 | 150 | 0.907 | 0.165 | 0.893 | 0.687 | 0.683 | 0.967 | 0.364 | 0.963 | 0.801 | 0.797 |
| semantic | hybrid | faiss_hnsw+bm25 | rewritten | 20 | 150 | 0.873 | 0.152 | 0.857 | 0.675 | 0.672 | 0.940 | 0.333 | 0.933 | 0.801 | 0.782 |
| semantic | hybrid | faiss_ivf+bm25 | multi | 20 | 150 | 0.840 | 0.137 | 0.823 | 0.599 | 0.614 | 0.920 | 0.300 | 0.913 | 0.726 | 0.730 |
| semantic | hybrid | faiss_ivf+bm25 | original | 20 | 150 | 0.860 | 0.148 | 0.843 | 0.588 | 0.616 | 0.927 | 0.317 | 0.923 | 0.723 | 0.730 |
| semantic | hybrid | faiss_ivf+bm25 | rewritten | 20 | 150 | 0.827 | 0.136 | 0.800 | 0.561 | 0.585 | 0.900 | 0.290 | 0.890 | 0.716 | 0.709 |
| semantic | hybrid | qdrant+bm25 | multi | 20 | 150 | 0.893 | 0.154 | 0.887 | 0.693 | 0.684 | 0.953 | 0.343 | 0.953 | 0.831 | 0.801 |
| semantic | hybrid | qdrant+bm25 | original | 20 | 150 | 0.913 | 0.166 | 0.903 | 0.717 | 0.705 | 0.973 | 0.363 | 0.970 | 0.828 | 0.814 |
| semantic | hybrid | qdrant+bm25 | rewritten | 20 | 150 | 0.873 | 0.153 | 0.857 | 0.687 | 0.680 | 0.940 | 0.332 | 0.933 | 0.818 | 0.790 |
| sentence | bm25 | bm25 | multi | 20 | 150 | 0.807 | 0.113 | 0.770 | 0.624 | 0.625 | 0.880 | 0.259 | 0.870 | 0.742 | 0.738 |
| sentence | bm25 | bm25 | original | 20 | 150 | 0.853 | 0.130 | 0.820 | 0.702 | 0.686 | 0.907 | 0.308 | 0.900 | 0.812 | 0.790 |
| sentence | bm25 | bm25 | rewritten | 20 | 150 | 0.787 | 0.116 | 0.757 | 0.649 | 0.633 | 0.880 | 0.252 | 0.870 | 0.768 | 0.746 |
| sentence | dense | faiss_flatip | hyde | 20 | 150 | 0.860 | 0.145 | 0.840 | 0.656 | 0.660 | 0.933 | 0.370 | 0.917 | 0.800 | 0.791 |
| sentence | dense | faiss_flatip | multi | 20 | 150 | 0.873 | 0.139 | 0.850 | 0.620 | 0.631 | 0.927 | 0.350 | 0.913 | 0.765 | 0.761 |
| sentence | dense | faiss_flatip | original | 20 | 150 | 0.853 | 0.135 | 0.837 | 0.630 | 0.646 | 0.900 | 0.343 | 0.887 | 0.752 | 0.757 |
| sentence | dense | faiss_flatip | rewritten | 20 | 150 | 0.860 | 0.134 | 0.837 | 0.604 | 0.629 | 0.933 | 0.338 | 0.923 | 0.764 | 0.762 |
| sentence | dense | faiss_hnsw | hyde | 20 | 150 | 0.807 | 0.139 | 0.783 | 0.624 | 0.629 | 0.893 | 0.365 | 0.877 | 0.771 | 0.766 |
| sentence | dense | faiss_hnsw | multi | 20 | 150 | 0.840 | 0.137 | 0.813 | 0.614 | 0.622 | 0.900 | 0.351 | 0.883 | 0.760 | 0.752 |
| sentence | dense | faiss_hnsw | original | 20 | 150 | 0.813 | 0.133 | 0.803 | 0.608 | 0.622 | 0.867 | 0.343 | 0.857 | 0.727 | 0.735 |
| sentence | dense | faiss_hnsw | rewritten | 20 | 150 | 0.780 | 0.129 | 0.757 | 0.569 | 0.586 | 0.867 | 0.331 | 0.853 | 0.724 | 0.719 |
| sentence | dense | faiss_ivf | hyde | 20 | 150 | 0.620 | 0.101 | 0.593 | 0.489 | 0.490 | 0.720 | 0.259 | 0.703 | 0.638 | 0.624 |
| sentence | dense | faiss_ivf | multi | 20 | 150 | 0.640 | 0.102 | 0.617 | 0.488 | 0.493 | 0.753 | 0.262 | 0.743 | 0.617 | 0.618 |
| sentence | dense | faiss_ivf | original | 20 | 150 | 0.627 | 0.098 | 0.607 | 0.463 | 0.475 | 0.720 | 0.269 | 0.703 | 0.586 | 0.598 |
| sentence | dense | faiss_ivf | rewritten | 20 | 150 | 0.573 | 0.092 | 0.553 | 0.445 | 0.445 | 0.693 | 0.249 | 0.677 | 0.580 | 0.574 |
| sentence | dense | qdrant | hyde | 20 | 150 | 0.860 | 0.145 | 0.840 | 0.656 | 0.660 | 0.933 | 0.370 | 0.917 | 0.800 | 0.791 |
| sentence | dense | qdrant | multi | 20 | 150 | 0.873 | 0.139 | 0.850 | 0.620 | 0.631 | 0.927 | 0.351 | 0.913 | 0.765 | 0.761 |
| sentence | dense | qdrant | original | 20 | 150 | 0.847 | 0.135 | 0.830 | 0.624 | 0.640 | 0.893 | 0.343 | 0.880 | 0.745 | 0.751 |
| sentence | dense | qdrant | rewritten | 20 | 150 | 0.860 | 0.134 | 0.837 | 0.604 | 0.629 | 0.933 | 0.337 | 0.923 | 0.764 | 0.763 |
| sentence | hybrid | faiss_flatip+bm25 | multi | 20 | 150 | 0.907 | 0.146 | 0.900 | 0.681 | 0.677 | 0.953 | 0.340 | 0.953 | 0.831 | 0.806 |
| sentence | hybrid | faiss_flatip+bm25 | original | 20 | 150 | 0.940 | 0.154 | 0.927 | 0.718 | 0.712 | 0.987 | 0.363 | 0.983 | 0.828 | 0.823 |
| sentence | hybrid | faiss_flatip+bm25 | rewritten | 20 | 150 | 0.907 | 0.146 | 0.887 | 0.681 | 0.679 | 0.967 | 0.335 | 0.957 | 0.820 | 0.801 |
| sentence | hybrid | faiss_hnsw+bm25 | multi | 20 | 150 | 0.900 | 0.145 | 0.887 | 0.658 | 0.663 | 0.947 | 0.340 | 0.947 | 0.814 | 0.795 |
| sentence | hybrid | faiss_hnsw+bm25 | original | 20 | 150 | 0.940 | 0.153 | 0.923 | 0.705 | 0.708 | 0.980 | 0.360 | 0.973 | 0.812 | 0.816 |
| sentence | hybrid | faiss_hnsw+bm25 | rewritten | 20 | 150 | 0.893 | 0.145 | 0.873 | 0.656 | 0.660 | 0.960 | 0.334 | 0.950 | 0.793 | 0.784 |
| sentence | hybrid | faiss_ivf+bm25 | multi | 20 | 150 | 0.860 | 0.133 | 0.837 | 0.589 | 0.596 | 0.940 | 0.308 | 0.930 | 0.701 | 0.726 |
| sentence | hybrid | faiss_ivf+bm25 | original | 20 | 150 | 0.900 | 0.147 | 0.873 | 0.584 | 0.621 | 0.953 | 0.341 | 0.947 | 0.711 | 0.750 |
| sentence | hybrid | faiss_ivf+bm25 | rewritten | 20 | 150 | 0.853 | 0.132 | 0.823 | 0.544 | 0.585 | 0.927 | 0.302 | 0.913 | 0.695 | 0.713 |
| sentence | hybrid | qdrant+bm25 | multi | 20 | 150 | 0.907 | 0.146 | 0.900 | 0.681 | 0.677 | 0.953 | 0.340 | 0.953 | 0.831 | 0.806 |
| sentence | hybrid | qdrant+bm25 | original | 20 | 150 | 0.940 | 0.154 | 0.927 | 0.716 | 0.710 | 0.987 | 0.363 | 0.983 | 0.827 | 0.822 |
| sentence | hybrid | qdrant+bm25 | rewritten | 20 | 150 | 0.907 | 0.146 | 0.887 | 0.681 | 0.679 | 0.967 | 0.335 | 0.957 | 0.820 | 0.801 |

</details>

<details>
<summary>All 93 reranked rows (k=5)</summary>

| Chunk | Retrieval | Index | Query | k | n | c_hit | c_prec | c_rec | c_mrr | c_ndcg | d_hit | d_prec | d_rec | d_mrr | d_ndcg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed | bm25 | bm25 | multi | 5 | 150 | 0.787 | 0.304 | 0.753 | 0.725 | 0.727 | 0.873 | 0.551 | 0.853 | 0.847 | 0.840 |
| fixed | bm25 | bm25 | original | 5 | 150 | 0.820 | 0.316 | 0.787 | 0.747 | 0.747 | 0.893 | 0.564 | 0.873 | 0.853 | 0.849 |
| fixed | bm25 | bm25 | rewritten | 5 | 150 | 0.753 | 0.295 | 0.723 | 0.698 | 0.698 | 0.860 | 0.536 | 0.843 | 0.826 | 0.824 |
| fixed | dense | faiss_flatip | hyde | 5 | 150 | 0.833 | 0.333 | 0.793 | 0.745 | 0.749 | 0.927 | 0.592 | 0.907 | 0.861 | 0.868 |
| fixed | dense | faiss_flatip | multi | 5 | 150 | 0.827 | 0.327 | 0.787 | 0.738 | 0.741 | 0.900 | 0.579 | 0.880 | 0.853 | 0.853 |
| fixed | dense | faiss_flatip | original | 5 | 150 | 0.793 | 0.317 | 0.773 | 0.722 | 0.720 | 0.893 | 0.567 | 0.883 | 0.836 | 0.840 |
| fixed | dense | faiss_flatip | rewritten | 5 | 150 | 0.807 | 0.324 | 0.773 | 0.732 | 0.731 | 0.900 | 0.579 | 0.883 | 0.848 | 0.851 |
| fixed | dense | faiss_hnsw | hyde | 5 | 150 | 0.827 | 0.328 | 0.787 | 0.741 | 0.746 | 0.920 | 0.592 | 0.900 | 0.857 | 0.864 |
| fixed | dense | faiss_hnsw | multi | 5 | 150 | 0.813 | 0.323 | 0.767 | 0.716 | 0.725 | 0.893 | 0.577 | 0.873 | 0.834 | 0.837 |
| fixed | dense | faiss_hnsw | original | 5 | 150 | 0.767 | 0.311 | 0.743 | 0.701 | 0.700 | 0.873 | 0.563 | 0.857 | 0.818 | 0.823 |
| fixed | dense | faiss_hnsw | rewritten | 5 | 150 | 0.760 | 0.309 | 0.730 | 0.692 | 0.689 | 0.840 | 0.552 | 0.830 | 0.790 | 0.794 |
| fixed | dense | faiss_ivf | hyde | 5 | 150 | 0.553 | 0.217 | 0.523 | 0.502 | 0.506 | 0.700 | 0.417 | 0.687 | 0.656 | 0.660 |
| fixed | dense | faiss_ivf | multi | 5 | 150 | 0.587 | 0.232 | 0.557 | 0.538 | 0.539 | 0.727 | 0.444 | 0.713 | 0.686 | 0.685 |
| fixed | dense | faiss_ivf | original | 5 | 150 | 0.520 | 0.211 | 0.493 | 0.472 | 0.473 | 0.660 | 0.424 | 0.643 | 0.615 | 0.618 |
| fixed | dense | faiss_ivf | rewritten | 5 | 150 | 0.500 | 0.199 | 0.480 | 0.450 | 0.452 | 0.640 | 0.387 | 0.627 | 0.586 | 0.591 |
| fixed | dense | qdrant | hyde | 5 | 150 | 0.833 | 0.333 | 0.793 | 0.745 | 0.749 | 0.927 | 0.592 | 0.907 | 0.861 | 0.869 |
| fixed | dense | qdrant | multi | 5 | 150 | 0.827 | 0.327 | 0.787 | 0.738 | 0.741 | 0.900 | 0.577 | 0.880 | 0.853 | 0.852 |
| fixed | dense | qdrant | original | 5 | 150 | 0.793 | 0.315 | 0.770 | 0.718 | 0.718 | 0.893 | 0.565 | 0.880 | 0.832 | 0.837 |
| fixed | dense | qdrant | rewritten | 5 | 150 | 0.807 | 0.320 | 0.770 | 0.725 | 0.727 | 0.900 | 0.573 | 0.883 | 0.848 | 0.851 |
| fixed | hybrid | faiss_flatip+bm25 | multi | 5 | 150 | 0.840 | 0.337 | 0.813 | 0.757 | 0.757 | 0.927 | 0.599 | 0.910 | 0.868 | 0.869 |
| fixed | hybrid | faiss_flatip+bm25 | original | 5 | 150 | 0.840 | 0.348 | 0.817 | 0.771 | 0.767 | 0.927 | 0.611 | 0.907 | 0.881 | 0.877 |
| fixed | hybrid | faiss_flatip+bm25 | rewritten | 5 | 150 | 0.833 | 0.344 | 0.803 | 0.754 | 0.754 | 0.927 | 0.608 | 0.907 | 0.880 | 0.883 |
| fixed | hybrid | faiss_hnsw+bm25 | multi | 5 | 150 | 0.840 | 0.336 | 0.813 | 0.758 | 0.758 | 0.927 | 0.600 | 0.910 | 0.869 | 0.869 |
| fixed | hybrid | faiss_hnsw+bm25 | original | 5 | 150 | 0.840 | 0.344 | 0.817 | 0.771 | 0.768 | 0.927 | 0.605 | 0.907 | 0.881 | 0.878 |
| fixed | hybrid | faiss_hnsw+bm25 | rewritten | 5 | 150 | 0.820 | 0.340 | 0.790 | 0.746 | 0.746 | 0.920 | 0.601 | 0.900 | 0.875 | 0.877 |
| fixed | hybrid | faiss_ivf+bm25 | multi | 5 | 150 | 0.813 | 0.321 | 0.780 | 0.748 | 0.746 | 0.900 | 0.577 | 0.877 | 0.859 | 0.854 |
| fixed | hybrid | faiss_ivf+bm25 | original | 5 | 150 | 0.820 | 0.329 | 0.783 | 0.739 | 0.742 | 0.900 | 0.581 | 0.883 | 0.849 | 0.849 |
| fixed | hybrid | faiss_ivf+bm25 | rewritten | 5 | 150 | 0.800 | 0.319 | 0.767 | 0.740 | 0.734 | 0.887 | 0.573 | 0.867 | 0.854 | 0.847 |
| fixed | hybrid | qdrant+bm25 | multi | 5 | 150 | 0.840 | 0.337 | 0.813 | 0.757 | 0.757 | 0.927 | 0.599 | 0.910 | 0.868 | 0.869 |
| fixed | hybrid | qdrant+bm25 | original | 5 | 150 | 0.840 | 0.347 | 0.817 | 0.771 | 0.766 | 0.927 | 0.611 | 0.907 | 0.881 | 0.876 |
| fixed | hybrid | qdrant+bm25 | rewritten | 5 | 150 | 0.833 | 0.343 | 0.803 | 0.754 | 0.754 | 0.927 | 0.607 | 0.907 | 0.880 | 0.883 |
| semantic | bm25 | bm25 | multi | 5 | 150 | 0.767 | 0.309 | 0.733 | 0.709 | 0.710 | 0.860 | 0.504 | 0.843 | 0.821 | 0.824 |
| semantic | bm25 | bm25 | original | 5 | 150 | 0.793 | 0.343 | 0.763 | 0.722 | 0.723 | 0.873 | 0.557 | 0.863 | 0.817 | 0.823 |
| semantic | bm25 | bm25 | rewritten | 5 | 150 | 0.767 | 0.320 | 0.730 | 0.714 | 0.714 | 0.860 | 0.507 | 0.850 | 0.817 | 0.824 |
| semantic | dense | faiss_flatip | hyde | 5 | 150 | 0.833 | 0.348 | 0.800 | 0.760 | 0.762 | 0.927 | 0.593 | 0.910 | 0.879 | 0.881 |
| semantic | dense | faiss_flatip | multi | 5 | 150 | 0.820 | 0.352 | 0.783 | 0.730 | 0.741 | 0.893 | 0.571 | 0.877 | 0.845 | 0.848 |
| semantic | dense | faiss_flatip | original | 5 | 150 | 0.807 | 0.337 | 0.773 | 0.726 | 0.727 | 0.893 | 0.559 | 0.880 | 0.838 | 0.842 |
| semantic | dense | faiss_flatip | rewritten | 5 | 150 | 0.813 | 0.337 | 0.777 | 0.747 | 0.746 | 0.907 | 0.569 | 0.883 | 0.865 | 0.864 |
| semantic | dense | faiss_hnsw | hyde | 5 | 150 | 0.800 | 0.341 | 0.767 | 0.719 | 0.724 | 0.893 | 0.583 | 0.877 | 0.841 | 0.843 |
| semantic | dense | faiss_hnsw | multi | 5 | 150 | 0.787 | 0.345 | 0.750 | 0.711 | 0.717 | 0.880 | 0.565 | 0.863 | 0.831 | 0.836 |
| semantic | dense | faiss_hnsw | original | 5 | 150 | 0.747 | 0.319 | 0.713 | 0.675 | 0.674 | 0.840 | 0.539 | 0.820 | 0.791 | 0.793 |
| semantic | dense | faiss_hnsw | rewritten | 5 | 150 | 0.773 | 0.329 | 0.740 | 0.707 | 0.705 | 0.867 | 0.563 | 0.847 | 0.822 | 0.823 |
| semantic | dense | faiss_ivf | hyde | 5 | 150 | 0.587 | 0.231 | 0.537 | 0.513 | 0.523 | 0.707 | 0.427 | 0.690 | 0.663 | 0.667 |
| semantic | dense | faiss_ivf | multi | 5 | 150 | 0.640 | 0.253 | 0.597 | 0.582 | 0.587 | 0.733 | 0.439 | 0.713 | 0.693 | 0.695 |
| semantic | dense | faiss_ivf | original | 5 | 150 | 0.547 | 0.212 | 0.507 | 0.488 | 0.497 | 0.673 | 0.391 | 0.653 | 0.632 | 0.637 |
| semantic | dense | faiss_ivf | rewritten | 5 | 150 | 0.540 | 0.211 | 0.503 | 0.483 | 0.492 | 0.693 | 0.391 | 0.677 | 0.649 | 0.657 |
| semantic | dense | qdrant | hyde | 5 | 150 | 0.833 | 0.348 | 0.800 | 0.760 | 0.762 | 0.927 | 0.593 | 0.910 | 0.879 | 0.881 |
| semantic | dense | qdrant | multi | 5 | 150 | 0.820 | 0.352 | 0.783 | 0.730 | 0.741 | 0.893 | 0.571 | 0.877 | 0.845 | 0.848 |
| semantic | dense | qdrant | original | 5 | 150 | 0.800 | 0.336 | 0.767 | 0.723 | 0.722 | 0.887 | 0.556 | 0.870 | 0.831 | 0.835 |
| semantic | dense | qdrant | rewritten | 5 | 150 | 0.813 | 0.337 | 0.777 | 0.747 | 0.746 | 0.907 | 0.569 | 0.883 | 0.865 | 0.864 |
| semantic | hybrid | faiss_flatip+bm25 | multi | 5 | 150 | 0.840 | 0.375 | 0.817 | 0.778 | 0.773 | 0.913 | 0.589 | 0.903 | 0.876 | 0.875 |
| semantic | hybrid | faiss_flatip+bm25 | original | 5 | 150 | 0.840 | 0.379 | 0.820 | 0.772 | 0.768 | 0.920 | 0.596 | 0.913 | 0.875 | 0.879 |
| semantic | hybrid | faiss_flatip+bm25 | rewritten | 5 | 150 | 0.853 | 0.383 | 0.827 | 0.778 | 0.780 | 0.920 | 0.604 | 0.907 | 0.877 | 0.882 |
| semantic | hybrid | faiss_hnsw+bm25 | multi | 5 | 150 | 0.833 | 0.371 | 0.810 | 0.770 | 0.765 | 0.907 | 0.587 | 0.897 | 0.869 | 0.868 |
| semantic | hybrid | faiss_hnsw+bm25 | original | 5 | 150 | 0.833 | 0.375 | 0.807 | 0.765 | 0.759 | 0.920 | 0.593 | 0.910 | 0.869 | 0.873 |
| semantic | hybrid | faiss_hnsw+bm25 | rewritten | 5 | 150 | 0.853 | 0.383 | 0.827 | 0.779 | 0.779 | 0.920 | 0.603 | 0.907 | 0.877 | 0.881 |
| semantic | hybrid | faiss_ivf+bm25 | multi | 5 | 150 | 0.793 | 0.343 | 0.767 | 0.734 | 0.728 | 0.880 | 0.556 | 0.870 | 0.843 | 0.840 |
| semantic | hybrid | faiss_ivf+bm25 | original | 5 | 150 | 0.800 | 0.357 | 0.777 | 0.729 | 0.726 | 0.880 | 0.573 | 0.870 | 0.835 | 0.838 |
| semantic | hybrid | faiss_ivf+bm25 | rewritten | 5 | 150 | 0.813 | 0.359 | 0.777 | 0.750 | 0.749 | 0.887 | 0.564 | 0.873 | 0.840 | 0.847 |
| semantic | hybrid | qdrant+bm25 | multi | 5 | 150 | 0.840 | 0.375 | 0.817 | 0.778 | 0.773 | 0.913 | 0.589 | 0.903 | 0.876 | 0.875 |
| semantic | hybrid | qdrant+bm25 | original | 5 | 150 | 0.840 | 0.377 | 0.820 | 0.772 | 0.768 | 0.920 | 0.596 | 0.913 | 0.875 | 0.879 |
| semantic | hybrid | qdrant+bm25 | rewritten | 5 | 150 | 0.853 | 0.383 | 0.827 | 0.778 | 0.780 | 0.920 | 0.604 | 0.907 | 0.877 | 0.882 |
| sentence | bm25 | bm25 | multi | 5 | 150 | 0.760 | 0.293 | 0.720 | 0.680 | 0.686 | 0.860 | 0.509 | 0.843 | 0.818 | 0.822 |
| sentence | bm25 | bm25 | original | 5 | 150 | 0.807 | 0.320 | 0.777 | 0.727 | 0.729 | 0.893 | 0.560 | 0.883 | 0.841 | 0.846 |
| sentence | bm25 | bm25 | rewritten | 5 | 150 | 0.760 | 0.311 | 0.730 | 0.692 | 0.692 | 0.853 | 0.519 | 0.847 | 0.811 | 0.813 |
| sentence | dense | faiss_flatip | hyde | 5 | 150 | 0.807 | 0.340 | 0.777 | 0.734 | 0.740 | 0.907 | 0.589 | 0.890 | 0.856 | 0.861 |
| sentence | dense | faiss_flatip | multi | 5 | 150 | 0.807 | 0.340 | 0.777 | 0.737 | 0.737 | 0.893 | 0.571 | 0.877 | 0.841 | 0.847 |
| sentence | dense | faiss_flatip | original | 5 | 150 | 0.787 | 0.325 | 0.760 | 0.702 | 0.706 | 0.867 | 0.560 | 0.857 | 0.813 | 0.819 |
| sentence | dense | faiss_flatip | rewritten | 5 | 150 | 0.827 | 0.341 | 0.790 | 0.755 | 0.752 | 0.920 | 0.584 | 0.903 | 0.869 | 0.871 |
| sentence | dense | faiss_hnsw | hyde | 5 | 150 | 0.760 | 0.321 | 0.730 | 0.688 | 0.693 | 0.867 | 0.575 | 0.850 | 0.823 | 0.826 |
| sentence | dense | faiss_hnsw | multi | 5 | 150 | 0.773 | 0.328 | 0.740 | 0.700 | 0.701 | 0.860 | 0.560 | 0.840 | 0.804 | 0.811 |
| sentence | dense | faiss_hnsw | original | 5 | 150 | 0.753 | 0.315 | 0.730 | 0.673 | 0.676 | 0.840 | 0.545 | 0.830 | 0.783 | 0.791 |
| sentence | dense | faiss_hnsw | rewritten | 5 | 150 | 0.740 | 0.313 | 0.713 | 0.667 | 0.667 | 0.840 | 0.540 | 0.830 | 0.787 | 0.788 |
| sentence | dense | faiss_ivf | hyde | 5 | 150 | 0.587 | 0.248 | 0.550 | 0.526 | 0.531 | 0.667 | 0.428 | 0.650 | 0.644 | 0.641 |
| sentence | dense | faiss_ivf | multi | 5 | 150 | 0.587 | 0.248 | 0.560 | 0.548 | 0.544 | 0.693 | 0.439 | 0.687 | 0.658 | 0.655 |
| sentence | dense | faiss_ivf | original | 5 | 150 | 0.587 | 0.239 | 0.557 | 0.525 | 0.528 | 0.673 | 0.439 | 0.657 | 0.634 | 0.634 |
| sentence | dense | faiss_ivf | rewritten | 5 | 150 | 0.527 | 0.220 | 0.503 | 0.498 | 0.496 | 0.653 | 0.405 | 0.643 | 0.631 | 0.628 |
| sentence | dense | qdrant | hyde | 5 | 150 | 0.807 | 0.341 | 0.777 | 0.734 | 0.740 | 0.907 | 0.589 | 0.890 | 0.856 | 0.861 |
| sentence | dense | qdrant | multi | 5 | 150 | 0.807 | 0.340 | 0.777 | 0.737 | 0.737 | 0.893 | 0.572 | 0.877 | 0.841 | 0.847 |
| sentence | dense | qdrant | original | 5 | 150 | 0.780 | 0.327 | 0.753 | 0.699 | 0.702 | 0.860 | 0.560 | 0.850 | 0.807 | 0.812 |
| sentence | dense | qdrant | rewritten | 5 | 150 | 0.827 | 0.341 | 0.790 | 0.755 | 0.752 | 0.920 | 0.583 | 0.903 | 0.869 | 0.871 |
| sentence | hybrid | faiss_flatip+bm25 | multi | 5 | 150 | 0.833 | 0.356 | 0.813 | 0.760 | 0.758 | 0.920 | 0.589 | 0.910 | 0.869 | 0.869 |
| sentence | hybrid | faiss_flatip+bm25 | original | 5 | 150 | 0.853 | 0.360 | 0.837 | 0.768 | 0.768 | 0.947 | 0.611 | 0.943 | 0.885 | 0.889 |
| sentence | hybrid | faiss_flatip+bm25 | rewritten | 5 | 150 | 0.853 | 0.365 | 0.823 | 0.784 | 0.785 | 0.940 | 0.607 | 0.930 | 0.907 | 0.906 |
| sentence | hybrid | faiss_hnsw+bm25 | multi | 5 | 150 | 0.813 | 0.352 | 0.793 | 0.741 | 0.738 | 0.900 | 0.585 | 0.893 | 0.856 | 0.851 |
| sentence | hybrid | faiss_hnsw+bm25 | original | 5 | 150 | 0.847 | 0.355 | 0.827 | 0.763 | 0.763 | 0.933 | 0.599 | 0.923 | 0.876 | 0.880 |
| sentence | hybrid | faiss_hnsw+bm25 | rewritten | 5 | 150 | 0.840 | 0.357 | 0.810 | 0.767 | 0.770 | 0.927 | 0.599 | 0.920 | 0.889 | 0.891 |
| sentence | hybrid | faiss_ivf+bm25 | multi | 5 | 150 | 0.813 | 0.335 | 0.777 | 0.736 | 0.735 | 0.907 | 0.564 | 0.893 | 0.865 | 0.863 |
| sentence | hybrid | faiss_ivf+bm25 | original | 5 | 150 | 0.827 | 0.355 | 0.800 | 0.741 | 0.741 | 0.907 | 0.593 | 0.893 | 0.852 | 0.858 |
| sentence | hybrid | faiss_ivf+bm25 | rewritten | 5 | 150 | 0.813 | 0.333 | 0.773 | 0.740 | 0.742 | 0.900 | 0.564 | 0.890 | 0.864 | 0.863 |
| sentence | hybrid | qdrant+bm25 | multi | 5 | 150 | 0.833 | 0.356 | 0.813 | 0.760 | 0.758 | 0.920 | 0.589 | 0.910 | 0.869 | 0.869 |
| sentence | hybrid | qdrant+bm25 | original | 5 | 150 | 0.853 | 0.360 | 0.837 | 0.768 | 0.768 | 0.947 | 0.611 | 0.943 | 0.885 | 0.889 |
| sentence | hybrid | qdrant+bm25 | rewritten | 5 | 150 | 0.853 | 0.365 | 0.823 | 0.784 | 0.785 | 0.940 | 0.607 | 0.930 | 0.907 | 0.906 |

</details>

**Filtered retrieval quality.** The table above has 186 setups, 27,900 `(Retrieved, RetrievalQuality)` pairs: 4,650 plain and 4,650 reranked per chunk type. Each `Retrieved` that goes on to the next layers (context, prompt, generation, answer quality) becomes one LLM call, and generation and answer evaluation may use a paid API. So only the best few setups go on.

After the summary, `eval.py` calls `retrieval_evaluation.get_filtered_retrieval_qualities(n_retrieval_quality, retrieval_qualities)` and prints the summary again for what it keeps. The method:
1. **Scores each setup.** It groups the pairs by setup (chunk type, retrieval type, index, query type, reranking) and scores each setup by its mean chunk-level recall, with mean chunk-level MRR breaking ties. Chunk level is used because generation needs the passage that holds the answer, not just the right document. Recall comes first because a missed passage cannot be answered from. MRR breaks ties because the right chunk near the top helps the LLM.
2. **Keeps the best of each kind.** It keeps the best `--n_retrieval_quality` plain setups and, separately, the best `--n_retrieval_quality` reranked setups. They are ranked apart because plain setups are scored at k=20 and reranked ones at k=5, so plain would nearly always win a joint ranking only because it has more chunks.
3. **Skips duplicates.** It drops any `Retrieved` with the same question and the same chunks, in the same order, as one already kept, because it would give the LLM the same prompt and cost a call for nothing. This only happens when n ≥ 2 picks setups that return the same chunks, such as `faiss_flatip` and `qdrant`.

`--n_retrieval_quality` is 1 (`N_RETRIEVAL_QUALITY`), so 1 plain and 1 reranked setup are kept: 2 × 150 = 300 records instead of 27,900. A plain record has 20 chunks and a reranked one 5, so none of them can be duplicates of each other. The filter only regroups and sorts the qualities, so it takes well under a second and is not cached; it is recomputed from the cached retrieved outputs on every run.

On the 5000-doc built-in run, loaded from the cache in 38.47s (`log/l6_retrieval/log_5k_retrieval_quality_rerank_cached.txt`), the filter kept 300 records with 0 duplicates skipped:

| Chunk | Retrieval | Index | Query | Rerank | k | n | c_hit | c_prec | c_rec | c_mrr | c_ndcg | d_hit | d_prec | d_rec | d_mrr | d_ndcg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sentence | hybrid | faiss_flatip+bm25 | original | no | 20 | 150 | 0.940 | 0.154 | 0.927 | 0.718 | 0.712 | 0.987 | 0.363 | 0.983 | 0.828 | 0.823 |
| sentence | hybrid | qdrant+bm25 | original | yes | 5 | 150 | 0.853 | 0.360 | 0.837 | 0.768 | 0.768 | 0.947 | 0.611 | 0.943 | 0.885 | 0.889 |

- **Plain:** faiss_flatip+bm25 and qdrant+bm25 have the same recall (0.927), and faiss_flatip+bm25 wins on MRR (0.718 vs 0.716).
- **Reranked:** faiss_flatip+bm25 and qdrant+bm25 have exactly the same recall and MRR. When scores tie, the setup that sorts last alphabetically is kept, so qdrant+bm25 was chosen. The two are effectively the same setup, since both are exact search over the same vectors.
- **Both kept setups** use sentence chunks, hybrid retrieval and the original question.

**LangChain retrieval quality, 5000 documents, 150 tuning queries** (from `log/l6_retrieval/log_5k_langchain_retrieval_quality_rerank.txt`). Same setups as built-in, searched on the LangChain chunks and indexes with the same l5 queries. The best setup per chunk type (ranked like the filter, by chunk-level recall then MRR):

| Chunk | Rerank | Built-in best | c_rec / c_mrr / c_hit | LangChain best | c_rec / c_mrr / c_hit |
|---|---|---|---|---|---|
| fixed | no | hybrid qdrant+bm25 original | 0.900 / 0.729 / 0.913 | hybrid faiss_flatip+bm25 original | 0.900 / 0.725 / 0.913 |
| fixed | yes | hybrid qdrant+bm25 original | 0.817 / 0.771 / 0.840 | hybrid qdrant+bm25 original | 0.817 / 0.772 / 0.840 |
| sentence | no | hybrid faiss_flatip+bm25 original | 0.927 / 0.718 / 0.940 | hybrid qdrant+bm25 original | 0.927 / 0.721 / 0.940 |
| sentence | yes | hybrid qdrant+bm25 original | 0.837 / 0.768 / 0.853 | hybrid qdrant+bm25 original | 0.837 / 0.768 / 0.853 |
| semantic | no | hybrid faiss_flatip+bm25 original | 0.903 / 0.723 / 0.913 | hybrid faiss_flatip+bm25 original | 0.903 / 0.719 / 0.920 |
| semantic | yes | hybrid faiss_hnsw+bm25 rewritten | 0.827 / 0.779 / 0.853 | hybrid faiss_hnsw+bm25 original | 0.827 / 0.770 / 0.853 |

- **LangChain and built-in score the same.** Averaged over each chunk type's 62 rows, LangChain's chunk hit differs from built-in's by less than 0.3 points. The best recall per chunk type is identical.
- **Fixed and sentence rows differ by at most 2.0 and 2.7 points** of chunk hit. Both frameworks have the same number of chunks, so the small differences come from slightly different chunk boundaries and text.
- **Semantic rows differ by up to 6.7 points,** but 0.05 points on average. LangChain's semantic chunker makes 7318 more chunks, so individual setups move up or down, but neither framework is better overall.
- **The filtered 300 are the same setups** as built-in: sentence, hybrid, original question. Plain qdrant+bm25 (c_rec 0.927, MRR 0.721, a tie on recall that LangChain's qdrant wins on MRR) and reranked qdrant+bm25 (c_rec 0.837, MRR 0.768). 0 duplicates were skipped.

<details>
<summary>All 93 LangChain plain rows (k=20)</summary>

| Chunk | Retrieval | Index | Query | k | n | c_hit | c_prec | c_rec | c_mrr | c_ndcg | d_hit | d_prec | d_rec | d_mrr | d_ndcg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed | bm25 | bm25 | multi | 20 | 150 | 0.820 | 0.108 | 0.797 | 0.612 | 0.624 | 0.880 | 0.280 | 0.870 | 0.745 | 0.738 |
| fixed | bm25 | bm25 | original | 20 | 150 | 0.867 | 0.119 | 0.843 | 0.699 | 0.703 | 0.913 | 0.321 | 0.903 | 0.811 | 0.799 |
| fixed | bm25 | bm25 | rewritten | 20 | 150 | 0.780 | 0.103 | 0.757 | 0.596 | 0.614 | 0.887 | 0.265 | 0.877 | 0.743 | 0.734 |
| fixed | dense | faiss_flatip | hyde | 20 | 150 | 0.873 | 0.129 | 0.847 | 0.670 | 0.685 | 0.947 | 0.378 | 0.930 | 0.827 | 0.817 |
| fixed | dense | faiss_flatip | multi | 20 | 150 | 0.860 | 0.125 | 0.833 | 0.663 | 0.665 | 0.933 | 0.354 | 0.920 | 0.789 | 0.778 |
| fixed | dense | faiss_flatip | original | 20 | 150 | 0.840 | 0.124 | 0.813 | 0.650 | 0.661 | 0.920 | 0.357 | 0.903 | 0.784 | 0.778 |
| fixed | dense | faiss_flatip | rewritten | 20 | 150 | 0.820 | 0.122 | 0.800 | 0.657 | 0.644 | 0.913 | 0.343 | 0.900 | 0.780 | 0.763 |
| fixed | dense | faiss_hnsw | hyde | 20 | 150 | 0.853 | 0.125 | 0.820 | 0.663 | 0.676 | 0.927 | 0.377 | 0.907 | 0.819 | 0.810 |
| fixed | dense | faiss_hnsw | multi | 20 | 150 | 0.840 | 0.125 | 0.807 | 0.650 | 0.648 | 0.913 | 0.359 | 0.897 | 0.779 | 0.763 |
| fixed | dense | faiss_hnsw | original | 20 | 150 | 0.787 | 0.121 | 0.767 | 0.613 | 0.621 | 0.867 | 0.350 | 0.847 | 0.735 | 0.735 |
| fixed | dense | faiss_hnsw | rewritten | 20 | 150 | 0.800 | 0.119 | 0.773 | 0.640 | 0.631 | 0.873 | 0.339 | 0.860 | 0.754 | 0.740 |
| fixed | dense | faiss_ivf | hyde | 20 | 150 | 0.613 | 0.084 | 0.587 | 0.492 | 0.497 | 0.733 | 0.253 | 0.720 | 0.653 | 0.645 |
| fixed | dense | faiss_ivf | multi | 20 | 150 | 0.600 | 0.089 | 0.577 | 0.477 | 0.475 | 0.747 | 0.271 | 0.733 | 0.637 | 0.621 |
| fixed | dense | faiss_ivf | original | 20 | 150 | 0.553 | 0.082 | 0.530 | 0.442 | 0.445 | 0.680 | 0.262 | 0.667 | 0.596 | 0.586 |
| fixed | dense | faiss_ivf | rewritten | 20 | 150 | 0.513 | 0.073 | 0.497 | 0.421 | 0.415 | 0.640 | 0.227 | 0.630 | 0.566 | 0.550 |
| fixed | dense | qdrant | hyde | 20 | 150 | 0.873 | 0.129 | 0.847 | 0.670 | 0.685 | 0.947 | 0.378 | 0.930 | 0.827 | 0.817 |
| fixed | dense | qdrant | multi | 20 | 150 | 0.860 | 0.125 | 0.833 | 0.663 | 0.665 | 0.933 | 0.354 | 0.920 | 0.789 | 0.779 |
| fixed | dense | qdrant | original | 20 | 150 | 0.847 | 0.124 | 0.817 | 0.648 | 0.661 | 0.920 | 0.356 | 0.903 | 0.784 | 0.777 |
| fixed | dense | qdrant | rewritten | 20 | 150 | 0.827 | 0.122 | 0.807 | 0.657 | 0.646 | 0.913 | 0.343 | 0.900 | 0.785 | 0.765 |
| fixed | hybrid | faiss_flatip+bm25 | multi | 20 | 150 | 0.880 | 0.131 | 0.863 | 0.706 | 0.691 | 0.947 | 0.352 | 0.943 | 0.816 | 0.800 |
| fixed | hybrid | faiss_flatip+bm25 | original | 20 | 150 | 0.913 | 0.138 | 0.900 | 0.725 | 0.719 | 0.967 | 0.370 | 0.963 | 0.856 | 0.834 |
| fixed | hybrid | faiss_flatip+bm25 | rewritten | 20 | 150 | 0.880 | 0.131 | 0.863 | 0.686 | 0.684 | 0.960 | 0.339 | 0.950 | 0.806 | 0.806 |
| fixed | hybrid | faiss_hnsw+bm25 | multi | 20 | 150 | 0.880 | 0.131 | 0.863 | 0.693 | 0.681 | 0.947 | 0.353 | 0.943 | 0.804 | 0.792 |
| fixed | hybrid | faiss_hnsw+bm25 | original | 20 | 150 | 0.913 | 0.136 | 0.897 | 0.695 | 0.698 | 0.967 | 0.371 | 0.963 | 0.825 | 0.817 |
| fixed | hybrid | faiss_hnsw+bm25 | rewritten | 20 | 150 | 0.880 | 0.131 | 0.863 | 0.674 | 0.673 | 0.960 | 0.338 | 0.950 | 0.788 | 0.793 |
| fixed | hybrid | faiss_ivf+bm25 | multi | 20 | 150 | 0.833 | 0.122 | 0.813 | 0.556 | 0.590 | 0.900 | 0.318 | 0.887 | 0.707 | 0.721 |
| fixed | hybrid | faiss_ivf+bm25 | original | 20 | 150 | 0.867 | 0.127 | 0.837 | 0.579 | 0.614 | 0.933 | 0.341 | 0.923 | 0.735 | 0.743 |
| fixed | hybrid | faiss_ivf+bm25 | rewritten | 20 | 150 | 0.827 | 0.116 | 0.800 | 0.509 | 0.561 | 0.913 | 0.296 | 0.900 | 0.675 | 0.690 |
| fixed | hybrid | qdrant+bm25 | multi | 20 | 150 | 0.880 | 0.131 | 0.863 | 0.707 | 0.691 | 0.947 | 0.352 | 0.943 | 0.817 | 0.801 |
| fixed | hybrid | qdrant+bm25 | original | 20 | 150 | 0.913 | 0.138 | 0.900 | 0.722 | 0.718 | 0.967 | 0.370 | 0.963 | 0.856 | 0.834 |
| fixed | hybrid | qdrant+bm25 | rewritten | 20 | 150 | 0.880 | 0.131 | 0.863 | 0.692 | 0.688 | 0.960 | 0.339 | 0.950 | 0.812 | 0.809 |
| semantic | bm25 | bm25 | multi | 20 | 150 | 0.820 | 0.110 | 0.790 | 0.611 | 0.625 | 0.880 | 0.275 | 0.870 | 0.739 | 0.725 |
| semantic | bm25 | bm25 | original | 20 | 150 | 0.867 | 0.123 | 0.840 | 0.684 | 0.691 | 0.920 | 0.313 | 0.913 | 0.803 | 0.799 |
| semantic | bm25 | bm25 | rewritten | 20 | 150 | 0.773 | 0.103 | 0.750 | 0.605 | 0.616 | 0.853 | 0.261 | 0.843 | 0.749 | 0.728 |
| semantic | dense | faiss_flatip | hyde | 20 | 150 | 0.853 | 0.134 | 0.833 | 0.641 | 0.658 | 0.927 | 0.376 | 0.910 | 0.802 | 0.790 |
| semantic | dense | faiss_flatip | multi | 20 | 150 | 0.873 | 0.133 | 0.840 | 0.669 | 0.676 | 0.927 | 0.354 | 0.913 | 0.804 | 0.789 |
| semantic | dense | faiss_flatip | original | 20 | 150 | 0.833 | 0.134 | 0.803 | 0.651 | 0.658 | 0.927 | 0.352 | 0.907 | 0.788 | 0.781 |
| semantic | dense | faiss_flatip | rewritten | 20 | 150 | 0.840 | 0.128 | 0.807 | 0.631 | 0.641 | 0.913 | 0.345 | 0.900 | 0.775 | 0.760 |
| semantic | dense | faiss_hnsw | hyde | 20 | 150 | 0.813 | 0.132 | 0.790 | 0.619 | 0.634 | 0.907 | 0.374 | 0.890 | 0.785 | 0.773 |
| semantic | dense | faiss_hnsw | multi | 20 | 150 | 0.867 | 0.135 | 0.833 | 0.656 | 0.660 | 0.920 | 0.357 | 0.907 | 0.791 | 0.775 |
| semantic | dense | faiss_hnsw | original | 20 | 150 | 0.793 | 0.133 | 0.763 | 0.617 | 0.624 | 0.873 | 0.352 | 0.850 | 0.750 | 0.741 |
| semantic | dense | faiss_hnsw | rewritten | 20 | 150 | 0.813 | 0.127 | 0.780 | 0.612 | 0.620 | 0.887 | 0.345 | 0.870 | 0.755 | 0.741 |
| semantic | dense | faiss_ivf | hyde | 20 | 150 | 0.627 | 0.092 | 0.583 | 0.496 | 0.503 | 0.740 | 0.256 | 0.720 | 0.651 | 0.641 |
| semantic | dense | faiss_ivf | multi | 20 | 150 | 0.613 | 0.096 | 0.577 | 0.473 | 0.486 | 0.747 | 0.254 | 0.730 | 0.631 | 0.632 |
| semantic | dense | faiss_ivf | original | 20 | 150 | 0.533 | 0.088 | 0.500 | 0.410 | 0.421 | 0.653 | 0.224 | 0.637 | 0.555 | 0.553 |
| semantic | dense | faiss_ivf | rewritten | 20 | 150 | 0.547 | 0.081 | 0.510 | 0.440 | 0.437 | 0.680 | 0.221 | 0.660 | 0.581 | 0.570 |
| semantic | dense | qdrant | hyde | 20 | 150 | 0.853 | 0.134 | 0.833 | 0.641 | 0.658 | 0.927 | 0.376 | 0.910 | 0.802 | 0.790 |
| semantic | dense | qdrant | multi | 20 | 150 | 0.873 | 0.134 | 0.840 | 0.669 | 0.676 | 0.927 | 0.354 | 0.913 | 0.804 | 0.789 |
| semantic | dense | qdrant | original | 20 | 150 | 0.827 | 0.134 | 0.797 | 0.647 | 0.655 | 0.920 | 0.352 | 0.900 | 0.785 | 0.777 |
| semantic | dense | qdrant | rewritten | 20 | 150 | 0.833 | 0.128 | 0.800 | 0.623 | 0.635 | 0.907 | 0.344 | 0.893 | 0.767 | 0.754 |
| semantic | hybrid | faiss_flatip+bm25 | multi | 20 | 150 | 0.887 | 0.143 | 0.870 | 0.695 | 0.689 | 0.940 | 0.347 | 0.940 | 0.826 | 0.798 |
| semantic | hybrid | faiss_flatip+bm25 | original | 20 | 150 | 0.920 | 0.149 | 0.903 | 0.719 | 0.717 | 0.980 | 0.375 | 0.977 | 0.837 | 0.828 |
| semantic | hybrid | faiss_flatip+bm25 | rewritten | 20 | 150 | 0.887 | 0.135 | 0.877 | 0.659 | 0.668 | 0.960 | 0.339 | 0.953 | 0.797 | 0.785 |
| semantic | hybrid | faiss_hnsw+bm25 | multi | 20 | 150 | 0.887 | 0.143 | 0.870 | 0.689 | 0.687 | 0.940 | 0.349 | 0.940 | 0.820 | 0.796 |
| semantic | hybrid | faiss_hnsw+bm25 | original | 20 | 150 | 0.920 | 0.148 | 0.900 | 0.695 | 0.701 | 0.973 | 0.375 | 0.970 | 0.811 | 0.810 |
| semantic | hybrid | faiss_hnsw+bm25 | rewritten | 20 | 150 | 0.880 | 0.135 | 0.867 | 0.640 | 0.656 | 0.947 | 0.339 | 0.937 | 0.778 | 0.776 |
| semantic | hybrid | faiss_ivf+bm25 | multi | 20 | 150 | 0.853 | 0.128 | 0.837 | 0.557 | 0.597 | 0.913 | 0.309 | 0.907 | 0.712 | 0.728 |
| semantic | hybrid | faiss_ivf+bm25 | original | 20 | 150 | 0.900 | 0.136 | 0.877 | 0.554 | 0.610 | 0.953 | 0.328 | 0.947 | 0.677 | 0.723 |
| semantic | hybrid | faiss_ivf+bm25 | rewritten | 20 | 150 | 0.853 | 0.117 | 0.827 | 0.537 | 0.578 | 0.927 | 0.289 | 0.913 | 0.686 | 0.702 |
| semantic | hybrid | qdrant+bm25 | multi | 20 | 150 | 0.887 | 0.143 | 0.870 | 0.695 | 0.689 | 0.940 | 0.347 | 0.940 | 0.826 | 0.798 |
| semantic | hybrid | qdrant+bm25 | original | 20 | 150 | 0.920 | 0.148 | 0.903 | 0.716 | 0.715 | 0.980 | 0.375 | 0.977 | 0.834 | 0.825 |
| semantic | hybrid | qdrant+bm25 | rewritten | 20 | 150 | 0.887 | 0.135 | 0.877 | 0.655 | 0.666 | 0.960 | 0.339 | 0.953 | 0.794 | 0.783 |
| sentence | bm25 | bm25 | multi | 20 | 150 | 0.807 | 0.113 | 0.770 | 0.624 | 0.625 | 0.880 | 0.259 | 0.870 | 0.742 | 0.738 |
| sentence | bm25 | bm25 | original | 20 | 150 | 0.853 | 0.130 | 0.820 | 0.702 | 0.686 | 0.907 | 0.308 | 0.900 | 0.812 | 0.790 |
| sentence | bm25 | bm25 | rewritten | 20 | 150 | 0.787 | 0.116 | 0.757 | 0.649 | 0.633 | 0.880 | 0.252 | 0.870 | 0.768 | 0.746 |
| sentence | dense | faiss_flatip | hyde | 20 | 150 | 0.860 | 0.145 | 0.840 | 0.656 | 0.660 | 0.933 | 0.370 | 0.917 | 0.800 | 0.791 |
| sentence | dense | faiss_flatip | multi | 20 | 150 | 0.873 | 0.139 | 0.850 | 0.620 | 0.631 | 0.927 | 0.350 | 0.913 | 0.765 | 0.761 |
| sentence | dense | faiss_flatip | original | 20 | 150 | 0.853 | 0.135 | 0.837 | 0.630 | 0.646 | 0.900 | 0.343 | 0.887 | 0.752 | 0.757 |
| sentence | dense | faiss_flatip | rewritten | 20 | 150 | 0.860 | 0.134 | 0.837 | 0.604 | 0.629 | 0.933 | 0.338 | 0.923 | 0.764 | 0.762 |
| sentence | dense | faiss_hnsw | hyde | 20 | 150 | 0.827 | 0.142 | 0.800 | 0.644 | 0.646 | 0.913 | 0.370 | 0.893 | 0.792 | 0.783 |
| sentence | dense | faiss_hnsw | multi | 20 | 150 | 0.847 | 0.137 | 0.823 | 0.616 | 0.625 | 0.913 | 0.352 | 0.897 | 0.762 | 0.755 |
| sentence | dense | faiss_hnsw | original | 20 | 150 | 0.813 | 0.131 | 0.790 | 0.604 | 0.618 | 0.860 | 0.342 | 0.837 | 0.721 | 0.727 |
| sentence | dense | faiss_hnsw | rewritten | 20 | 150 | 0.807 | 0.128 | 0.770 | 0.581 | 0.600 | 0.893 | 0.336 | 0.873 | 0.742 | 0.740 |
| sentence | dense | faiss_ivf | hyde | 20 | 150 | 0.627 | 0.102 | 0.600 | 0.491 | 0.496 | 0.747 | 0.270 | 0.730 | 0.658 | 0.644 |
| sentence | dense | faiss_ivf | multi | 20 | 150 | 0.647 | 0.107 | 0.627 | 0.499 | 0.500 | 0.773 | 0.273 | 0.763 | 0.627 | 0.629 |
| sentence | dense | faiss_ivf | original | 20 | 150 | 0.607 | 0.099 | 0.587 | 0.461 | 0.469 | 0.713 | 0.264 | 0.697 | 0.581 | 0.591 |
| sentence | dense | faiss_ivf | rewritten | 20 | 150 | 0.573 | 0.096 | 0.550 | 0.451 | 0.449 | 0.707 | 0.255 | 0.690 | 0.590 | 0.585 |
| sentence | dense | qdrant | hyde | 20 | 150 | 0.860 | 0.145 | 0.840 | 0.656 | 0.660 | 0.933 | 0.370 | 0.917 | 0.800 | 0.791 |
| sentence | dense | qdrant | multi | 20 | 150 | 0.873 | 0.139 | 0.850 | 0.620 | 0.632 | 0.927 | 0.350 | 0.913 | 0.765 | 0.761 |
| sentence | dense | qdrant | original | 20 | 150 | 0.853 | 0.135 | 0.837 | 0.630 | 0.646 | 0.900 | 0.343 | 0.887 | 0.752 | 0.758 |
| sentence | dense | qdrant | rewritten | 20 | 150 | 0.853 | 0.134 | 0.827 | 0.604 | 0.627 | 0.927 | 0.337 | 0.917 | 0.763 | 0.760 |
| sentence | hybrid | faiss_flatip+bm25 | multi | 20 | 150 | 0.907 | 0.146 | 0.900 | 0.681 | 0.677 | 0.953 | 0.340 | 0.953 | 0.831 | 0.807 |
| sentence | hybrid | faiss_flatip+bm25 | original | 20 | 150 | 0.940 | 0.154 | 0.927 | 0.718 | 0.712 | 0.987 | 0.363 | 0.983 | 0.828 | 0.823 |
| sentence | hybrid | faiss_flatip+bm25 | rewritten | 20 | 150 | 0.907 | 0.146 | 0.887 | 0.681 | 0.679 | 0.967 | 0.335 | 0.957 | 0.820 | 0.802 |
| sentence | hybrid | faiss_hnsw+bm25 | multi | 20 | 150 | 0.907 | 0.145 | 0.893 | 0.657 | 0.660 | 0.953 | 0.340 | 0.953 | 0.813 | 0.795 |
| sentence | hybrid | faiss_hnsw+bm25 | original | 20 | 150 | 0.940 | 0.152 | 0.927 | 0.702 | 0.707 | 0.980 | 0.360 | 0.977 | 0.808 | 0.816 |
| sentence | hybrid | faiss_hnsw+bm25 | rewritten | 20 | 150 | 0.893 | 0.144 | 0.870 | 0.666 | 0.666 | 0.960 | 0.335 | 0.947 | 0.804 | 0.791 |
| sentence | hybrid | faiss_ivf+bm25 | multi | 20 | 150 | 0.867 | 0.137 | 0.840 | 0.598 | 0.604 | 0.940 | 0.314 | 0.930 | 0.714 | 0.731 |
| sentence | hybrid | faiss_ivf+bm25 | original | 20 | 150 | 0.900 | 0.147 | 0.873 | 0.585 | 0.624 | 0.953 | 0.339 | 0.947 | 0.708 | 0.749 |
| sentence | hybrid | faiss_ivf+bm25 | rewritten | 20 | 150 | 0.853 | 0.134 | 0.827 | 0.548 | 0.588 | 0.927 | 0.306 | 0.913 | 0.695 | 0.712 |
| sentence | hybrid | qdrant+bm25 | multi | 20 | 150 | 0.907 | 0.146 | 0.900 | 0.681 | 0.677 | 0.953 | 0.340 | 0.953 | 0.831 | 0.807 |
| sentence | hybrid | qdrant+bm25 | original | 20 | 150 | 0.940 | 0.154 | 0.927 | 0.721 | 0.713 | 0.987 | 0.363 | 0.983 | 0.831 | 0.824 |
| sentence | hybrid | qdrant+bm25 | rewritten | 20 | 150 | 0.907 | 0.146 | 0.883 | 0.681 | 0.680 | 0.967 | 0.335 | 0.957 | 0.820 | 0.802 |

</details>

<details>
<summary>All 93 LangChain reranked rows (k=5)</summary>

| Chunk | Retrieval | Index | Query | k | n | c_hit | c_prec | c_rec | c_mrr | c_ndcg | d_hit | d_prec | d_rec | d_mrr | d_ndcg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed | bm25 | bm25 | multi | 5 | 150 | 0.787 | 0.304 | 0.753 | 0.725 | 0.727 | 0.873 | 0.551 | 0.853 | 0.846 | 0.839 |
| fixed | bm25 | bm25 | original | 5 | 150 | 0.813 | 0.315 | 0.780 | 0.746 | 0.744 | 0.887 | 0.561 | 0.867 | 0.852 | 0.847 |
| fixed | bm25 | bm25 | rewritten | 5 | 150 | 0.753 | 0.296 | 0.723 | 0.698 | 0.698 | 0.860 | 0.536 | 0.843 | 0.826 | 0.824 |
| fixed | dense | faiss_flatip | hyde | 5 | 150 | 0.833 | 0.333 | 0.793 | 0.745 | 0.749 | 0.927 | 0.593 | 0.907 | 0.861 | 0.868 |
| fixed | dense | faiss_flatip | multi | 5 | 150 | 0.827 | 0.327 | 0.787 | 0.738 | 0.741 | 0.900 | 0.577 | 0.880 | 0.853 | 0.853 |
| fixed | dense | faiss_flatip | original | 5 | 150 | 0.793 | 0.317 | 0.773 | 0.722 | 0.720 | 0.893 | 0.567 | 0.883 | 0.836 | 0.840 |
| fixed | dense | faiss_flatip | rewritten | 5 | 150 | 0.807 | 0.324 | 0.773 | 0.732 | 0.731 | 0.900 | 0.579 | 0.883 | 0.848 | 0.851 |
| fixed | dense | faiss_hnsw | hyde | 5 | 150 | 0.813 | 0.323 | 0.770 | 0.725 | 0.731 | 0.907 | 0.588 | 0.883 | 0.839 | 0.847 |
| fixed | dense | faiss_hnsw | multi | 5 | 150 | 0.800 | 0.323 | 0.760 | 0.712 | 0.716 | 0.880 | 0.579 | 0.860 | 0.826 | 0.828 |
| fixed | dense | faiss_hnsw | original | 5 | 150 | 0.747 | 0.304 | 0.727 | 0.686 | 0.684 | 0.847 | 0.553 | 0.833 | 0.797 | 0.801 |
| fixed | dense | faiss_hnsw | rewritten | 5 | 150 | 0.780 | 0.315 | 0.747 | 0.700 | 0.701 | 0.853 | 0.557 | 0.840 | 0.799 | 0.806 |
| fixed | dense | faiss_ivf | hyde | 5 | 150 | 0.567 | 0.219 | 0.533 | 0.519 | 0.521 | 0.713 | 0.420 | 0.700 | 0.667 | 0.668 |
| fixed | dense | faiss_ivf | multi | 5 | 150 | 0.580 | 0.229 | 0.553 | 0.531 | 0.530 | 0.713 | 0.449 | 0.703 | 0.677 | 0.673 |
| fixed | dense | faiss_ivf | original | 5 | 150 | 0.533 | 0.211 | 0.507 | 0.482 | 0.485 | 0.667 | 0.432 | 0.653 | 0.620 | 0.623 |
| fixed | dense | faiss_ivf | rewritten | 5 | 150 | 0.493 | 0.196 | 0.473 | 0.445 | 0.445 | 0.620 | 0.392 | 0.610 | 0.583 | 0.580 |
| fixed | dense | qdrant | hyde | 5 | 150 | 0.833 | 0.333 | 0.793 | 0.745 | 0.749 | 0.927 | 0.593 | 0.907 | 0.861 | 0.868 |
| fixed | dense | qdrant | multi | 5 | 150 | 0.827 | 0.327 | 0.787 | 0.738 | 0.741 | 0.900 | 0.576 | 0.880 | 0.853 | 0.853 |
| fixed | dense | qdrant | original | 5 | 150 | 0.800 | 0.317 | 0.777 | 0.725 | 0.726 | 0.900 | 0.567 | 0.887 | 0.843 | 0.847 |
| fixed | dense | qdrant | rewritten | 5 | 150 | 0.807 | 0.323 | 0.773 | 0.732 | 0.731 | 0.900 | 0.576 | 0.883 | 0.848 | 0.851 |
| fixed | hybrid | faiss_flatip+bm25 | multi | 5 | 150 | 0.840 | 0.337 | 0.813 | 0.757 | 0.757 | 0.927 | 0.599 | 0.910 | 0.868 | 0.868 |
| fixed | hybrid | faiss_flatip+bm25 | original | 5 | 150 | 0.840 | 0.348 | 0.817 | 0.772 | 0.768 | 0.927 | 0.611 | 0.907 | 0.881 | 0.876 |
| fixed | hybrid | faiss_flatip+bm25 | rewritten | 5 | 150 | 0.833 | 0.344 | 0.803 | 0.754 | 0.754 | 0.927 | 0.607 | 0.907 | 0.880 | 0.883 |
| fixed | hybrid | faiss_hnsw+bm25 | multi | 5 | 150 | 0.840 | 0.337 | 0.813 | 0.756 | 0.756 | 0.927 | 0.601 | 0.910 | 0.867 | 0.868 |
| fixed | hybrid | faiss_hnsw+bm25 | original | 5 | 150 | 0.840 | 0.345 | 0.817 | 0.771 | 0.766 | 0.927 | 0.609 | 0.907 | 0.881 | 0.877 |
| fixed | hybrid | faiss_hnsw+bm25 | rewritten | 5 | 150 | 0.840 | 0.341 | 0.810 | 0.756 | 0.758 | 0.933 | 0.601 | 0.913 | 0.882 | 0.885 |
| fixed | hybrid | faiss_ivf+bm25 | multi | 5 | 150 | 0.807 | 0.323 | 0.773 | 0.742 | 0.739 | 0.893 | 0.572 | 0.870 | 0.852 | 0.846 |
| fixed | hybrid | faiss_ivf+bm25 | original | 5 | 150 | 0.807 | 0.327 | 0.777 | 0.733 | 0.734 | 0.893 | 0.581 | 0.880 | 0.846 | 0.843 |
| fixed | hybrid | faiss_ivf+bm25 | rewritten | 5 | 150 | 0.793 | 0.319 | 0.757 | 0.730 | 0.725 | 0.893 | 0.573 | 0.873 | 0.856 | 0.851 |
| fixed | hybrid | qdrant+bm25 | multi | 5 | 150 | 0.840 | 0.337 | 0.813 | 0.757 | 0.757 | 0.927 | 0.600 | 0.910 | 0.868 | 0.868 |
| fixed | hybrid | qdrant+bm25 | original | 5 | 150 | 0.840 | 0.348 | 0.817 | 0.772 | 0.768 | 0.927 | 0.611 | 0.907 | 0.881 | 0.877 |
| fixed | hybrid | qdrant+bm25 | rewritten | 5 | 150 | 0.827 | 0.343 | 0.797 | 0.753 | 0.752 | 0.927 | 0.607 | 0.907 | 0.880 | 0.883 |
| semantic | bm25 | bm25 | multi | 5 | 150 | 0.793 | 0.300 | 0.753 | 0.721 | 0.730 | 0.867 | 0.536 | 0.847 | 0.838 | 0.834 |
| semantic | bm25 | bm25 | original | 5 | 150 | 0.827 | 0.321 | 0.793 | 0.728 | 0.745 | 0.893 | 0.545 | 0.887 | 0.840 | 0.847 |
| semantic | bm25 | bm25 | rewritten | 5 | 150 | 0.753 | 0.285 | 0.723 | 0.678 | 0.687 | 0.853 | 0.520 | 0.837 | 0.804 | 0.809 |
| semantic | dense | faiss_flatip | hyde | 5 | 150 | 0.813 | 0.327 | 0.767 | 0.724 | 0.734 | 0.913 | 0.596 | 0.890 | 0.867 | 0.862 |
| semantic | dense | faiss_flatip | multi | 5 | 150 | 0.827 | 0.323 | 0.777 | 0.737 | 0.751 | 0.907 | 0.577 | 0.887 | 0.871 | 0.864 |
| semantic | dense | faiss_flatip | original | 5 | 150 | 0.793 | 0.317 | 0.753 | 0.704 | 0.716 | 0.900 | 0.571 | 0.880 | 0.850 | 0.852 |
| semantic | dense | faiss_flatip | rewritten | 5 | 150 | 0.807 | 0.320 | 0.760 | 0.747 | 0.750 | 0.907 | 0.569 | 0.883 | 0.877 | 0.869 |
| semantic | dense | faiss_hnsw | hyde | 5 | 150 | 0.773 | 0.316 | 0.723 | 0.689 | 0.699 | 0.887 | 0.587 | 0.863 | 0.832 | 0.835 |
| semantic | dense | faiss_hnsw | multi | 5 | 150 | 0.813 | 0.320 | 0.767 | 0.725 | 0.738 | 0.887 | 0.575 | 0.870 | 0.855 | 0.847 |
| semantic | dense | faiss_hnsw | original | 5 | 150 | 0.747 | 0.307 | 0.707 | 0.667 | 0.679 | 0.853 | 0.556 | 0.830 | 0.805 | 0.810 |
| semantic | dense | faiss_hnsw | rewritten | 5 | 150 | 0.773 | 0.312 | 0.723 | 0.710 | 0.715 | 0.873 | 0.555 | 0.847 | 0.848 | 0.838 |
| semantic | dense | faiss_ivf | hyde | 5 | 150 | 0.607 | 0.237 | 0.553 | 0.527 | 0.540 | 0.720 | 0.431 | 0.700 | 0.675 | 0.675 |
| semantic | dense | faiss_ivf | multi | 5 | 150 | 0.573 | 0.233 | 0.530 | 0.511 | 0.520 | 0.733 | 0.429 | 0.717 | 0.664 | 0.670 |
| semantic | dense | faiss_ivf | original | 5 | 150 | 0.513 | 0.205 | 0.473 | 0.429 | 0.444 | 0.620 | 0.383 | 0.607 | 0.557 | 0.565 |
| semantic | dense | faiss_ivf | rewritten | 5 | 150 | 0.513 | 0.207 | 0.470 | 0.459 | 0.466 | 0.640 | 0.383 | 0.623 | 0.611 | 0.608 |
| semantic | dense | qdrant | hyde | 5 | 150 | 0.813 | 0.327 | 0.767 | 0.724 | 0.734 | 0.913 | 0.596 | 0.890 | 0.867 | 0.862 |
| semantic | dense | qdrant | multi | 5 | 150 | 0.827 | 0.323 | 0.777 | 0.737 | 0.751 | 0.907 | 0.577 | 0.887 | 0.871 | 0.865 |
| semantic | dense | qdrant | original | 5 | 150 | 0.787 | 0.316 | 0.747 | 0.704 | 0.715 | 0.893 | 0.569 | 0.873 | 0.850 | 0.850 |
| semantic | dense | qdrant | rewritten | 5 | 150 | 0.800 | 0.317 | 0.753 | 0.740 | 0.743 | 0.900 | 0.567 | 0.877 | 0.870 | 0.861 |
| semantic | hybrid | faiss_flatip+bm25 | multi | 5 | 150 | 0.833 | 0.348 | 0.800 | 0.745 | 0.753 | 0.907 | 0.603 | 0.890 | 0.871 | 0.863 |
| semantic | hybrid | faiss_flatip+bm25 | original | 5 | 150 | 0.847 | 0.351 | 0.817 | 0.761 | 0.768 | 0.927 | 0.615 | 0.913 | 0.884 | 0.881 |
| semantic | hybrid | faiss_flatip+bm25 | rewritten | 5 | 150 | 0.847 | 0.352 | 0.817 | 0.773 | 0.777 | 0.933 | 0.612 | 0.923 | 0.899 | 0.891 |
| semantic | hybrid | faiss_hnsw+bm25 | multi | 5 | 150 | 0.833 | 0.348 | 0.800 | 0.746 | 0.754 | 0.907 | 0.601 | 0.890 | 0.871 | 0.865 |
| semantic | hybrid | faiss_hnsw+bm25 | original | 5 | 150 | 0.853 | 0.355 | 0.827 | 0.770 | 0.776 | 0.920 | 0.616 | 0.910 | 0.882 | 0.878 |
| semantic | hybrid | faiss_hnsw+bm25 | rewritten | 5 | 150 | 0.847 | 0.352 | 0.817 | 0.768 | 0.773 | 0.927 | 0.609 | 0.917 | 0.895 | 0.886 |
| semantic | hybrid | faiss_ivf+bm25 | multi | 5 | 150 | 0.813 | 0.321 | 0.770 | 0.739 | 0.750 | 0.900 | 0.565 | 0.880 | 0.861 | 0.858 |
| semantic | hybrid | faiss_ivf+bm25 | original | 5 | 150 | 0.840 | 0.340 | 0.800 | 0.746 | 0.755 | 0.913 | 0.595 | 0.900 | 0.872 | 0.865 |
| semantic | hybrid | faiss_ivf+bm25 | rewritten | 5 | 150 | 0.827 | 0.325 | 0.780 | 0.747 | 0.756 | 0.913 | 0.575 | 0.893 | 0.876 | 0.872 |
| semantic | hybrid | qdrant+bm25 | multi | 5 | 150 | 0.833 | 0.347 | 0.800 | 0.745 | 0.753 | 0.907 | 0.603 | 0.890 | 0.871 | 0.863 |
| semantic | hybrid | qdrant+bm25 | original | 5 | 150 | 0.847 | 0.351 | 0.817 | 0.762 | 0.769 | 0.927 | 0.615 | 0.913 | 0.885 | 0.882 |
| semantic | hybrid | qdrant+bm25 | rewritten | 5 | 150 | 0.847 | 0.351 | 0.817 | 0.773 | 0.777 | 0.933 | 0.609 | 0.923 | 0.899 | 0.891 |
| sentence | bm25 | bm25 | multi | 5 | 150 | 0.760 | 0.293 | 0.720 | 0.680 | 0.686 | 0.860 | 0.509 | 0.843 | 0.818 | 0.822 |
| sentence | bm25 | bm25 | original | 5 | 150 | 0.807 | 0.320 | 0.777 | 0.727 | 0.729 | 0.893 | 0.560 | 0.883 | 0.841 | 0.846 |
| sentence | bm25 | bm25 | rewritten | 5 | 150 | 0.760 | 0.311 | 0.730 | 0.692 | 0.693 | 0.853 | 0.519 | 0.847 | 0.812 | 0.813 |
| sentence | dense | faiss_flatip | hyde | 5 | 150 | 0.807 | 0.340 | 0.777 | 0.734 | 0.740 | 0.907 | 0.589 | 0.890 | 0.856 | 0.861 |
| sentence | dense | faiss_flatip | multi | 5 | 150 | 0.807 | 0.340 | 0.777 | 0.737 | 0.737 | 0.893 | 0.571 | 0.877 | 0.841 | 0.847 |
| sentence | dense | faiss_flatip | original | 5 | 150 | 0.780 | 0.324 | 0.753 | 0.701 | 0.704 | 0.867 | 0.560 | 0.857 | 0.813 | 0.819 |
| sentence | dense | faiss_flatip | rewritten | 5 | 150 | 0.827 | 0.341 | 0.790 | 0.755 | 0.752 | 0.920 | 0.584 | 0.903 | 0.869 | 0.871 |
| sentence | dense | faiss_hnsw | hyde | 5 | 150 | 0.780 | 0.332 | 0.750 | 0.708 | 0.713 | 0.887 | 0.585 | 0.870 | 0.842 | 0.844 |
| sentence | dense | faiss_hnsw | multi | 5 | 150 | 0.787 | 0.328 | 0.750 | 0.712 | 0.714 | 0.887 | 0.565 | 0.867 | 0.827 | 0.835 |
| sentence | dense | faiss_hnsw | original | 5 | 150 | 0.740 | 0.309 | 0.707 | 0.667 | 0.670 | 0.833 | 0.539 | 0.813 | 0.782 | 0.788 |
| sentence | dense | faiss_hnsw | rewritten | 5 | 150 | 0.767 | 0.316 | 0.727 | 0.689 | 0.691 | 0.860 | 0.549 | 0.843 | 0.811 | 0.813 |
| sentence | dense | faiss_ivf | hyde | 5 | 150 | 0.600 | 0.249 | 0.563 | 0.539 | 0.542 | 0.687 | 0.436 | 0.670 | 0.654 | 0.654 |
| sentence | dense | faiss_ivf | multi | 5 | 150 | 0.607 | 0.267 | 0.583 | 0.557 | 0.555 | 0.727 | 0.456 | 0.720 | 0.684 | 0.683 |
| sentence | dense | faiss_ivf | original | 5 | 150 | 0.573 | 0.239 | 0.543 | 0.522 | 0.522 | 0.660 | 0.427 | 0.643 | 0.621 | 0.621 |
| sentence | dense | faiss_ivf | rewritten | 5 | 150 | 0.533 | 0.229 | 0.507 | 0.506 | 0.502 | 0.667 | 0.420 | 0.657 | 0.634 | 0.635 |
| sentence | dense | qdrant | hyde | 5 | 150 | 0.807 | 0.340 | 0.777 | 0.738 | 0.743 | 0.907 | 0.589 | 0.890 | 0.859 | 0.863 |
| sentence | dense | qdrant | multi | 5 | 150 | 0.807 | 0.340 | 0.777 | 0.737 | 0.737 | 0.893 | 0.571 | 0.877 | 0.841 | 0.847 |
| sentence | dense | qdrant | original | 5 | 150 | 0.780 | 0.325 | 0.753 | 0.704 | 0.705 | 0.867 | 0.561 | 0.857 | 0.813 | 0.819 |
| sentence | dense | qdrant | rewritten | 5 | 150 | 0.820 | 0.339 | 0.780 | 0.744 | 0.743 | 0.913 | 0.583 | 0.897 | 0.862 | 0.864 |
| sentence | hybrid | faiss_flatip+bm25 | multi | 5 | 150 | 0.833 | 0.356 | 0.813 | 0.760 | 0.758 | 0.920 | 0.589 | 0.910 | 0.869 | 0.869 |
| sentence | hybrid | faiss_flatip+bm25 | original | 5 | 150 | 0.853 | 0.360 | 0.837 | 0.768 | 0.768 | 0.947 | 0.611 | 0.943 | 0.885 | 0.889 |
| sentence | hybrid | faiss_flatip+bm25 | rewritten | 5 | 150 | 0.860 | 0.367 | 0.830 | 0.785 | 0.788 | 0.947 | 0.608 | 0.937 | 0.909 | 0.909 |
| sentence | hybrid | faiss_hnsw+bm25 | multi | 5 | 150 | 0.820 | 0.353 | 0.800 | 0.747 | 0.745 | 0.907 | 0.585 | 0.900 | 0.866 | 0.861 |
| sentence | hybrid | faiss_hnsw+bm25 | original | 5 | 150 | 0.847 | 0.353 | 0.830 | 0.760 | 0.761 | 0.940 | 0.600 | 0.933 | 0.875 | 0.881 |
| sentence | hybrid | faiss_hnsw+bm25 | rewritten | 5 | 150 | 0.847 | 0.360 | 0.813 | 0.768 | 0.770 | 0.933 | 0.600 | 0.923 | 0.890 | 0.892 |
| sentence | hybrid | faiss_ivf+bm25 | multi | 5 | 150 | 0.820 | 0.344 | 0.783 | 0.738 | 0.738 | 0.900 | 0.572 | 0.887 | 0.853 | 0.851 |
| sentence | hybrid | faiss_ivf+bm25 | original | 5 | 150 | 0.827 | 0.356 | 0.800 | 0.741 | 0.742 | 0.900 | 0.592 | 0.887 | 0.851 | 0.855 |
| sentence | hybrid | faiss_ivf+bm25 | rewritten | 5 | 150 | 0.820 | 0.335 | 0.780 | 0.743 | 0.746 | 0.907 | 0.569 | 0.897 | 0.865 | 0.864 |
| sentence | hybrid | qdrant+bm25 | multi | 5 | 150 | 0.833 | 0.356 | 0.813 | 0.760 | 0.758 | 0.920 | 0.589 | 0.910 | 0.869 | 0.869 |
| sentence | hybrid | qdrant+bm25 | original | 5 | 150 | 0.853 | 0.360 | 0.837 | 0.768 | 0.767 | 0.947 | 0.612 | 0.943 | 0.885 | 0.889 |
| sentence | hybrid | qdrant+bm25 | rewritten | 5 | 150 | 0.860 | 0.365 | 0.827 | 0.782 | 0.786 | 0.947 | 0.607 | 0.937 | 0.909 | 0.909 |

</details>

## Dependency notes

- **`langchain-community` is pinned to 0.4.1, not 0.4.2.** `ragas` 0.4.3 (the latest release, used for eval QnA generation in `src/qna/qna.py`) imports `langchain_community.chat_models.vertexai`, which was removed in `langchain-community` 0.4.2, so `import ragas` fails on 0.4.2. Older `ragas` releases (0.3.x) have the same import.
- **`langchain-experimental` is pinned to 0.4.1, not 0.4.2.** 0.4.2 requires `langchain-community>=0.4.2`. 0.4.1 accepts 0.4.1, and its `SemanticChunker` (used by LangChain semantic chunking in l2) differs from 0.4.2 only by a type cast, so chunking results are unchanged.
- **`langchain-core` moved from 1.6.7 to 1.6.9.** Installing `ragas` and `langchain-anthropic` upgraded it.
