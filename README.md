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

## qna

### Overview

`qna` builds the eval question-answer set the RAG pipeline is scored against. `src/eval.py` calls `get_eval_qna(output_path, documents, eval_qna_args)` after loading the l1 documents (`doc_id` to `Document`), the l2 chunks and the l4 indexes. `EvalQnAArgs` holds `--eval_qna_llm` (the LLM provider, default `claude`; the models are picked in `llm`), `--eval_qna_model` (the local SentenceTransformer RAGAS uses to link related documents, default `all-MiniLM-L6-v2`), `--n_sample_documents` and `--n_sample_evals`. It returns a list of `EvalQnA`, each with the question, the reference answer, the source passages, the RAGAS question type, and the `doc_ids` and source paths of the documents the passages came from.

`get_eval_qna()` first validates `EvalQnAArgs`, before any LLM call. The cache file is `{output folder}/eval_qna_{n_sample_documents}_{n_sample_evals}_{eval_qna_llm}_{eval_qna_model}.json`. It does not include the dataset size, so one eval set made once is reused at every corpus size (5k, 50k, 100k) as long as its source documents are in the corpus. If that file exists it is loaded and returned, even if it holds fewer items than asked for. Otherwise `generate_eval_qna()`:

1. Keeps only documents with more than `EVAL_MIN_DOCUMENT_TOKENS` (500) tokens, counted with RAGAS's own counter (tiktoken `cl100k_base`), because RAGAS only extracts headlines and summaries from documents over 500 tokens. It raises an error if fewer documents than `n_sample_documents` are left.
2. Sorts those doc_ids and picks `n_sample_documents` of them with a fixed seed (`SAMPLE_DOCUMENTS_SEED` = 42), so the same documents are picked every run.
3. Gets the two Claude models with `llm.get_llm()` (`CLAUDE_TRANSFORMS_MODEL` and `CLAUDE_QUESTION_MODEL`), wraps each for RAGAS as `transforms_llm` and `question_llm`, and turns each picked document into a LangChain `Document`.
4. Builds the RAGAS knowledge graph with `transforms_llm` (Claude Haiku 5.5): headlines, splitting into chunks, summaries, a filter that drops chunks not useful for questions, themes, named entities, and the links between related chunks. The graph is saved to `{output folder}/eval_kg_{n_sample_documents}_{eval_qna_llm}_{eval_qna_model}.json` and loaded from there on later runs, so a failed or repeated run does not pay to build it again. Its name has no `n_sample_evals`, so more questions can later be generated from the same documents and only the question writing is paid.
5. Runs the RAGAS `TestsetGenerator` on the graph with `question_llm` (Claude Sonnet 5.5) to write the personas, scenarios, questions and answers. After generation it prints the calls, tokens and cost of each model and the total (see `llm`).
6. Raises an error if RAGAS generated nothing, so no empty file is saved. If it generated fewer than `n_sample_evals`, it prints a warning and keeps the ones generated.
7. Converts each RAGAS sample into an `EvalQnA`. A sample's `doc_ids` are the picked documents that contain the start of one of its passages: the first `EVAL_PASSAGE_MATCH_CHARS` (80) characters, after removing the `<N-hop>` marker RAGAS puts in front of each multi-hop passage and collapsing whitespace in both texts. Only the start is matched because RAGAS can change the rest of a passage (see the 5000-document run below).

The list is then saved to the cache file, written to a temporary file first and renamed into place.

### Architecture

- **`qna.py`** — Loads the cached eval QnA, or generates it with RAGAS and the LLM generator from a sample of documents and saves it.
- **`schema.py`** — Defines `EvalQnA`, the layer's output record.
- **`validate.py`** — Checks the eval QnA parameters and raises an error for the first invalid one, before any LLM call.

The LLMs come from the `llm` module (see below). The cache file path is built in `src/common/paths.py`.

### Run results

**5000 documents, 100 sample documents, 200 eval QnA: the eval set used for the project** (from `log/qna/qna_5k_100_200_log.txt`). This is the eval set the retrieval and generation layers are scored against. It was made from the same 5000 `pixparse/pdfa-eng-wds` documents the build phase used, through a one-off script that calls `get_eval_qna()` with `--eval_qna_llm claude --eval_qna_model all-MiniLM-L6-v2 --n_sample_documents 100 --n_sample_evals 200`. l1 loaded all 5000 documents from its cache in 8.529310777000319s.

| | |
|---|---|
| Documents with more than 500 tokens | 4028 of 5000 |
| Documents sampled for RAGAS | 100 of 4028 |
| Eval QnA generated | 200 of 200 (100 `single_hop_specific_query_synthesizer`, 100 `multi_hop_specific_query_synthesizer`) |
| Duplicate questions | 0 |
| Source documents the questions come from | 54 of the 100 sampled |
| Questions linked to 1 document / 2 documents | 190 / 10 (all 10 are multi-hop) |
| Question length in words, min / median / max | 6 / 28 / 144 |
| Saved to | `outputs/pdfa-eng-wds/eval_qna_100_200_claude_all-MiniLM-L6-v2.json` |

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

## llm

### Overview

`llm` gives the rest of the project the LLMs it needs without exposing which models are used. `get_llm(llm, llm_model)` takes a provider name (`claude` or `ollama`) and a model, checks the model can be used on that provider, and returns a `ClaudeLLM` or `OllamaLLM`. Both have `generate(prompt)`, which returns the reply text, and `compute_cost()`, which prints and returns the cost of its calls (always 0 for Ollama). `qna` passes `--eval_qna_llm` (default `claude`) with each of its two Claude models, wraps each `ClaudeLLM` for RAGAS, and adds up the two costs.

For `claude` the models are chosen in `src/common/constants.py`:

| Role | Model | Price per million tokens (input / output) |
|---|---|---|
| Knowledge graph steps (headlines, summaries, filter, themes, entities) | `claude-haiku-5-5` (`CLAUDE_TRANSFORMS_MODEL`) | $0.10 / $0.50 |
| Personas, scenarios, questions and answers | `claude-sonnet-5-5` (`CLAUDE_QUESTION_MODEL`) | $2.00 / $10.00 |

The knowledge graph steps make most of the calls and send whole documents and chunks, so they run on the cheaper model. Each model is a LangChain `ChatAnthropic` with at most 16000 output tokens and a 180s timeout per call, wrapped for RAGAS. Each has its own token counter, so `compute_cost()` prints the calls, tokens and cost per model and the total. The API key is read from `ANTHROPIC_API_KEY`.

### Architecture

- **`llm.py`** — `get_llm()`: checks the provider and model and returns its LLM. For `claude` the model must have a price in `CLAUDE_PRICES_PER_MILLION_TOKENS`; for `ollama` the server must be running and the model pulled. Raises an error otherwise.
- **`claude.py`** — `ClaudeLLM`: one Claude model with `generate()`, the token counter attached to each call, and `compute_cost()` from `CLAUDE_PRICES_PER_MILLION_TOKENS`.
- **`ollama.py`** — `OllamaLLM`: one local Ollama model with `generate()`, called over HTTP at temperature 0, and `compute_cost()`, which is always 0 because local calls are free.
- **`gemini.py`** — Empty placeholder for Gemini models.
- **`chatgpt.py`** — Empty placeholder for OpenAI models.

## l5_query

### Overview

`l5_query` turns each eval question into the queries the retrieval layer searches with. `src/eval.py` calls `get_queries(output_path, eval_qna, eval_qna_args, query_args)` after loading the eval QnA. `QueryArgs` holds the build's embedding model (`embedding_model`, taken from `--model_name` so it always matches the chunk embeddings), `--query_llm` (the LLM provider, default `ollama`), `--query_llm_model` (default `llama3.1:8b`) and `--n_multi_qus` (default 3). It returns one `Query` per eval question, with the same `id` as its `EvalQnA`, holding four ways of searching for it: the original question (`original_qus`), an LLM rewrite (`rewritten_qus`), `n_multi_qus` LLM phrasings (`multi_qus`) and a HyDE hypothetical answer passage (`hyde_ans`). Each question (`Qus`) has its text, its BM25 tokens (lowercased and split on whitespace, the same as the l4 BM25 chunks) and its embedding; the HyDE answer (`Ans`) has its text and embedding only, because it is searched by embedding alone.

`get_queries()` first validates `QueryArgs`. The cache folder is `{output folder}/queries/{n_sample_documents}_{n_sample_evals}_{eval_qna_llm}_{eval_qna_model}_{embedding_model}_{query_llm}_{query_llm_model}_{n_multi_qus}`, with `/` and `:` in model names replaced by `-`. It does not include the dataset size, so one set of queries is reused at every corpus size. If the folder holds both cache files they are loaded and returned. Otherwise `generate_queries()`:

1. Gets the query LLM with `llm.get_llm()`, which checks the Ollama server is running and the model is pulled. The LLM runs locally at temperature 0, with replies cut at `OLLAMA_MAX_TOKENS` (400) tokens, so the transformations cost nothing.
2. Makes three LLM calls per question: the rewrite, the `n_multi_qus` phrasings (one per line) and the HyDE passage. If the phrasings reply does not have exactly `n_multi_qus` non-empty lines, the question gets no phrasings; if the HyDE reply still starts with "I don't know", it is stored as exactly `I don't know`. Both are printed and counted, and the run goes on. It prints the LLM cost ($0 for Ollama).
3. Embeds every question and answer with `embedding_model` (a SentenceTransformer, normalized to length 1 like the l3 chunk embeddings).
4. Builds one `Query` per question.

The queries are then saved as `queries.json` (the texts and tokens, readable) and `embeddings.npz` (the float32 vectors), each written to a temporary file first and renamed into place.

**HyDE and hallucination.** The HyDE answer is written by the LLM without seeing the documents, and the prompt asks for a plausible passage even when the LLM does not know the answer, so it can hold invented facts. It is only used as a search probe: it is embedded and searched with, and is never shown or used as an answer. The risk is in retrieval, not in the answer: invented names, numbers or claims can pull in wrong chunks, which the retrieval evaluation measures. If the HyDE text is ever passed to a later layer (for example into the l8 prompt or l9 generation), it can bring those invented facts into the generated answer and cause hallucination, so later layers must use only the retrieved chunks, never `hyde_ans`.

### Architecture

- **`query.py`** — Loads the cached queries, or transforms each eval question with the query LLM, embeds the results and saves them.
- **`schema.py`** — Defines `Qus` (a searchable question), `Ans` (the HyDE answer) and `Query`, the layer's output record.
- **`validate.py`** — Checks the query parameters and raises an error for the first invalid one, before any LLM call.
- **`cache.py`** — Saves and loads the queries of one run as `queries.json` and `embeddings.npz`.
- **`prompt.py`** — The rewrite, multi-query and HyDE prompt templates.

The query LLM comes from the `llm` module. The cache folder path is built in `src/common/paths.py`.

### Run results

**200 eval questions, llama3.1:8b, 3 phrasings each** (from `log/l5_query/query_5k_100_200_log.txt`). The input is the 200-question eval set made from the 5000 `pixparse/pdfa-eng-wds` documents (see qna), loaded from its cache file `eval_qna_100_200_claude_all-MiniLM-L6-v2.json`. Settings: `embedding_model` `all-MiniLM-L6-v2`, `query_llm` `ollama`, `query_llm_model` `llama3.1:8b`, `n_multi_qus` 3. There was no query cache, so the queries were generated.

| | Value |
|---|---|
| Eval QnA loaded | 200 |
| Questions transformed | 200 of 200 |
| Multi-query I don't know | 0 of 200 |
| HyDE I don't know | 0 of 200 |
| LLM cost | $0.0000 |
| Queries saved | 200, to `outputs/pdfa-eng-wds/queries/100_200_claude_all-MiniLM-L6-v2_all-MiniLM-L6-v2_ollama_llama3.1-8b_3` |
| Query time | 763.0530879497528s |

The log has no per-stage timing, only the total. Nearly all of it is the LLM: each question makes three calls to `llama3.1:8b`, one after another, so 600 calls in total, each generating its reply token by token on the GPU. Embedding the results is a single batched SentenceTransformer pass per query type over short texts, and loading the eval set and saving the two cache files are file reads and writes, so they are small next to the 600 generations.

## l6_retrieval

### Overview

`l6_retrieval` finds the chunks for each eval question. `src/eval.py` splits the 200 queries with a fixed seed (`TUNING_SPLIT_SEED` 42) into `--tuning_queries_size` (150) tuning queries and 50 held-out queries, and only the tuning queries are retrieved. It creates one `Retrieval(retrieval_args)` before the loops. `RetrievalArgs` holds `--framework`, `--retrieval_types` (any of `dense`, `bm25`, `hybrid`), `--top_k` (20), `--reranking` (`true` or `false`) and `--rerank_top_k` (5). It then loops one item at a time:

1. For each chunk type in `--chunk_types`, it loads that chunk type's l2 chunks.
2. For each index type in `--index_types`, it loads that one l4 index with `index_cache.load_index()` (load-only, never builds) and runs:
   - **Dense** (`compute_dense_retrieval`) on `faiss_flatip`, `faiss_ivf`, `faiss_hnsw` and `qdrant`. It searches with each query type's embedding and keeps the top_k chunks. The score is cosine similarity. The multi-query phrasings are searched one by one and merged, keeping each chunk's best score. A HyDE answer stored as `I don't know` is not searched.
   - **BM25** (`compute_bm25_retrieval`) on `bm25`. It scores every chunk with the query tokens and keeps the top_k chunks with a score above 0. HyDE has no tokens, so BM25 has no HyDE record.
   - Each method returns `[]` when its type is not in `--retrieval_types` or the index is the wrong kind, so the loop needs no type checks.
3. After the index loop, **hybrid** (`compute_hybrid_retrieval`) fuses each dense index's outputs with the BM25 outputs of the same question and query type, using reciprocal rank fusion. A chunk gets `1 / (RRF_K + rank)` from each list it is in, with `RRF_K` 60, and the top_k by fused score are kept. Ranks are used, not scores, because dense and BM25 scores are on different scales. `index_type` is written as e.g. `faiss_flatip+bm25`.
4. With `--reranking true`, each plain output list is reranked separately (`get_reranked_dense_retrieval`, `get_reranked_bm25_retrieval`, `get_reranked_hybrid_retrieval`). The cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` (`CROSS_ENCODER_MODEL`, loaded once in `Retrieval.__init__`) scores each chunk's text against the original eval question, and the best `rerank_top_k` are kept. Every query type is reranked against the original question, because that is what the answer must match. The reranked records are a separate list from the plain ones, and hybrid always fuses the plain top_k lists.

Each output is a list of `Retrieved`, one per eval question and query type:
- **Setup fields:** `id` (the `EvalQnA` id), `query_type`, `chunk_type`, `retrieval_type`, `index_type`, `top_k`, `reranking`, `rerank_top_k`.
- **Result fields:** the ranked `chunk_ids`, their `chunk_scores` and `doc_ids`. The doc id is the part of the chunk id before the last `_`.

Records per index, for 150 tuning queries:

| Retrieval | Query types | Records |
|---|---|---|
| Dense | original, rewritten, multi, hyde | 600 |
| BM25 | original, rewritten, multi | 450 |
| Hybrid | original, rewritten, multi (no BM25 for HyDE) | 450 |

A query type with nothing to search (no multi-query phrasings, or a HyDE `I don't know`) still gets a `Retrieved` with no chunks, so it counts as a miss. Nothing is saved to disk.

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

Retrieval quality (hit, precision, recall, MRR, nDCG at chunk and doc level) is measured in `l10_evaluation`, not here.

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

## Dependency notes

- **`langchain-community` is pinned to 0.4.1, not 0.4.2.** `ragas` 0.4.3 (the latest release, used for eval QnA generation in `src/qna/qna.py`) imports `langchain_community.chat_models.vertexai`, which was removed in `langchain-community` 0.4.2, so `import ragas` fails on 0.4.2. Older `ragas` releases (0.3.x) have the same import.
- **`langchain-experimental` is pinned to 0.4.1, not 0.4.2.** 0.4.2 requires `langchain-community>=0.4.2`. 0.4.1 accepts 0.4.1, and its `SemanticChunker` (used by LangChain semantic chunking in l2) differs from 0.4.2 only by a type cast, so chunking results are unchanged.
- **`langchain-core` moved from 1.6.7 to 1.6.9.** Installing `ragas` and `langchain-anthropic` upgraded it.
