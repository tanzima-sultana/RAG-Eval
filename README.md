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
- **Qdrant**, a collection on the Qdrant server at `QDRANT_URL` (`http://localhost:6333`, started with the `qdrant-docker` skill) with cosine distance, uploaded in batches of `--qdrant_batch_size`. Each point stores its `chunk_id` in its payload. The collection is named `{dataset_name}_{chunk_type}_{model_name}_{min}_{max}_{overlap}_{threshold}`.

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
