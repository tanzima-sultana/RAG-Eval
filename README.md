## l2_chunking

### Overview

`l2_chunking` is the second pipeline layer. `src/rag.py` runs it right after `l1_ingestion` and passes it the ingested documents in memory, as a dict of `doc_id` to `Document`, together with an output folder `{--output_path}/{dataset_name}`, where `dataset_name` is the last part of `--input_path` (for example `outputs/pdfa-eng-wds`). The chunking settings travel as a `ChunkingArgs`: `--framework`, `--model_name`, `--chunk_type` (`fixed`, `sentence` or `semantic`), `--min_chunk_size`, `--max_chunk_size`, `--fixed_chunk_overlap` and `--semantic_threshold`. `Chunking.compute_chunks()` first validates these settings, then splits each document into `Chunk` records (`id`, `text`, `size` in tokens, `type`). Token counts come from the tokenizer of the SentenceTransformer model named by `--model_name`; `semantic` chunking also uses that model to embed every sentence. All chunks of one document are saved together as a JSON list in `{output folder}/chunks/{chunk_type}/{model_name}/{min}_{max}_{overlap}_{threshold}/{doc_id}.json`. The file is written to a temporary file first and then renamed into place, so it is either complete or absent. A document whose file already exists is loaded from it instead of being chunked again, and for `semantic` chunking only the documents that are not cached have their sentences embedded. `compute_chunks()` returns all chunks as a dict of `chunk_id` to `Chunk` and prints the chunk folder and how many documents and chunks came from cache or were newly computed. `--framework` accepts `""` or `"langchain"`, but only the built-in chunkers exist so far.

### Architecture

- **`chunking.py`** — `Chunking`, the orchestrator: loads the SentenceTransformer model, validates the settings, loads cached chunks, embeds sentences for semantic chunking, chunks the remaining documents, saves them and prints the run summary.
- **`chunker.py`** — The abstract `Chunker` base class shared by all strategies (chunk id and size handling, and a sliding token-window splitter), and `get_chunker()`, the registry that maps a `chunk_type` to its strategy.
- **`chunkers/fixed.py`** — `FixedChunking`: splits a document's tokens into windows of `max_chunk_size` tokens that overlap by `fixed_chunk_overlap` tokens.
- **`chunkers/sentence.py`** — `SentenceChunking`: packs whole sentences into chunks of up to `max_chunk_size` tokens; a sentence longer than that is split with the fixed window.
- **`chunkers/semantic.py`** — `SemanticChunking`: like sentence chunking, but also starts a new chunk where the cosine similarity between neighbouring sentence embeddings falls below `semantic_threshold`, then merges chunks smaller than `min_chunk_size` into the previous chunk when they fit.
- **`cache.py`** — Saves and loads one JSON file per document in the chunk folder.
- **`validate.py`** — Checks the `ChunkingArgs` values before any chunking starts.
- **`schema.py`** — Defines `Chunk`, the layer's output record.

### Run results

All runs below use `pixparse/pdfa-eng-wds` with `--model_name all-MiniLM-L6-v2 --min_chunk_size 128 --max_chunk_size 256 --fixed_chunk_overlap 32 --semantic_threshold 0.2`. The times are the `Chunking time` printed by `src/rag.py`, which covers the whole `Chunking(...).compute_chunks(...)` call, including loading the model.

**100 documents, fixed** (from `log/l2_chunking/log_100_fixed.txt`):

- Documents: 100 (0 from cache, 100 newly chunked)
- Chunks: 1555 (0 from cache, 1555 new)
- Chunking time: 5.563709735870361s

**5000 documents, fixed / sentence / semantic** (from `log/l2_chunking/log_5k_fixed.txt`, `log_5k_sentence.txt`, `log_5k_semantic.txt`, and the chunk size statistics in `log/l2_chunking/chunk_stats_5k.txt`):

| | fixed | sentence | semantic |
|---|---|---|---|
| Documents from cache / newly chunked | 100 / 4900 | 0 / 5000 | 0 / 5000 |
| Chunks from cache / new | 1555 / 90811 | 0 / 94473 | 0 / 105050 |
| Total chunks | 92366 | 94473 | 105050 |
| Chunks per document | 18.5 | 18.9 | 21.0 |
| Mean chunk size (tokens) | 248.1 | 217.3 | 195.4 |
| Median chunk size (tokens) | 256.0 | 239 | 220.0 |
| Min / max chunk size (tokens) | 1 / 256 | 1 / 256 | 1 / 256 |
| Chunks below `min_chunk_size` | 2862 (3.1%) | 8673 (9.2%) | 17509 (16.7%) |
| Total tokens in all chunks | 22912548 | 20530137 | 20530137 |
| Sentence embedding time | – | – | 265.496s (639475 sentences) |
| Chunking time | 41.890793561935425s | 45.025853872299194s | 322.9234690666199s |

The fixed run's chunking time covers 4900 documents, because the 100 documents from the earlier 100-document run were already cached; the sentence and semantic times cover all 5000.

Fixed and sentence chunking take about the same time, because both spend it tokenizing every document and decoding or joining the pieces back into text, which is cheap. Fixed chunking fills almost every window to the full 256 tokens, and only the last window of each document is shorter, which is why its median is 256. It has about 2.4 million more tokens in total than the other two because neighbouring windows repeat 32 tokens. Sentence chunking only cuts between sentences, so a chunk usually ends some tokens before 256, giving a lower mean and median.

Semantic chunking is dominated by sentence embedding: 265.496s of its 322.9234690666199s go to running the SentenceTransformer model over all 639475 sentences, while the rest is the same sentence splitting and tokenizing work that sentence chunking does. It covers the same text as sentence chunking, with no overlap, so its total token count is identical (20530137), but it also cuts wherever neighbouring sentences are dissimilar, which produces more and smaller chunks; 16.7% stay below `min_chunk_size` even after small chunks are merged into the previous one.

All three strategies have chunks of 1 token. In fixed chunking these are the last windows of documents whose remaining tokens are already inside the previous window; in sentence and semantic chunking they are single-word sentences. The log line `Token indices sequence length is longer than the specified maximum sequence length for this model` is printed by the tokenizer when a whole document is tokenized for counting; the model is not run on those tokens, and every chunk is cut to at most 256 tokens.
