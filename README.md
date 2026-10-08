## l2_chunking

### Overview

`l2_chunking` is the second pipeline layer. It takes the documents `l1_doc_processing` produced for a dataset — the first `--dataset_size` doc_ids listed in `{dataset_name}/data/progress.json`, loaded from `{dataset_name}/data/processed/` in that order — and splits each one into `Chunk` records (`id`, `text`, `size` in tokens, `type`) using the strategy named by `--chunk_type`: `fixed`, `sentence` or `semantic`. The chunk parameters (`--min_chunk_size`, `--max_chunk_size`, `--fixed_chunk_overlap`, `--semantic_threshold`) travel together as a `ChunkConfig`. Before loading anything, the layer checks that the chunk config is valid and that enough processed documents and a `progress.json` exist. Token counts come from the tokenizer of the SentenceTransformer model given by `--model_name`; `semantic` chunking also uses that model to embed every sentence. Each chunk is written to `{dataset_name}/chunks/{chunk_type}/{min}_{max}_{overlap}_{threshold}/{doc_id}_{n}.json`; a document whose chunks already exist in that folder is loaded from there instead of being chunked again. `Chunking.compute_chunks()` returns all chunks as a dict of `chunk_id` to `Chunk`.

### Architecture

- **`main.py`** — CLI entry point. Parses the layer's arguments, builds the `ChunkConfig`, runs `Chunking.compute_chunks()`, and prints the chunk statistics and the layer's total run time.
- **`chunking.py`** — `Chunking`, the orchestrator: validates the config and processed dataset, loads the documents, computes sentence embeddings for semantic chunking, chunks each document or loads its cached chunks, and prints the run summaries.
- **`chunker.py`** — The abstract `Chunker` base class shared by all strategies (chunk id and size handling, and a sliding token-window splitter), and `get_chunker()`, the registry that maps a `chunk_type` to its strategy.
- **`chunkers/fixed.py`** — `FixedChunking`: splits a document's tokens into windows of `max_chunk_size` tokens that overlap by `fixed_chunk_overlap` tokens.
- **`chunkers/sentence.py`** — `SentenceChunking`: packs whole sentences into chunks of up to `max_chunk_size` tokens; a sentence longer than that is split with the fixed window.
- **`chunkers/semantic.py`** — `SemanticChunking`: like sentence chunking, but also starts a new chunk where the cosine similarity between neighbouring sentence embeddings falls below `semantic_threshold`, then merges chunks smaller than `min_chunk_size` into the previous chunk when they fit.
- **`cache.py`** — Saves and loads chunks as JSON files in the chunk folder.
- **`summary.py`** — Prints the run summary, the sentence embedding time, the per-document chunk time, and the chunk statistics.
- **`schema.py`** — Defines `Chunk`, the layer's output record.

### Run results

**docx-corpus, fixed, 500 documents** (`--dataset_name docx-corpus --dataset_size 500 --chunk_type fixed --min_chunk_size 128 --max_chunk_size 256 --fixed_chunk_overlap 32 --semantic_threshold 0.3 --model_name all-MiniLM-L6-v2`, from `log/l2_chunking/log_docx-corpus_fixed.txt`):

- Processed documents loaded: 500
- Documents with cached chunks: 0 (0 chunks)
- Documents newly chunked: 500 (9835 chunks)
- Total chunks: 9835

| Document chunk time | |
|---|---|
| Documents chunked | 500 |
| Total | 3.902s |
| Average | 0.008s |
| Fastest | 0.000s (`docx_b94b6f44ab3e0331`) |
| Slowest | 0.413s (`docx_0dfbdf0ea8c4cec6`) |

| Chunk statistics | |
|---|---|
| Total chunks | 9835 |
| Average chunk size | 248.6 tokens |
| Total chunk file size | 11965826 bytes |
| Average chunk file size | 1216.7 bytes |

- L2 total time: 9.216s (0m 9.2s)

Chunking the 500 documents takes 3.902s of the 9.216s run. Fixed chunking tokenizes a document once, then cuts the token list into 256-token windows, moving 224 tokens forward each time (256 minus the 32-token overlap), and decodes each window back to text. Every window is a full 256 tokens except the last one of each document, which is why the average chunk size (248.6 tokens) is close to the maximum. The remaining 5.314s covers loading the SentenceTransformer model and tokenizer at startup, reading the 500 processed documents, and writing the 9835 chunk files; the log does not break that remainder down further. The log line `Token indices sequence length is longer than the specified maximum sequence length for this model (343 > 256)` is printed by the tokenizer when a whole document is tokenized at once; the model is not run on those tokens, and every chunk is cut to at most 256 tokens.

**CUAD, fixed, 19 documents** (`--dataset_name CUAD --dataset_size 19`, same chunk settings, from `log/l2_chunking/log_fixed.txt`). This run was made with an earlier version of the layer, before the run summary and document chunk time were added, so its log only has the total time and chunk statistics:

- Chunking elapsed time: 4.553313493728638s
- Total chunks produced: 860
- Average chunk size: 253.00348837209302 tokens
- Total chunk output size: 1144670 bytes
- Average chunk file size: 1331.0116279069769 bytes

The log's stage timing breakdown table is empty, because no stage recorded a time in that version. CUAD's documents are long contracts, so fewer documents produce more full-size windows per document, and the average chunk size (253.0 tokens) is closer to 256 than in the docx run.
