## l1_doc_processing

### Overview

`l1_doc_processing` is the first pipeline layer. It takes a dataset registered in `src/common/dataset_config.py` by `--dataset_name` — either a Hugging Face dataset streamed from `--dataset_url` or a folder of files at `--dataset_path` — plus a `--doc_type` and a `--dataset_size`, and produces `dataset_size` unique, parsed `Document` records. `dataset_size` counts processed documents, not raw rows: rows are fetched in batches sized to the number of documents still missing (downloaded or decoded in parallel), parsed, deduplicated, and fetched again for any shortfall caused by parse failures or duplicates, until `dataset_size` documents exist or the source runs out (in which case `main.py` raises an error, and the documents processed so far stay on disk). Raw files are written to `{dataset_name}/data/raw/{doc_id}.{extension}`, each processed `Document` to `{dataset_name}/data/processed/{doc_id}.json`, and the run state (rows read, processed and skipped `doc_id`s in order) to `{dataset_name}/data/progress.json`, which later runs validate and resume from instead of redoing work. `DocProcessing.process()` returns the documents as a dict of `doc_id` to `Document`, in processing order.

### Architecture

- **`main.py`** — CLI entry point. Parses the layer's arguments, requires exactly one of `--dataset_url`/`--dataset_path`, runs `DocProcessing.process()`, raises if fewer than `dataset_size` documents were produced, and prints the layer's total run time.
- **`doc_processing.py`** — `DocProcessing`, the orchestrator: validates the dataset and saved progress, loads already processed documents, then runs the fetch → parse → dedup → save batch loop and saves progress after each batch.
- **`raw_source.py`** — `RawSource`, which reads the dataset's source rows (streamed Hugging Face rows or sorted local files), applies the dataset's `row_filter`, decodes or downloads each row's content in parallel, and saves it to `raw/`.
- **`cache.py`** — Saves and loads processed `Document`s as JSON files in `processed/`, and loads a dataset's first processed documents in `progress.json` order.
- **`progress.py`** — The `Progress` dataclass and the functions that build, save and load `progress.json`.
- **`dedup.py`** — `Deduplicator`, which detects exact duplicates by content hash and near duplicates with MinHash + LSH over word shingles.
- **`summary.py`** — Counts skipped rows and documents by reason, and prints the run summary and the document parse time summary.
- **`schema.py`** — Defines `Document`, the layer's output record.
- **`parser.py`** — The abstract `Parser` class and the registry that maps a `doc_type` to its parser.
- **`parsers/pdf.py`** — `PDFParser`: extracts PDF text, title, author and page count with `pypdf`.
- **`parsers/docx.py`** — `DocxParser`: extracts Word `.docx` body text, title, author and page count using only the standard library (`zipfile`, `xml.etree`).
- **`text_cleanup.py`** — `clean_text()`: removes repeated header/footer lines (optional per parser) and collapses extra whitespace and blank lines.
- **`metadata.py`** — `extract_metadata()`: the format-independent metadata (`format`, `raw_bytes_size`) computed from the raw bytes.

### Run results

Both runs started from an empty `{dataset_name}/` folder.

**CUAD, pdf, 500 documents** (`--dataset_name cuad --doc_type pdf --dataset_size 500 --dataset_url dvgodoy/CUAD_v1_Contract_Understanding_PDF`, from `log/l1_doc_processing/log.txt`):

- Batches: batch 1 fetched 500 rows, batch 2 fetched 1 row
- Documents newly processed: 500
- Rows skipped by `row_filter`: 0
- Rows skipped due to decode or download errors: 0
- Documents skipped due to parse errors: 0
- Documents skipped as duplicates: 1

| Parse time | |
|---|---|
| Documents parsed | 501 |
| Total | 146.711s |
| Average | 0.293s |
| Fastest | 0.004s (`pdf_85a6eed70dbd355f`) |
| Slowest | 52.787s (`pdf_47aecdabb5438fa7`) |

- L1 total time: 185.038s (3m 5.0s)

**docx-corpus, docx, 1,000 documents** (`--dataset_name docx-corpus --doc_type docx --dataset_size 1000 --dataset_url superdoc-dev/docx-corpus`, from `log/l1_doc_processing/log_docx-corpus.txt`):

- Batches: batch 1 fetched 1000 rows, batch 2 fetched 5 rows
- Documents newly processed: 1000
- Rows skipped by `row_filter` (not English): 2104
- Rows skipped due to decode or download errors: 0
- Documents skipped due to parse errors: 2 (`docx_6f0aa9bed2d655bf`, `docx_2155fedff637e8b5`: `KeyError: "There is no item named 'word/document.xml' in the archive"`)
- Documents skipped as duplicates: 3

| Parse time | |
|---|---|
| Documents parsed | 1003 |
| Total | 9.340s |
| Average | 0.009s |
| Fastest | 0.000s (`docx_2ca28b4c230af3a1`) |
| Slowest | 0.559s (`docx_094d9d56dd7c68a5`) |

- L1 total time: 46.541s (0m 46.5s)

In both runs the second batch fetched exactly the number of documents the first batch fell short by: 1 duplicate for CUAD, and 2 parse failures plus 3 duplicates for docx-corpus. "Documents parsed" is one per document that parsed successfully, including the duplicates, because a duplicate can only be recognized from its parsed text.

For CUAD, parsing takes 146.711s of the 185.038s run. `PDFParser` spends that time in `pypdf`'s page-by-page text extraction; CUAD's documents are long legal contracts, and a single document (`pdf_47aecdabb5438fa7`, 52.787s) accounts for about a third of all parse time on its own. The remaining 38.327s covers the dataset check, streaming the rows (whose PDFs are embedded in the rows as base64), deduplication and file writes; the log does not break that remainder down further.

For docx-corpus, parsing takes only 9.340s of the 46.541s run, because a `.docx` file is a zip of XML that `DocxParser` reads directly, with no page layout to reconstruct. Most of the remaining 37.201s is network time: streaming 3,109 metadata rows, of which 2,104 are dropped by the English `row_filter` without being downloaded, and downloading 1,005 files from the dataset's download links, 16 at a time. The log does not break that remainder down further. The 2 parse failures are files that are valid zip archives but contain no `word/document.xml`, so they are not Word documents despite their `.docx` name.

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
