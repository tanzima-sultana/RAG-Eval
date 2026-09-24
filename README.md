## l1_doc_processing

### Overview

`l1_doc_processing` is the first pipeline layer. It takes a dataset registered in `src/common/dataset_config.py` by `--dataset_name` — either a Hugging Face dataset streamed from `--dataset_url` or a folder of files at `--dataset_path` — plus a `--doc_type` and a `--dataset_size`, and produces `dataset_size` unique, parsed `Document` records. `dataset_size` counts processed documents, not raw rows: rows are fetched in batches sized to the number of documents still missing (downloaded or decoded in `DOWNLOAD_WORKERS` threads), parsed in `PARSE_WORKERS` worker processes, deduplicated, and fetched again for any shortfall caused by parse failures or duplicates, until `dataset_size` documents exist or the source runs out (in which case `main.py` raises an error, and the documents processed so far stay on disk). Raw files are written to `{dataset_name}/data/raw/{doc_id}.{extension}`, each processed `Document` to `{dataset_name}/data/processed/{doc_id}.json`, and the run state (rows read, processed and skipped `doc_id`s in order) to `{dataset_name}/data/progress.json`, which later runs validate and resume from instead of redoing work. `DocProcessing.process()` returns the documents as a dict of `doc_id` to `Document`, in processing order.

### Architecture

- **`main.py`** — CLI entry point. Parses the layer's arguments, requires exactly one of `--dataset_url`/`--dataset_path`, runs `DocProcessing.process()`, raises if fewer than `dataset_size` documents were produced, and prints the layer's total run time.
- **`doc_processing.py`** — `DocProcessing`, the orchestrator: validates the dataset and saved progress, loads already processed documents, then runs the fetch → parse → dedup → save batch loop and saves progress after each batch. Each batch is parsed in a process pool; dedup and saves run in the main process, in row order.
- **`raw_source.py`** — `RawSource`, which reads the dataset's source rows (streamed Hugging Face rows or sorted local files), applies the dataset's `row_filter`, decodes or downloads each row's content in parallel, and saves it to `raw/`.
- **`cache.py`** — Saves and loads processed `Document`s as JSON files in `processed/`, and loads a dataset's first processed documents in `progress.json` order.
- **`progress.py`** — The `Progress` dataclass and the functions that build, save and load `progress.json`.
- **`dedup.py`** — `Deduplicator`, which detects exact duplicates by content hash and near duplicates with MinHash + LSH over word shingles.
- **`summary.py`** — Counts skipped rows and documents by reason, and prints the run summary and the document parse time summary.
- **`schema.py`** — Defines `Document`, the layer's output record.
- **`parser.py`** — The abstract `Parser` class, the registry that maps a `doc_type` to its parser, and `parse_document_from_file()`, the function each parse worker process runs.
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

#### docx-corpus: 1,000 vs 5,000 documents

Two consecutive runs of `scripts/run_doc_processing.sh` settings (`--dataset_name docx-corpus --doc_type docx --dataset_url superdoc-dev/docx-corpus`). The 1,000 run started from an empty `docx-corpus/` folder (from `log/l1_doc_processing/log_docx-corpus_1000.txt`). The 5,000 run then resumed from its `progress.json`: it loaded the 1,000 documents already processed and skipped the 3,109 source rows already read, so it only fetched the 4,000 missing documents (from `log/l1_doc_processing/log_docx-corpus_5000_dl16_resumed.txt`). Both runs used 16 download workers and parsed one document at a time. This 1,000 run is a separate run from the docx-corpus 1,000 run above, so its timings differ.

| | 1,000 run | 5,000 run |
|---|---|---|
| Already processed documents loaded | 0 | 1000 |
| Batches | batch 1: 1000 rows, batch 2: 5 rows | batch 1: 4000 rows, batch 2: 21 rows |
| Documents newly processed | 1000 | 4000 |
| Total documents | 1000 | 5000 |
| Rows skipped by `row_filter` (not English) | 2104 | 7718 |
| Rows skipped due to decode or download errors | 0 | 0 |
| Documents skipped as failed in an earlier run | 0 | 0 |
| Documents skipped due to parse errors | 2 | 0 |
| Documents skipped as duplicates | 3 | 21 |
| Documents parsed | 1003 | 4021 |
| Parse time total | 9.126s | 35.426s |
| Parse time average | 0.009s | 0.009s |
| Fastest parse | 0.000s (`docx_2ca28b4c230af3a1`) | 0.000s (`docx_88216cc1c7d17ca4`) |
| Slowest parse | 0.540s (`docx_bbe05264ab95cce9`) | 0.532s (`docx_dd65460867bc5777`) |
| L1 total time | 55.795s (0m 55.8s) | 209.823s (3m 29.8s) |

The 1,000 run's 2 parse failures are the same two non-Word files as in the earlier run (`docx_6f0aa9bed2d655bf`, `docx_2155fedff637e8b5`). The 5,000 run reports 0 documents skipped as failed in an earlier run because it started reading after the rows that held them, so it never saw them again.

Each run's second batch fetched exactly its first batch's shortfall: 2 parse failures plus 3 duplicates in the 1,000 run, and 21 duplicates in the 5,000 run.

Source rows read: 3,109 in the 1,000 run (1,005 fetched + 2,104 filtered) and 11,739 in the 5,000 run (4,021 fetched + 7,718 filtered). Together that is the 14,848 `rows_consumed` saved in `docx-corpus/data/progress.json`. The `row_filter` dropped about the same share of rows in both runs: 67.7% and 65.7%.

Run time grows in step with the number of source rows read. The 5,000 run read 3.78× as many rows as the 1,000 run and took 3.76× as long. Per newly processed document that is 0.056s in the 1,000 run and 0.052s in the 5,000 run. Parsing costs the same per document in both runs (0.009s average) and is a small share of each run: 9.126s of 55.795s, and 35.426s of 209.823s. The rest (46.669s and 174.397s) goes mostly to streaming metadata rows and downloading files, 16 at a time. The log does not break that remainder down further. Duplicates rise from 3 in 1,003 parsed documents to 21 in 4,021. Each new document is checked against every document processed before it, including the 1,000 loaded from the earlier run, so the pool it can collide with keeps growing.

#### docx-corpus 5,000: download and parse workers

Same settings as above, with `DOWNLOAD_WORKERS` and `PARSE_WORKERS` (`src/common/constants.py`) changed between runs. Every run except the first started from an empty `docx-corpus/` folder. The first row combines the 1,000 run and the resumed 5,000 run above. Runs with "yes" in the token column had `HF_TOKEN` set from `.env`; the others sent unauthenticated Hugging Face requests. `PARSE_DOC_COUNT` (documents sent to a parse worker at a time) was 16 in every parallel run. "Parse workers: 1" means parsing ran in the main process, one document at a time, before parallel parsing was added.

| Download workers | Parse workers | `HF_TOKEN` | Parse time total | Parse time average | Slowest parse | L1 total time | Log |
|---|---|---|---|---|---|---|---|
| 16 | 1 | no | 44.552s (9.126s + 35.426s) | 0.009s | 0.540s (`docx_bbe05264ab95cce9`) | 265.618s (55.795s + 209.823s) | `log_docx-corpus_1000.txt` + `log_docx-corpus_5000_dl16_resumed.txt` |
| 64 | 1 | no | 45.650s | 0.009s | 0.581s (`docx_bbe05264ab95cce9`) | 158.651s (2m 38.7s) | `log_docx-corpus_5000_dl64.txt` |
| 64 | 16 | yes | 98.916s | 0.020s | 1.220s (`docx_dd65460867bc5777`) | 129.603s (2m 9.6s) | `log_docx-corpus_5000_dl64_parse16.txt` |
| 128 | 16 | yes | 96.428s | 0.019s | 1.244s (`docx_bbe05264ab95cce9`) | 121.458s (2m 1.5s) | `log_docx-corpus_5000_dl128_parse16.txt` |
| 64 | 32 | yes | 190.581s | 0.038s | 2.419s (`docx_094d9d56dd7c68a5`) | 124.649s (2m 4.6s) | `log_docx-corpus_5000_dl64_parse32.txt` |

All logs are in `log/l1_doc_processing/`. The document counts are identical in every run: 5,000 documents processed, 9,822 rows skipped by `row_filter`, 0 decode or download errors, 2 parse errors (`docx_6f0aa9bed2d655bf`, `docx_2155fedff637e8b5`), 24 duplicates and 5,024 documents parsed. The runs from an empty folder all used 2 batches, of 5,000 and 26 rows. So the worker counts change only the timing, not the output: parse results come back in row order and dedup runs in that same order.

"Parse time total" is the sum of each document's own parse time. With one parse worker it is the time spent parsing. With parallel workers it is not: 16 or 32 documents are parsed at once, and the log does not record how long the parse step took from start to finish. Dividing the sum by the worker count gives 6.182s, 6.027s and 5.956s for the three parallel runs, which is what the parse step takes if the work is spread evenly across the workers.

- **16 → 64 download workers** (one parse worker, no token): 265.618s → 158.651s, 106.967s (40.3%) less. Parse time is almost unchanged (44.552s vs 45.650s), so the whole difference is outside parsing: 221.066s → 113.001s. With 16 workers, most of that time is spent waiting on individual downloads; 64 workers keep four times as many downloads in progress. The 16-worker figure also includes a second run's startup and dataset check.
- **1 → 16 parse workers, and `HF_TOKEN` added**: 158.651s → 129.603s, 29.048s (18.3%) less. Both changes happened in the same run, and the log does not separate streaming, download and parse time, so their shares of the difference cannot be told apart. The parse step drops from 45.650s to about 6s, roughly 39.5s less, which is more than the 29.048s the whole run saved. The time outside parsing was therefore about 10s longer in this run, which is within the run-to-run variation of download speed.
- **64 → 128 download workers**: 129.603s → 121.458s, 8.145s (6.3%) less. Doubling the download workers no longer halves the time outside parsing.
- **16 → 32 parse workers**: 129.603s → 124.649s, 4.954s (3.8%) less. The average parse time per document doubles (0.020s → 0.038s), and the summed time divided by the worker count stays at about 6s. The machine (Intel Core i7-12700) has 12 physical cores and 20 threads, so 32 worker processes take turns on the same threads and each one runs about half as fast. The same effect explains why the average grows from 0.009s to 0.020s with 16 workers: some of them share a physical core or run on the slower efficiency cores.

Each setup was run once, and download speed varies from run to run, so gaps of a few seconds between the last three runs are within that variation. In those three runs, parsing takes about 6s of 121–130s. The rest is spent outside parsing: reading and filtering 14,848 source rows one at a time in the main process, downloading 5,026 files from the dataset's download links, and dedup and saves, which run one document at a time in row order. Within each batch, parsing starts only after every download in that batch has finished. The log does not break this time down further.

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
