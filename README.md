## l1_doc_processing

### Overview

`l1_doc_processing` is the first pipeline layer. It takes raw source documents — either streamed from a Hugging Face dataset (`--dataset_url`) or read from a local directory (`--dataset_path`) — and turns each one into a normalized `Document` record: extracted text, title, format-specific metadata, a content hash, and a timestamp. Raw bytes are cached to `data/raw/{doc_type}_{dataset_name}/`, keyed by a content-addressed `doc_id`. Parsed output is cached to `data/processed/{doc_type}_{dataset_name}/docs/` as one JSON file per `doc_id`. Both caches are consulted before any network or parsing work happens, so re-running the layer with the same `doc_type`/`dataset_name` reuses whatever was already fetched or parsed rather than redoing it. Exact and near-duplicate documents are detected during processing and excluded from the output rather than being written to `data/processed/`.

### Architecture

- **`schema.py`** — Defines `Document`, the dataclass every parsed document is normalized into: `id`, `text`, `title`, `path`, `format`, `metadata`, `content_hash`, and `timestamp`. This is the shared record type the rest of the layer, and downstream layers, consume.
- **`parser.py`** — Defines `Parser`, the abstract base class every format-specific parser implements (a single `parse(doc_id, raw_bytes, data_path)` method), and `get_parser(doc_type)`, the registry lookup that maps a `doc_type` string to its `Parser` implementation. The registry is built lazily inside `get_parser` to avoid a circular import with the concrete parser modules.
- **`parsers/pdf.py`** — `PDFParser`, the only registered parser (`doc_type="pdf"`). Uses `pypdf` to extract per-page text (joined across pages with a blank line between them), pull `title`/`author` from the PDF's document information dictionary if present, and record `page_count`. This is the only module allowed to depend on `pypdf` or know anything PDF-specific.
- **`text_cleanup.py`** — `clean_text()`, run on every document's extracted text before it's stored. Strips lines that repeat often enough across the document to be a running header or footer, collapses runs of in-line whitespace left over from the source's visual layout, and collapses repeated blank lines. Each rule's threshold is a parameter with a default, not a hardcoded constant.
- **`metadata.py`** — `extract_metadata()`, the format-agnostic half of a document's metadata: `format` and `raw_bytes_size`, computed directly from the raw bytes before any parsing happens. Format-specific metadata (e.g. `page_count`, `author` from `pdf.py`) is layered on top of this by the parser.
- **`dedup.py`** — Duplicate detection. `is_exact_duplicate()` compares a document's `content_hash` against every `content_hash` already seen. `make_minhash()` and `make_lsh_index()` build and query a MinHash/LSH index (via `datasketch`) over each document's word-shingle set (from `common.identifiers.make_shingles`), and `is_near_duplicate()` reports whether an incrementally built index already holds a document estimated to be similar enough (default Jaccard threshold 0.90) to be a near-duplicate, without pairwise-comparing against every previously kept document.
- **`doc_processing.py`** — `DocProcessing`, the class that orchestrates the layer. `validate()` checks that the requested `doc_type` matches the dataset's registered `DatasetConfig` and, where the dataset publishes a document count, that the requested `dataset_size` doesn't exceed what's actually available. `compute_raw_data()` fills in raw bytes for the request from the raw cache, a streaming Hugging Face dataset, or a local directory. `process_raw_data()` parses each document (reusing an already-processed one from the processed cache where present), runs it through the exact/near-duplicate checks in `dedup.py`, and persists the ones that pass.
- **`main.py`** — The CLI entry point for this layer. Parses `--doc_type`, `--dataset_name`, `--dataset_size`, `--dataset_url`, and `--dataset_path`; enforces that exactly one of `--dataset_url`/`--dataset_path` is given; and drives a `DocProcessing` instance through `process()`.

### 1k document run

Run against `pixparse/pdfa-eng-wds` (`--dataset_url pixparse/pdfa-eng-wds --dataset_size 1000`), starting from an empty raw/processed cache:

- Raw ingestion errors: 0
- Parsing errors: 9
- Duplicate documents skipped: 0
- Documents successfully processed: 991 (out of 1000 raw documents fetched)
- Total elapsed time: 358.5348250865936s

Per-stage timing breakdown (slowest first):

| Stage | Total time | Calls | Avg/call |
|---|---|---|---|
| `process_raw_data_total` | 320.939s | 1 | 320.93885s |
| `parse_document` | 308.806s | 991 | 0.31161s |
| `pdf_extract_text` | 307.211s | 991 | 0.31000s |
| `compute_raw_data_total` | 35.523s | 1 | 35.52264s |
| `download_row` | 29.925s | 1000 | 0.02992s |
| `make_minhash` | 11.102s | 990 | 0.01121s |
| `pdf_open_reader` | 0.844s | 1000 | 0.00084s |
| `make_shingles` | 0.626s | 991 | 0.00063s |
| `text_cleanup` | 0.618s | 991 | 0.00062s |
| `decode_raw_bytes` | 0.540s | 1000 | 0.00054s |
| `write_raw_cache_file` | 0.308s | 1000 | 0.00031s |
| `save_processed_document` | 0.198s | 991 | 0.00020s |
| `pdf_extract_title` | 0.117s | 991 | 0.00012s |
| `dedup_check` | 0.022s | 991 | 0.00002s |
| `make_content_hash` | 0.019s | 991 | 0.00002s |
| `load_processed_document` | 0.013s | 1000 | 0.00001s |
| `extract_metadata` | 0.011s | 991 | 0.00001s |
| `load_cache` | 0.000s | 1 | 0.00001s |

`pdf_extract_text` — `pypdf`'s own page-by-page text extraction inside `PDFParser.parse()` — accounts for 307.211s of the 358.5s run, and for 307.211s of the 308.806s spent in `parse_document`: parsing is almost entirely text extraction, with reader setup (`pdf_open_reader`), title lookup (`pdf_extract_title`), our own cleanup (`text_cleanup`), and metadata extraction (`extract_metadata`) contributing very little by comparison. `download_row`, the time spent waiting on the Hugging Face streaming iterator for each row, totals 29.925s across all 1000 rows — an order of magnitude less than text extraction. `make_minhash`, run once per successfully parsed document to build its near-duplicate signature, totals 11.102s; every other stage (shingling, dedup lookups, content hashing, disk reads/writes) is under a second in total across all 991-1000 calls. The 9 parsing errors are documents where `PDFParser.parse()` raised (caught and counted in `process_raw_data()`); the 0 duplicate count means no document's `content_hash` or MinHash signature matched an earlier one in this run.
