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

## l2_chunking

### Overview

`l2_chunking` is the second pipeline layer. It reads the `Document` records `l1_doc_processing` already wrote to `data/processed/{doc_type}_{dataset_name}/docs/` and splits each one's text into `Chunk` records — `id`, `text`, `size` (token count), and `type` — using whichever strategy is selected by `--chunk_type` (`fixed`, `sentence`, or `semantic`), together with `--min_chunk_size`, `--max_chunk_size`, `--fixed_chunk_overlap`, `--semantic_threshold`, and `--model_name`. Output is written to `chunks/{chunk_type}/{doc_type}_{dataset_name}/{max_chunk_size}_{fixed_chunk_overlap}_{semantic_threshold}/` as one JSON file per chunk, and returned as a dict of `chunk_id` to `Chunk`. If that output directory already exists from a prior run with the same `max_chunk_size`/`fixed_chunk_overlap`/`semantic_threshold`, chunks are read back from disk instead of being recomputed — the cache key does not currently include `min_chunk_size`, so two runs that differ only in `--min_chunk_size` will collide on the same cached output; the second run's `--min_chunk_size` is silently ignored rather than triggering a recompute.

### Architecture

- **`schema.py`** — Defines `Chunk`, the dataclass every chunking strategy produces: `id`, `text`, `size`, and `type`.
- **`chunker.py`** — Defines `Chunker`, the abstract base class every strategy subclasses, and `get_chunker(chunk_type)`, the registry lookup that maps a `chunk_type` string to its strategy class. `Chunker` also holds the config and model state copied over from the running `Chunking` instance, plus two helpers every strategy reuses: `_split_tokens_into_chunks()` (the fixed-window sliding split) and `_finalize_chunk()` (building a `Chunk` from text and its token count).
- **`chunkers/fixed.py`** — `FixedChunking`. Slides a fixed-size token window across a document's whole text, advancing by `max_chunk_size - fixed_chunk_overlap` tokens each step.
- **`chunkers/sentence.py`** — `SentenceChunking`. Accumulates whole sentences into a chunk until adding the next one would exceed `max_chunk_size`, then starts a new chunk with that sentence; a single sentence longer than `max_chunk_size` on its own falls back to `Chunker`'s fixed-window split.
- **`chunkers/semantic.py`** — `SemanticChunking`. Extends the same sentence-accumulation approach with a second closing condition: a chunk is also ended when the cosine similarity between consecutive sentences' embeddings drops below `semantic_threshold`. After the whole document is split this way, `merge_chunks()` makes a second pass over the resulting chunks and folds any chunk smaller than `min_chunk_size` into the chunk immediately before it, as long as the combined size still fits `max_chunk_size` — this is what keeps a low `semantic_threshold` (which splits often) from leaving behind a lot of very small chunks.
- **`chunking.py`** — Defines `Chunking`, the class that orchestrates the layer: loads the embedding model (`--model_name`) and its tokenizer, loads the requested dataset's processed documents, computes sentence embeddings for the whole dataset up front when `chunk_type` is `semantic`, dispatches each document to the selected strategy, and saves/caches the resulting chunks.
- **`main.py`** — The CLI entry point for this layer. Parses `--chunk_type`, `--min_chunk_size`, `--max_chunk_size`, `--fixed_chunk_overlap`, `--semantic_threshold`, and `--model_name`; exits with an error if `--chunk_type semantic` is requested without a nonzero `--semantic_threshold`; and drives a `Chunking` instance through `compute_chunks()`.

### 991-document run (fixed / sentence / semantic)

Run against the already-processed `pdf_pdfa_eng_wds` documents under `data/processed/` (991 documents, from `l1_doc_processing`'s earlier 1k run — `l1_doc_processing` was not re-run for this), with `--min_chunk_size 128 --max_chunk_size 256 --fixed_chunk_overlap 32 --semantic_threshold 0.3 --model_name all-MiniLM-L6-v2`, one run per `--chunk_type`:

| `chunk_type` | elapsed time | chunks produced | avg chunk size (tokens) | total output size (bytes) | avg chunk file size (bytes) |
|---|---|---|---|---|---|
| `fixed` | 16.53658366203308s | 16,435 | 247.04472163066626 | 18,714,750 | 1,138.7131122604198 |
| `sentence` | 11.789691925048828s | 16,590 | 219.01404460518384 | 17,002,756 | 1,024.8798071127185 |
| `semantic` | 58.50146722793579s | 18,608 | 195.26241401547722 | 17,164,016 | 922.3998280309544 |

`fixed` and `sentence` were served from the on-disk cache for this run (`min_chunk_size` doesn't affect either strategy, and their cache entries from an earlier run at the same `max_chunk_size`/`fixed_chunk_overlap`/`semantic_threshold` already existed), so their numbers above are identical to the ones from that earlier run. `semantic`'s cache entry was deleted before this run specifically so it would recompute with `merge_chunks()` applied, rather than silently serving stale, pre-merge output (see the cache-key gap noted in Overview above).

No per-stage timing breakdown is available for this layer — `l2_chunking` never calls `common.timing.record()`, so that section of the log is always empty regardless of `chunk_type`.

`semantic` takes roughly 3.5-5x as long as `fixed`/`sentence` because, on top of the same sentence-accumulation work `sentence` already does, it runs every sentence in the dataset through the `all-MiniLM-L6-v2` embedding model once up front (the `model.encode()` call inside `Chunking.compute_chunks`) before any chunk can be produced — an actual neural-network forward pass per sentence, which `fixed` and `sentence` never perform. `fixed` and `sentence` produce a similar number of chunks (16,435 vs 16,590) but `fixed` still takes noticeably longer (16.5s vs 11.8s); `fixed` tokenizes and detokenizes each document's text as one long, unsegmented token stream, while `sentence` tokenizes many shorter, sentence-length strings, which plausibly accounts for the difference, though no per-stage timing is recorded inside `l2_chunking` to confirm which specific step is responsible.

#### Effect of `merge_chunks()` on `semantic`'s output

Before `merge_chunks()` existed, a `semantic` run at these same parameters (`max_chunk_size=256`, `semantic_threshold=0.3`) produced 59,086 chunks averaging 61.49 tokens each, for 20,240,180 total output bytes (342.55 bytes/file average) — see the numbers this same run produced prior to this session's `merge_chunks()` work. With `merge_chunks()` now folding any chunk under `min_chunk_size=128` tokens into its immediate predecessor (when the combined size still fits within `max_chunk_size=256`), the same 991 documents at the same `semantic_threshold` now produce:

- **68.5% fewer chunks** (18,608 vs 59,086) — most of the very small, dissimilarity-triggered chunks get absorbed into a neighbor instead of staying as their own chunk.
- **~3.2x larger average chunk** (195.26 vs 61.49 tokens) — consistent with most chunks no longer stopping the moment a dissimilarity boundary is hit; they keep absorbing small follow-on chunks until either `max_chunk_size` or a large-enough neighbor stops them.
- **15.2% smaller total output** (17,164,016 vs 20,240,180 bytes) despite each individual chunk now holding roughly the same underlying document text — with a third as many files, the fixed per-file JSON overhead (`id`, `text`, `size`, `type` keys and formatting) is paid far fewer times.
- **Elapsed time is effectively unchanged** (58.50s vs 59.49s) — `merge_chunks()` is one linear pass over already-computed chunks; the embedding-model forward pass that dominates `semantic`'s runtime happens identically either way.
