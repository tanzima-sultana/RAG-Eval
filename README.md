## l1_doc_processing

### Overview

`l1_doc_processing` is the first pipeline layer. It takes a dataset registered in `src/common/dataset_config.py` by `--dataset_name` — either a Hugging Face dataset streamed from `--dataset_url` or a folder of files at `--dataset_path` — plus a `--doc_type` and a `--dataset_size`, and produces `dataset_size` unique, parsed `Document` records. `dataset_size` counts processed documents, not raw rows: rows are fetched in batches sized to the number of documents still missing (downloaded or decoded in parallel), parsed, deduplicated, and fetched again for any shortfall caused by parse failures or duplicates, until `dataset_size` documents exist or the source runs out (in which case `main.py` raises an error, and the documents processed so far stay on disk). Raw files are written to `{dataset_name}/data/raw/{doc_id}.{extension}`, each processed `Document` to `{dataset_name}/data/processed/{doc_id}.json`, and the run state (rows read, processed and skipped `doc_id`s in order) to `{dataset_name}/data/progress.json`, which later runs validate and resume from instead of redoing work. `DocProcessing.process()` returns the documents as a dict of `doc_id` to `Document`, in processing order.

### Architecture

- **`main.py`** — CLI entry point. Parses the layer's arguments, requires exactly one of `--dataset_url`/`--dataset_path`, runs `DocProcessing.process()`, raises if fewer than `dataset_size` documents were produced, and prints the layer's total run time.
- **`doc_processing.py`** — `DocProcessing`, the orchestrator: validates the dataset and saved progress, loads already processed documents, then runs the fetch → parse → dedup → save batch loop and saves progress after each batch.
- **`raw_source.py`** — `RawSource`, which reads the dataset's source rows (streamed Hugging Face rows or sorted local files), applies the dataset's `row_filter`, decodes or downloads each row's content in parallel, and saves it to `raw/`.
- **`document_store.py`** — Saves and loads processed `Document`s as JSON files in `processed/`.
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
