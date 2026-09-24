"""
test_raw_source.py

Tests for l1_doc_processing.raw_source.RawSource: reading local and web
source rows, the row_filter, decoding/downloading content, saving raw
files, and fetching batches in parallel. Downloads use a fake urlopen,
so no network access is needed.
"""

import os
import threading
import time

import pytest

from common.dataset_config import DatasetConfig
from l1_doc_processing import raw_source as raw_source_module
from l1_doc_processing.raw_source import RawSource
from l1_doc_processing.summary import SummaryCounts


DATASET_NAME = "myset"


class _FakeUrlResponse:
    """
    Stand-in for the response urllib.request.urlopen returns: a context
    manager whose read() returns fixed bytes.
    """

    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return self.body


def _make_config(content_encoding="bytes", row_filter=None):
    """
    content_encoding: content_encoding of the config.
    row_filter: row_filter of the config.

    Returns a DatasetConfig whose rows keep their content in "content".
    """
    return DatasetConfig(
        identifier="someone/dataset",
        doc_type="pdf",
        extension="pdf",
        content_field="content",
        content_encoding=content_encoding,
        file_name_field="name",
        split="train",
        row_filter=row_filter,
    )


def _make_web_source(content_encoding="bytes", row_filter=None):
    """
    Returns a RawSource for a web dataset with the given config values.
    """
    return RawSource(DATASET_NAME, _make_config(content_encoding, row_filter), "pdf", "someone/dataset", None)


def _make_local_source(source_dir):
    """
    source_dir: folder holding the local dataset files.

    Returns a RawSource for a local dataset.
    """
    return RawSource(DATASET_NAME, _make_config(), "pdf", None, str(source_dir))


def _write_files(source_dir, file_texts):
    """
    source_dir: folder to write into.
    file_texts: list of (file_name, text) pairs.

    Returns nothing.
    """
    os.makedirs(source_dir, exist_ok=True)
    for file_name, text in file_texts:
        with open(os.path.join(source_dir, file_name), "w", encoding="utf-8") as f:
            f.write(text)


def _doc_id_for(text):
    """
    Returns the doc_id of a row whose raw content is text.
    """
    return _make_web_source().make_doc_id("pdf", text.encode("utf-8"))


# ---------- get_iterator ----------


def test_get_iterator_for_local_dataset_skips_rows_consumed(isolated_data_dir):
    """
    Input: three local files and rows_consumed of 1.
    Output: the file names get_iterator yields.
    """
    source_dir = isolated_data_dir / "docs"
    _write_files(source_dir, [("c.txt", "c"), ("a.txt", "a"), ("b.txt", "b")])
    source = _make_local_source(source_dir)

    file_names = []
    for file_name in source.get_iterator(1):
        file_names.append(file_name)

    assert file_names == ["b.txt", "c.txt"]


# ---------- row_matches_filter ----------


def test_row_matches_filter_is_true_without_a_filter():
    """
    Input: a web row and no row_filter.
    Output: row_matches_filter's return value.
    """
    source = _make_web_source(row_filter=None)

    assert source.row_matches_filter({"language": "fr"}) is True


def test_row_matches_filter_checks_every_filter_column():
    """
    Input: a two-column row_filter and rows that match all, some or none of it.
    Output: row_matches_filter's return values.
    """
    source = _make_web_source(row_filter={"language": "en", "type": "legal"})

    assert source.row_matches_filter({"language": "en", "type": "legal"}) is True
    assert source.row_matches_filter({"language": "en", "type": "forms"}) is False
    assert source.row_matches_filter({"language": "fr", "type": "legal"}) is False
    assert source.row_matches_filter({}) is False


def test_row_matches_filter_ignores_the_filter_for_local_files(isolated_data_dir):
    """
    Input: a row_filter on a local dataset, whose rows are file names.
    Output: row_matches_filter's return value.
    """
    source = _make_local_source(isolated_data_dir)
    source.dataset_config.row_filter = {"language": "en"}

    assert source.row_matches_filter("a.txt") is True


# ---------- decode_raw_bytes ----------


def test_decode_raw_bytes_reads_local_file(isolated_data_dir):
    """
    Input: a local file name.
    Output: decode_raw_bytes's return value.
    """
    source_dir = isolated_data_dir / "docs"
    _write_files(source_dir, [("a.txt", "hello")])
    source = _make_local_source(source_dir)

    assert source.decode_raw_bytes("a.txt") == b"hello"


@pytest.mark.parametrize(
    "content_encoding, raw_value, expected_bytes",
    [
        ("base64", "aGVsbG8=", b"hello"),
        ("bytes", b"hello", b"hello"),
        ("text", "hello", b"hello"),
    ],
)
def test_decode_raw_bytes_decodes_web_row_per_content_encoding(content_encoding, raw_value, expected_bytes):
    """
    Input: a web row whose content uses content_encoding.
    Output: decode_raw_bytes's return value.
    """
    source = _make_web_source(content_encoding=content_encoding)

    assert source.decode_raw_bytes({"content": raw_value}) == expected_bytes


def test_decode_raw_bytes_returns_none_on_unknown_content_encoding(capsys):
    """
    Input: a web row with an unknown content_encoding.
    Output: decode_raw_bytes's return value and printed output.
    """
    source = _make_web_source(content_encoding="zip")

    assert source.decode_raw_bytes({"content": "x"}) is None
    assert "decode/download failed" in capsys.readouterr().out


def test_decode_raw_bytes_returns_none_when_download_fails(monkeypatch, capsys):
    """
    Input: a url row whose download raises URLError.
    Output: decode_raw_bytes's return value and printed output.
    """
    def fake_urlopen(request, timeout):
        raise raw_source_module.urllib.error.URLError("not found")

    monkeypatch.setattr(raw_source_module.urllib.request, "urlopen", fake_urlopen)
    source = _make_web_source(content_encoding="url")

    assert source.decode_raw_bytes({"content": "https://example.com/missing"}) is None
    assert "URLError" in capsys.readouterr().out


def test_decode_raw_bytes_downloads_url_content(monkeypatch):
    """
    Input: a web row whose content is a download link.
    Output: decode_raw_bytes's return value and the request it made.
    """
    requests_made = []

    def fake_urlopen(request, timeout):
        requests_made.append((request.full_url, request.get_header("User-agent"), timeout))
        return _FakeUrlResponse(b"docx bytes")

    monkeypatch.setattr(raw_source_module.urllib.request, "urlopen", fake_urlopen)
    source = _make_web_source(content_encoding="url")

    raw_bytes = source.decode_raw_bytes({"content": "https://example.com/a.docx"})

    assert raw_bytes == b"docx bytes"
    assert requests_made == [("https://example.com/a.docx", "RAG-Eval/1.0", 60)]


# ---------- save_raw_bytes ----------


def test_save_raw_bytes_writes_file_and_returns_its_path(isolated_data_dir):
    """
    Input: raw bytes for one doc_id.
    Output: save_raw_bytes's return value and the written file.
    """
    source = _make_web_source()

    file_path = source.save_raw_bytes("pdf_x", b"raw content")

    assert file_path == f"{DATASET_NAME}/data/raw/pdf_x.pdf"
    with open(file_path, "rb") as f:
        assert f.read() == b"raw content"


# ---------- fetch_raw_batch ----------


def test_fetch_raw_batch_keeps_row_order_when_downloads_finish_out_of_order(isolated_data_dir, monkeypatch):
    """
    Input: three url rows whose first download is the slowest.
    Output: the RawBatch returned by fetch_raw_batch, and the raw files.
    """
    delays = {"https://example.com/1": 0.3, "https://example.com/2": 0.0, "https://example.com/3": 0.1}
    bodies = {"https://example.com/1": b"alpha", "https://example.com/2": b"beta", "https://example.com/3": b"gamma"}

    def fake_urlopen(request, timeout):
        time.sleep(delays[request.full_url])
        return _FakeUrlResponse(bodies[request.full_url])

    monkeypatch.setattr(raw_source_module.urllib.request, "urlopen", fake_urlopen)
    source = _make_web_source(content_encoding="url")
    rows = [{"content": "https://example.com/1"}, {"content": "https://example.com/2"}, {"content": "https://example.com/3"}]

    raw_batch = source.fetch_raw_batch(iter(rows), 3, SummaryCounts())

    doc_ids = []
    for doc_id, source_path in raw_batch.raw_items:
        doc_ids.append(doc_id)
        assert os.path.isfile(source_path)
    assert doc_ids == [_doc_id_for("alpha"), _doc_id_for("beta"), _doc_id_for("gamma")]
    assert raw_batch.rows_read == 3
    assert raw_batch.source_exhausted is False


def test_fetch_raw_batch_downloads_in_parallel(isolated_data_dir, monkeypatch):
    """
    Input: eight url rows whose downloads each take 0.2 seconds.
    Output: the most downloads running at the same moment.
    """
    lock = threading.Lock()
    running = [0]
    most_running = [0]

    def fake_urlopen(request, timeout):
        with lock:
            running[0] += 1
            if running[0] > most_running[0]:
                most_running[0] = running[0]
        time.sleep(0.2)
        with lock:
            running[0] -= 1
        return _FakeUrlResponse(request.full_url.encode("utf-8"))

    monkeypatch.setattr(raw_source_module.urllib.request, "urlopen", fake_urlopen)
    source = _make_web_source(content_encoding="url")
    rows = []
    for index in range(8):
        rows.append({"content": f"https://example.com/{index}"})

    source.fetch_raw_batch(iter(rows), 8, SummaryCounts())

    assert most_running[0] > 1


def test_fetch_raw_batch_counts_filtered_rows_and_decode_errors(isolated_data_dir, monkeypatch):
    """
    Input: rows where one is filtered out and one fails to download,
    asking for 3 matching rows from a source that has only 2.
    Output: the RawBatch and the SummaryCounts.
    """
    def fake_urlopen(request, timeout):
        if request.full_url == "https://example.com/bad":
            raise raw_source_module.urllib.error.URLError("not found")
        return _FakeUrlResponse(b"alpha")

    monkeypatch.setattr(raw_source_module.urllib.request, "urlopen", fake_urlopen)
    source = _make_web_source(content_encoding="url", row_filter={"language": "en"})
    rows = [
        {"content": "https://example.com/good", "language": "en"},
        {"content": "https://example.com/fr", "language": "fr"},
        {"content": "https://example.com/bad", "language": "en"},
    ]
    summary_counts = SummaryCounts()

    raw_batch = source.fetch_raw_batch(iter(rows), 3, summary_counts)

    assert len(raw_batch.raw_items) == 1
    assert raw_batch.rows_read == 3
    assert raw_batch.source_exhausted is True
    assert summary_counts.filtered == 1
    assert summary_counts.decode_error == 1


# ---------- make_doc_id ----------


def test_make_doc_id_is_doc_type_plus_16_hex_characters_of_the_raw_bytes_hash():
    """
    Input: the same raw bytes twice with doc_type "pdf", and different bytes.
    Output: make_doc_id's return values.
    """
    source = _make_web_source()

    first = source.make_doc_id("pdf", b"hello world")
    second = source.make_doc_id("pdf", b"hello world")
    other = source.make_doc_id("pdf", b"hello there")

    assert first == "pdf_b94d27b9934d3e08"
    assert first == second
    assert first != other
