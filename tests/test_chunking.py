import pickle

import pandas as pd
import pytest

import src.local.chunking as chunking


class FakeEmbeddingModel:
    """Deterministic stand-in for a SentenceTransformer, keyed by sentence text."""

    def __init__(self, vector_by_sentence):
        self._vector_by_sentence = vector_by_sentence

    def encode(self, sentences):
        return [self._vector_by_sentence[sentence] for sentence in sentences]


# ---- fixed_chunking ----------------------------------------------------


def test_fixed_chunking_windows_with_overlap():
    text = "abcdefghij"  # 10 chars

    chunks = chunking.fixed_chunking(text, max_chunk_size=4, fixed_chunk_overlap=1)

    assert chunks == ["abcd", "defg", "ghij"]


def test_fixed_chunking_raises_when_overlap_not_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunking.fixed_chunking("abcdef", max_chunk_size=4, fixed_chunk_overlap=4)


def test_fixed_chunking_empty_text_returns_no_chunks():
    assert chunking.fixed_chunking("   ", max_chunk_size=10, fixed_chunk_overlap=2) == []


# ---- sentence_aware_chunking --------------------------------------------


def test_sentence_aware_chunking_packs_without_splitting_sentences():
    s1 = "Short sentence one."
    s2 = "Short sentence two."
    s3 = "Short sentence three."
    text = " ".join([s1, s2, s3])

    chunks = chunking.sentence_aware_chunking(text, max_chunk_size=len(s1) + 1 + len(s2))

    assert chunks == [f"{s1} {s2}", s3]


def test_sentence_aware_chunking_falls_back_to_fixed_split_for_oversized_sentence():
    long_sentence = "x" * 50 + "."

    chunks = chunking.sentence_aware_chunking(long_sentence, max_chunk_size=20)

    assert len(chunks) > 1
    assert all(len(chunk) <= 20 for chunk in chunks)
    assert "".join(chunks) == long_sentence


def test_sentence_aware_chunking_empty_text_returns_no_chunks():
    assert chunking.sentence_aware_chunking("   ", max_chunk_size=10) == []


# ---- semantic_chunking ---------------------------------------------------


def test_semantic_chunking_breaks_on_low_similarity(monkeypatch):
    sentence_a1 = "Topic A first sentence."
    sentence_a2 = "Topic A second sentence."
    sentence_b1 = "Topic B first sentence."
    sentence_b2 = "Topic B second sentence."
    text = " ".join([sentence_a1, sentence_a2, sentence_b1, sentence_b2])

    vectors = {
        sentence_a1: [1.0, 0.0],
        sentence_a2: [1.0, 0.0],
        sentence_b1: [0.0, 1.0],
        sentence_b2: [0.0, 1.0],
    }
    monkeypatch.setattr(chunking, "_get_embedding_model", lambda: FakeEmbeddingModel(vectors))

    chunks = chunking.semantic_chunking(text, max_chunk_size=1000, semantic_threshold=0.5)

    assert chunks == [
        f"{sentence_a1} {sentence_a2}",
        f"{sentence_b1} {sentence_b2}",
    ]


def test_semantic_chunking_respects_max_chunk_size_even_with_high_similarity(monkeypatch):
    sentence_a = "Alpha sentence number one here."
    sentence_b = "Alpha sentence number two here."
    text = f"{sentence_a} {sentence_b}"

    vectors = {sentence_a: [1.0, 0.0], sentence_b: [1.0, 0.0]}
    monkeypatch.setattr(chunking, "_get_embedding_model", lambda: FakeEmbeddingModel(vectors))

    chunks = chunking.semantic_chunking(text, max_chunk_size=len(sentence_a), semantic_threshold=0.0)

    assert chunks == [sentence_a, sentence_b]


def test_semantic_chunking_single_sentence_skips_model_loading(monkeypatch):
    def _fail():
        raise AssertionError("embedding model should not be loaded for a single sentence")

    monkeypatch.setattr(chunking, "_get_embedding_model", _fail)

    chunks = chunking.semantic_chunking("Only one sentence here.", max_chunk_size=1000, semantic_threshold=0.5)

    assert chunks == ["Only one sentence here."]


def test_semantic_chunking_empty_text_returns_no_chunks(monkeypatch):
    def _fail():
        raise AssertionError("embedding model should not be loaded for empty text")

    monkeypatch.setattr(chunking, "_get_embedding_model", _fail)

    assert chunking.semantic_chunking("   ", max_chunk_size=1000, semantic_threshold=0.5) == []


# ---- chunk_dataset ---------------------------------------------------------


def test_chunk_dataset_writes_one_pickled_map_per_strategy(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "text": ["First document sentence.", "Second document sentence."],
        }
    )
    dataset_path = tmp_path / "processed.parquet"
    df.to_parquet(dataset_path)

    monkeypatch.setattr(chunking, "CHUNKED_DATA_PATH", str(tmp_path / "chunks"))

    output_paths = chunking.chunk_dataset(
        str(dataset_path),
        max_chunk_size=100,
        fixed_chunk_overlap=10,
        semantic_threshold=0.5,
    )

    assert set(output_paths.keys()) == {"fixed", "sentence", "semantic"}
    for strategy, path in output_paths.items():
        assert path == str(tmp_path / f"chunks_{strategy}.pkl")
        with open(path, "rb") as f:
            chunk_map = pickle.load(f)

        assert isinstance(chunk_map, dict)
        assert chunk_map
        for chunk_id, text in chunk_map.items():
            doc_id = chunk_id.rsplit("_", 1)[0]
            assert doc_id in {"1", "2"}
            assert isinstance(text, str)


def test_chunk_dataset_prints_avg_size_and_file_size_stats(tmp_path, monkeypatch, capsys):
    df = pd.DataFrame({"id": [1], "text": ["A single short document."]})
    dataset_path = tmp_path / "processed.parquet"
    df.to_parquet(dataset_path)

    monkeypatch.setattr(chunking, "CHUNKED_DATA_PATH", str(tmp_path / "chunks"))

    chunking.chunk_dataset(
        str(dataset_path),
        max_chunk_size=100,
        fixed_chunk_overlap=10,
        semantic_threshold=0.5,
    )

    output = capsys.readouterr().out
    for strategy in ("fixed", "sentence", "semantic"):
        assert f"[{strategy}]" in output
        assert "avg chunk size" in output
        assert "file size" in output
