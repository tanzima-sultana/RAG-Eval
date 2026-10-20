"""
test_cache.py

Tests for stages.evaluation.retrieval.cache: saving a Retrieved list as one JSON file and loading it back.
"""

import json
import os

from stages.evaluation.retrieval import cache


def test_missing_file_returns_none(tmp_path):
    """
    Input: a path with no file.
    Output: None.
    What it does: checks a missing cache is reported as None, so the retrieval runs.
    """
    assert cache.load_cached_retrieved(str(tmp_path / "dense_top3.json")) is None


def test_save_then_load_gives_the_same_outputs_in_order(tmp_path, make_retrieved):
    """
    Input: three Retrieved, one with no chunks and one reranked, saved into a folder that does not exist yet.
    Output: the loaded list equals the saved one, in order, and no .tmp file is left.
    What it does: checks the folder is created, every field survives the round trip and the write goes through a
    temporary file.
    """
    retrieved_outputs = [
        make_retrieved(),
        make_retrieved(id="eval_qna_1", query_type="hyde", chunk_ids=[], chunk_scores=[]),
        make_retrieved(id="eval_qna_2", reranking=True, rerank_top_k=1, chunk_ids=["doc_c_4"], chunk_scores=[7.5]),
    ]
    retrieved_file = str(tmp_path / "retrieved" / "fixed" / "dense_top3.json")

    cache.save_retrieved(retrieved_file, retrieved_outputs)

    assert cache.load_cached_retrieved(retrieved_file) == retrieved_outputs
    assert not os.path.exists(retrieved_file + ".tmp")
    with open(retrieved_file, encoding="utf-8") as f:
        assert len(json.load(f)) == 3


def test_saving_again_replaces_the_file(tmp_path, make_retrieved):
    """
    Input: two saves to the same file.
    Output: the file holds only the second list.
    What it does: checks a new save replaces the old outputs instead of adding to them.
    """
    retrieved_file = str(tmp_path / "dense_top3.json")
    cache.save_retrieved(retrieved_file, [make_retrieved(), make_retrieved(id="eval_qna_1")])
    cache.save_retrieved(retrieved_file, [make_retrieved(id="eval_qna_9")])

    loaded_outputs = cache.load_cached_retrieved(retrieved_file)
    assert len(loaded_outputs) == 1
    assert loaded_outputs[0].id == "eval_qna_9"
