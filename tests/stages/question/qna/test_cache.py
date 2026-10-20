"""
test_cache.py

Tests for stages.question.qna.qna.save_eval_qna and load_cached_eval_qna: saving the eval QnA
list as one JSON file named by its settings and QnA model, and loading it back.
"""

import json
import os

from common import paths
from stages.question.qna.qna import load_cached_eval_qna
from stages.question.qna.qna import save_eval_qna


def test_missing_cache_returns_none(tmp_path, make_eval_qna_args, make_qna_llm_args):
    """
    Input: an output folder with no eval QnA file.
    Output: None.
    What it does: checks a missing cache is reported as None, so get_eval_qna generates it.
    """
    assert load_cached_eval_qna(str(tmp_path), make_eval_qna_args(), make_qna_llm_args()) is None


def test_save_then_load_gives_the_same_items(tmp_path, make_eval_qna_args, make_qna_llm_args, make_eval_qna_item):
    """
    Input: two EvalQnA items, one multi-hop with two documents.
    Output: the loaded list equals the saved one.
    What it does: checks every field survives the round trip, in order.
    """
    eval_qna = [
        make_eval_qna_item(),
        make_eval_qna_item(
            id="eval_qna_1",
            context=["<1-hop>\n\nFirst passage.", "<2-hop>\n\nSecond passage."],
            qus_type="multi_hop_abstract_query_synthesizer",
            doc_ids=["doc_a", "doc_b"],
            doc_paths=["a.pdf", "b.pdf"],
        ),
    ]
    save_eval_qna(str(tmp_path), make_eval_qna_args(), make_qna_llm_args(), eval_qna)

    loaded_eval_qna = load_cached_eval_qna(str(tmp_path), make_eval_qna_args(), make_qna_llm_args())

    assert loaded_eval_qna == eval_qna


def test_saved_file_is_named_by_settings_and_model(tmp_path, make_eval_qna_args, make_qna_llm_args, make_eval_qna_item):
    """
    Input: one EvalQnA saved with n_sample_documents 2, n_sample_evals 2 and claude-sonnet-5-5.
    Output: the file at paths.get_eval_qna_path holds it as a JSON list, and no .tmp file is left.
    What it does: checks the file name and that the write goes through a temporary file that is renamed.
    """
    eval_qna_args = make_eval_qna_args()
    save_eval_qna(str(tmp_path), eval_qna_args, make_qna_llm_args(), [make_eval_qna_item()])

    cache_file = paths.get_eval_qna_path(str(tmp_path), eval_qna_args, "claude-sonnet-5-5")
    assert os.path.basename(cache_file) == "eval_qna_2_2_claude-sonnet-5-5_all-MiniLM-L6-v2.json"
    with open(cache_file, encoding="utf-8") as f:
        saved_dicts = json.load(f)
    assert len(saved_dicts) == 1
    assert saved_dicts[0]["llm_model"] == "claude-sonnet-5-5"
    assert not os.path.exists(cache_file + ".tmp")


def test_cache_of_another_model_is_not_loaded(tmp_path, make_eval_qna_args, make_qna_llm_args, make_eval_qna_item):
    """
    Input: eval QnA saved for claude-sonnet-5-5, loaded for gpt-4.1-mini.
    Output: None.
    What it does: checks a cache made by one QnA model is never used for another.
    """
    save_eval_qna(str(tmp_path), make_eval_qna_args(), make_qna_llm_args(), [make_eval_qna_item()])

    other_llm_args = make_qna_llm_args(llm="chatgpt", llm_model="gpt-4.1-mini")
    assert load_cached_eval_qna(str(tmp_path), make_eval_qna_args(), other_llm_args) is None


def test_cache_of_other_settings_is_not_loaded(tmp_path, make_eval_qna_args, make_qna_llm_args, make_eval_qna_item):
    """
    Input: eval QnA saved for n_sample_evals 2, loaded for n_sample_evals 5.
    Output: None.
    What it does: checks changing the number of eval items never reuses an old cache.
    """
    save_eval_qna(str(tmp_path), make_eval_qna_args(), make_qna_llm_args(), [make_eval_qna_item()])

    assert load_cached_eval_qna(str(tmp_path), make_eval_qna_args(n_sample_evals=5), make_qna_llm_args()) is None
