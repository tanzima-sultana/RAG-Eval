"""
test_cache.py

Tests for stages.evaluation.evaluation.cache: the run settings and their hash, and saving and loading the
EvalResults file the evaluation stage hands to the analysis stage.
"""

import argparse
import os

import pytest

from stages.evaluation.evaluation import cache
from stages.evaluation.evaluation.schema import AnswerEvaluation
from stages.evaluation.evaluation.schema import EvalResults
from stages.evaluation.evaluation.schema import Evaluation
from stages.evaluation.evaluation.schema import RetrievalEvaluation
from stages.evaluation.generation.schema import Generated
from stages.evaluation.prompt.prompt_template import SYSTEM_PROMPT
from stages.evaluation.retrieval.schema import Retrieved
from stages.question.qna.schema import EvalQnA


def make_args(top_k=20, n_compare_runs=5, n_worst_answers=10):
    """
    Input: top_k, n_compare_runs and n_worst_answers.
    Output: an argparse.Namespace like the parsed eval arguments, with those three settings.
    What it does: builds the args get_eval_settings reads.
    """
    return argparse.Namespace(top_k=top_k, n_compare_runs=n_compare_runs, n_worst_answers=n_worst_answers)


def make_eval_results(settings):
    """
    Input: settings, the run's settings.
    Output: an EvalResults with one Retrieved, its RetrievalEvaluation, answer, answer evaluation and eval question.
    What it does: builds a small EvalResults with every nested field filled in.
    """
    retrieved = Retrieved(
        id="eval_qna_1", query_type="original", chunk_type="fixed", retrieval_type="dense", index_type="faiss_flatip",
        top_k=2, reranking=False, rerank_top_k=0, chunk_ids=["doc1_0", "doc1_1"], chunk_scores=[0.9, 0.8],
        doc_ids=["doc1", "doc1"],
    )
    retrieval_evaluation = RetrievalEvaluation(
        k=2,
        chunk_evaluation=Evaluation(hit=1, precision=0.5, recall=1.0, mrr=1.0, ndcg=1.0, first_relevant_rank=1),
        doc_evaluation=Evaluation(hit=1, precision=1.0, recall=1.0, mrr=1.0, ndcg=1.0, first_relevant_rank=1),
    )
    generated = Generated(id="eval_qna_1", qus="Question?", answer="Answer.", llm="ollama", llm_model="llama")
    answer_evaluation = AnswerEvaluation(
        i_dont_know=0, correctness=0.7, faithfulness=None, relevance=0.9, ans_eval_framework="ragas", llm="openai",
        llm_model="judge",
    )
    eval_qna_item = EvalQnA(
        id="eval_qna_1", qus="Question?", ans="Reference.", context=["passage"], qus_type="single_hop",
        doc_ids=["doc1"], doc_paths=["owner/dataset#1"], llm="claude", llm_model="sonnet",
    )
    eval_results = EvalResults(
        settings=settings,
        settings_hash=cache.get_settings_hash(settings),
        retrieval_evaluations=[(retrieved, retrieval_evaluation)],
        retrieved_list=[retrieved],
        retrieval_evaluation_list=[retrieval_evaluation],
        generated_list=[generated],
        answer_evaluations=[answer_evaluation],
        eval_qna=[eval_qna_item],
    )
    return eval_results


def test_get_eval_settings_adds_the_prompts_and_leaves_out_the_analysis_settings():
    """
    Input: args with top_k and the two analysis settings.
    Output: get_eval_settings' return value.
    What it does: checks the eval setting and the prompts are kept, and n_compare_runs and n_worst_answers are not.
    """
    settings = cache.get_eval_settings(make_args())

    assert settings["top_k"] == 20
    assert settings["system_prompt"] == SYSTEM_PROMPT
    assert "user_prompt_template" in settings
    assert "n_compare_runs" not in settings
    assert "n_worst_answers" not in settings


def test_settings_hash_ignores_the_analysis_settings_but_not_the_eval_settings():
    """
    Input: args that differ only in an analysis setting, and args that differ in top_k.
    Output: the hashes of their settings.
    What it does: checks the analysis can change its own settings and still find the eval results, while a
    different eval setting gives a different hash.
    """
    first_hash = cache.get_settings_hash(cache.get_eval_settings(make_args()))
    analysis_changed_hash = cache.get_settings_hash(cache.get_eval_settings(make_args(n_compare_runs=2, n_worst_answers=3)))
    eval_changed_hash = cache.get_settings_hash(cache.get_eval_settings(make_args(top_k=10)))

    assert analysis_changed_hash == first_hash
    assert eval_changed_hash != first_hash


def test_save_and_load_eval_results_round_trip(tmp_path):
    """
    Input: an EvalResults saved under tmp_path, then loaded by its settings hash.
    Output: the loaded EvalResults and the saved file.
    What it does: checks every list and nested dataclass comes back equal, and the file is named by the hash.
    """
    eval_results = make_eval_results({"top_k": 20})

    eval_results_file = cache.save_eval_results(str(tmp_path), eval_results)
    loaded_eval_results = cache.load_eval_results(str(tmp_path), eval_results.settings_hash)

    assert os.path.basename(eval_results_file) == f"{eval_results.settings_hash}.json"
    assert loaded_eval_results == eval_results
    loaded_retrieved, loaded_retrieval_evaluation = loaded_eval_results.retrieval_evaluations[0]
    assert type(loaded_retrieved) is Retrieved
    assert type(loaded_retrieval_evaluation.chunk_evaluation) is Evaluation
    assert loaded_eval_results.answer_evaluations[0].faithfulness is None


def test_save_eval_results_overwrites_a_run_with_the_same_settings(tmp_path):
    """
    Input: two EvalResults with the same settings but different answers, saved one after the other.
    Output: the folder's files and the loaded EvalResults.
    What it does: checks there is one file and it holds the second run's answers.
    """
    first_eval_results = make_eval_results({"top_k": 20})
    cache.save_eval_results(str(tmp_path), first_eval_results)
    second_eval_results = make_eval_results({"top_k": 20})
    second_eval_results.generated_list[0].answer = "Another answer."
    cache.save_eval_results(str(tmp_path), second_eval_results)

    loaded_eval_results = cache.load_eval_results(str(tmp_path), second_eval_results.settings_hash)

    assert os.listdir(tmp_path / "eval_results") == [f"{second_eval_results.settings_hash}.json"]
    assert loaded_eval_results.generated_list[0].answer == "Another answer."


def test_load_eval_results_raises_when_no_run_has_these_settings(tmp_path):
    """
    Input: an output folder with no saved eval results.
    Output: the raised error.
    What it does: checks a FileNotFoundError tells to run eval.sh with the same settings.
    """
    with pytest.raises(FileNotFoundError, match="Run scripts/eval.sh with the same settings"):
        cache.load_eval_results(str(tmp_path), "no_such_hash")
