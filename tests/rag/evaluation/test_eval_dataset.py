"""
test_eval_dataset.py

Tests for evaluation.eval_dataset. No LLM is called: the cached path never builds one, and a cache miss without
ALLOW_PAID_API stops before any call.
"""

import os
import types

import pytest

from common import paths
from common.args import OllamaArgs
from common.args import EvalDatasetArgs
from evaluation import eval_dataset as eval_dataset_module
from evaluation.eval_dataset import EvalDataset
from evaluation.eval_dataset import ReplyDidNotFinishError
from evaluation.eval_dataset import check_reply_finished
from evaluation.eval_dataset import load_eval_dataset
from evaluation.eval_dataset import save_eval_dataset
from schema.llm import LLMArgs
from schema.eval_sample import EvalSample

SAVED_EVAL_DATASET_PATH = "temp/pdfa-eng-wds/eval_dataset/eval_dataset_100_200_anthropic-claude-sonnet-5-5_all-MiniLM-L6-v2.json"


def make_eval_dataset_args(**overrides):
    """
    Input: any EvalDatasetArgs field as a keyword argument.
    Output: an EvalDatasetArgs with small defaults plus the overrides.
    What it does: fills in defaults for every field not overridden.
    """
    field_values = {}
    field_values["eval_dataset_llm_args"] = LLMArgs("anthropic/claude-sonnet-5-5", 100, 30)
    field_values["eval_dataset_kg_llm_args"] = LLMArgs("anthropic/claude-haiku-5-5", 100, 30)
    field_values["eval_dataset_embedding_model"] = "fake-model"
    field_values["n_sample_documents"] = 1
    field_values["n_sample_evals"] = 2
    for field_name in overrides:
        field_values[field_name] = overrides[field_name]
    return EvalDatasetArgs(**field_values)


def make_ollama_args():
    """
    Input: none.
    Output: an OllamaArgs with test values.
    What it does: builds the Ollama settings EvalDataset takes. They are not used by the Claude models of these tests.
    """
    return OllamaArgs(ollama_api_base="http://ollama-test:11434", ollama_context_length=32768)


def make_eval_sample(item_number):
    """
    Input: item_number.
    Output: an EvalSample with id "eval_qna_{item_number}".
    What it does: builds one eval sample with placeholder values.
    """
    return EvalSample(
        id=f"eval_qna_{item_number}",
        qus=f"Question {item_number}?",
        ans=f"Answer {item_number}.",
        context=[f"Context {item_number}."],
        qus_type="single_hop_specific_query_synthesizer",
        doc_ids=["doc1"],
        doc_paths=["data/doc1.pdf"],
        llm="anthropic",
        llm_model="claude-sonnet-5-5",
    )


def make_finished_response(finish_reason):
    """
    Input: finish_reason, the stop reason of one reply.
    Output: a stand-in for a LangChain LLMResult with one reply.
    What it does: builds the fields check_reply_finished reads.
    """
    message = types.SimpleNamespace(response_metadata={}, usage_metadata={"output_tokens": 7})
    generation = types.SimpleNamespace(generation_info={"finish_reason": finish_reason}, message=message)
    return types.SimpleNamespace(generations=[[generation]])


# ---------- validate ----------


@pytest.mark.parametrize("overrides, message", [
    ({"eval_dataset_embedding_model": " "}, "eval_dataset_embedding_model"),
    ({"n_sample_documents": 0}, "n_sample_documents"),
    ({"n_sample_evals": 0}, "n_sample_evals"),
])
def test_validate_raises_for_invalid_setting(overrides, message):
    """
    Input: EvalDatasetArgs with one invalid setting.
    Output: ValueError.
    What it does: checks validate names the invalid setting.
    """
    with pytest.raises(ValueError, match=message):
        EvalDataset(make_eval_dataset_args(**overrides), make_ollama_args()).validate()


# ---------- save_eval_dataset / load_eval_dataset ----------


def test_save_and_load_eval_dataset_round_trip(tmp_path):
    """
    Input: two eval samples saved and loaded back.
    Output: the loaded items.
    What it does: checks the items come back the same, and a missing file gives None.
    """
    eval_dataset_path = str(tmp_path / "eval_dataset" / "eval_dataset.json")
    eval_samples = [make_eval_sample(0), make_eval_sample(1)]

    save_eval_dataset(eval_dataset_path, eval_samples)

    assert load_eval_dataset(eval_dataset_path) == eval_samples
    assert load_eval_dataset(str(tmp_path / "missing.json")) is None


# ---------- compute_eval_dataset ----------


def test_compute_eval_dataset_returns_saved_eval_dataset_without_building_an_llm(tmp_path, monkeypatch, capsys):
    """
    Input: a saved eval dataset file at the eval dataset path, and no documents.
    Output: the eval samples and what was printed.
    What it does: checks the saved eval dataset is returned and no LLMProvider is built, so no key or ALLOW_PAID_API is needed.
    """
    eval_dataset_args = make_eval_dataset_args()
    output_path = str(tmp_path)
    eval_samples = [make_eval_sample(0), make_eval_sample(1)]
    save_eval_dataset(paths.get_eval_dataset_path(output_path, eval_dataset_args), eval_samples)

    def fail_to_build_llm(llm_args, ollama_args):
        """
        Input: the arguments of LLMProvider.
        Output: none.
        What it does: fails the test if an LLM is built.
        """
        raise AssertionError("An LLM was built for a cached eval dataset")

    monkeypatch.setattr(eval_dataset_module, "LLMProvider", fail_to_build_llm)
    monkeypatch.delenv("ALLOW_PAID_API", raising=False)

    loaded_eval_samples = EvalDataset(eval_dataset_args, make_ollama_args()).compute_eval_dataset({}, output_path)

    assert loaded_eval_samples == eval_samples
    assert "Eval dataset : 2 (all from cache)" in capsys.readouterr().out


def test_compute_eval_dataset_without_saved_eval_dataset_stops_before_paid_call(tmp_path, monkeypatch, make_document):
    """
    Input: no saved eval dataset, one long document, and ALLOW_PAID_API not set.
    Output: ValueError, and the eval dataset folder.
    What it does: checks a cache miss never calls a paid LLM without the opt-in, and saves nothing.
    """
    monkeypatch.delenv("ALLOW_PAID_API", raising=False)
    documents = {}
    documents["doc1"] = make_document("doc1", "word " * 600)
    output_path = str(tmp_path)

    with pytest.raises(ValueError, match="ALLOW_PAID_API"):
        EvalDataset(make_eval_dataset_args(), make_ollama_args()).compute_eval_dataset(documents, output_path)

    assert not os.path.exists(os.path.join(output_path, "eval_dataset"))


# ---------- load_saved_eval_dataset ----------


def test_load_saved_eval_dataset_returns_saved_eval_dataset(tmp_path):
    """
    Input: a saved eval dataset file at the eval dataset path.
    Output: the eval samples.
    What it does: checks the saved eval dataset is returned.
    """
    eval_dataset_args = make_eval_dataset_args()
    output_path = str(tmp_path)
    eval_samples = [make_eval_sample(0), make_eval_sample(1)]
    save_eval_dataset(paths.get_eval_dataset_path(output_path, eval_dataset_args), eval_samples)

    loaded_eval_samples = EvalDataset(eval_dataset_args, make_ollama_args()).load_saved_eval_dataset(output_path)

    assert loaded_eval_samples == eval_samples


def test_load_saved_eval_dataset_raises_without_saved_eval_dataset(tmp_path, monkeypatch):
    """
    Input: no saved eval dataset, and ALLOW_PAID_API set.
    Output: ValueError, and the eval dataset folder.
    What it does: checks it never generates, even when paid calls are allowed, and saves nothing.
    """
    monkeypatch.setenv("ALLOW_PAID_API", "1")
    output_path = str(tmp_path)

    with pytest.raises(ValueError, match="scripts/eval_dataset.sh"):
        EvalDataset(make_eval_dataset_args(), make_ollama_args()).load_saved_eval_dataset(output_path)

    assert not os.path.exists(os.path.join(output_path, "eval_dataset"))


def test_generate_eval_dataset_raises_when_too_few_documents_are_long_enough(tmp_path, make_document):
    """
    Input: one short document with n_sample_documents 1.
    Output: ValueError.
    What it does: checks RAGAS is only given documents long enough for it, before any LLM is built.
    """
    documents = {}
    documents["doc1"] = make_document("doc1", "too short")

    with pytest.raises(ValueError, match="n_sample_documents"):
        EvalDataset(make_eval_dataset_args(), make_ollama_args()).generate_eval_dataset(documents, str(tmp_path))


# ---------- check_reply_finished ----------


@pytest.mark.parametrize("finish_reason", ["stop", "end_turn", "STOP", "eos_token"])
def test_check_reply_finished_accepts_a_finished_reply(finish_reason):
    """
    Input: a reply that finished normally.
    Output: True.
    What it does: checks a normal stop reason passes.
    """
    assert check_reply_finished(make_finished_response(finish_reason)) is True


def test_check_reply_finished_raises_for_a_cut_reply():
    """
    Input: a reply cut at max_tokens ("length").
    Output: ReplyDidNotFinishError.
    What it does: checks the error names the stop reason and the output tokens.
    """
    with pytest.raises(ReplyDidNotFinishError, match="stop reason length, output tokens 7"):
        check_reply_finished(make_finished_response("length"))


# ---------- the saved eval dataset ----------


def test_saved_eval_dataset_loads_into_eval_samples():
    """
    Input: the saved 200-item eval dataset copied into temp/.
    Output: the loaded items.
    What it does: checks the saved set loads into the EvalSample schema unchanged. Skipped where the file is not present.
    """
    if not os.path.exists(SAVED_EVAL_DATASET_PATH):
        pytest.skip(f"{SAVED_EVAL_DATASET_PATH} is not present")

    eval_samples = load_eval_dataset(SAVED_EVAL_DATASET_PATH)

    assert len(eval_samples) == 200
    assert isinstance(eval_samples[0], EvalSample)
    assert eval_samples[0].qus != ""
    assert len(eval_samples[0].doc_ids) > 0
