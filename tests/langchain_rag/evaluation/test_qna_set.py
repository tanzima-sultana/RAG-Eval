"""
test_qna_set.py

Tests for evaluation.qna_set. No LLM is called: the cached path never builds one, and a cache miss without
ALLOW_PAID_API stops before any call.
"""

import os
import types

import pytest

from common import paths_temp
from common.args_temp import OllamaArgs
from common.args_temp import QnaArgs
from evaluation import qna_set as qna_set_module
from evaluation.qna_set import QnaSet
from evaluation.qna_set import ReplyDidNotFinishError
from evaluation.qna_set import check_reply_finished
from evaluation.qna_set import load_qna
from evaluation.qna_set import save_qna
from schema.llm import LLMArgs
from schema.qna import Qna

SAVED_QNA_PATH = "temp/langchain/pdfa-eng-wds/qna/qna_100_200_anthropic-claude-sonnet-5-5_all-MiniLM-L6-v2.json"


def make_qna_args(**overrides):
    """
    Input: any QnaArgs field as a keyword argument.
    Output: a QnaArgs with small defaults plus the overrides.
    What it does: fills in defaults for every field not overridden.
    """
    field_values = {}
    field_values["qna_llm_args"] = LLMArgs("anthropic/claude-sonnet-5-5", 100, 30)
    field_values["qna_kg_llm_args"] = LLMArgs("anthropic/claude-haiku-5-5", 100, 30)
    field_values["qna_embedding_model"] = "fake-model"
    field_values["n_sample_documents"] = 1
    field_values["n_sample_evals"] = 2
    for field_name in overrides:
        field_values[field_name] = overrides[field_name]
    return QnaArgs(**field_values)


def make_ollama_args():
    """
    Input: none.
    Output: an OllamaArgs with test values.
    What it does: builds the Ollama settings QnaSet takes. They are not used by the Claude models of these tests.
    """
    return OllamaArgs(ollama_api_base="http://ollama-test:11434", ollama_context_length=32768)


def make_qna_item(item_number):
    """
    Input: item_number.
    Output: a Qna with id "eval_qna_{item_number}".
    What it does: builds one QnA item with placeholder values.
    """
    return Qna(
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
    ({"qna_embedding_model": " "}, "qna_embedding_model"),
    ({"n_sample_documents": 0}, "n_sample_documents"),
    ({"n_sample_evals": 0}, "n_sample_evals"),
])
def test_validate_raises_for_invalid_setting(overrides, message):
    """
    Input: QnaArgs with one invalid setting.
    Output: ValueError.
    What it does: checks validate names the invalid setting.
    """
    with pytest.raises(ValueError, match=message):
        QnaSet(make_qna_args(**overrides), make_ollama_args()).validate()


# ---------- save_qna / load_qna ----------


def test_save_and_load_qna_round_trip(tmp_path):
    """
    Input: two QnA items saved and loaded back.
    Output: the loaded items.
    What it does: checks the items come back the same, and a missing file gives None.
    """
    qna_path = str(tmp_path / "qna" / "qna.json")
    qna_items = [make_qna_item(0), make_qna_item(1)]

    save_qna(qna_path, qna_items)

    assert load_qna(qna_path) == qna_items
    assert load_qna(str(tmp_path / "missing.json")) is None


# ---------- compute_qna ----------


def test_compute_qna_returns_saved_qna_without_building_an_llm(tmp_path, monkeypatch, capsys):
    """
    Input: a saved QnA file at the QnA path, and no documents.
    Output: the QnA items and what was printed.
    What it does: checks the saved QnA is returned and no LLMProvider is built, so no key or ALLOW_PAID_API is needed.
    """
    qna_args = make_qna_args()
    output_path = str(tmp_path)
    qna_items = [make_qna_item(0), make_qna_item(1)]
    save_qna(paths_temp.get_qna_path(output_path, qna_args), qna_items)

    def fail_to_build_llm(llm_args, ollama_args):
        """
        Input: the arguments of LLMProvider.
        Output: none.
        What it does: fails the test if an LLM is built.
        """
        raise AssertionError("An LLM was built for a cached QnA")

    monkeypatch.setattr(qna_set_module, "LLMProvider", fail_to_build_llm)
    monkeypatch.delenv("ALLOW_PAID_API", raising=False)

    loaded_qna_items = QnaSet(qna_args, make_ollama_args()).compute_qna({}, output_path)

    assert loaded_qna_items == qna_items
    assert "QnA : 2 (all from cache)" in capsys.readouterr().out


def test_compute_qna_without_saved_qna_stops_before_paid_call(tmp_path, monkeypatch, make_document):
    """
    Input: no saved QnA, one long document, and ALLOW_PAID_API not set.
    Output: ValueError, and the QnA folder.
    What it does: checks a cache miss never calls a paid LLM without the opt-in, and saves nothing.
    """
    monkeypatch.delenv("ALLOW_PAID_API", raising=False)
    documents = {}
    documents["doc1"] = make_document("doc1", "word " * 600)
    output_path = str(tmp_path)

    with pytest.raises(ValueError, match="ALLOW_PAID_API"):
        QnaSet(make_qna_args(), make_ollama_args()).compute_qna(documents, output_path)

    assert not os.path.exists(os.path.join(output_path, "qna"))


def test_generate_qna_raises_when_too_few_documents_are_long_enough(tmp_path, make_document):
    """
    Input: one short document with n_sample_documents 1.
    Output: ValueError.
    What it does: checks RAGAS is only given documents long enough for it, before any LLM is built.
    """
    documents = {}
    documents["doc1"] = make_document("doc1", "too short")

    with pytest.raises(ValueError, match="n_sample_documents"):
        QnaSet(make_qna_args(), make_ollama_args()).generate_qna(documents, str(tmp_path))


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


# ---------- the saved QnA set ----------


def test_saved_qna_set_loads_into_qna():
    """
    Input: the saved 200-item QnA set copied into temp/.
    Output: the loaded items.
    What it does: checks the saved set loads into the Qna schema unchanged. Skipped where the file is not present.
    """
    if not os.path.exists(SAVED_QNA_PATH):
        pytest.skip(f"{SAVED_QNA_PATH} is not present")

    qna_items = load_qna(SAVED_QNA_PATH)

    assert len(qna_items) == 200
    assert isinstance(qna_items[0], Qna)
    assert qna_items[0].qus != ""
    assert len(qna_items[0].doc_ids) > 0
