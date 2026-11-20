"""
test_eval_dataset.py

Tests for evaluation.eval_dataset. No LLM is called: the cached path never builds one, and a cache miss without
ALLOW_PAID_API stops before any call.
"""

import hashlib
import json
import os
import types

import pytest

from common import paths
from common.args import OllamaArgs
from common.args import EvalDatasetArgs
from evaluation.eval_dataset import eval_dataset as eval_dataset_module
from evaluation.eval_dataset import eval_sample as eval_sample_module
from evaluation.eval_dataset.cache import load_eval_dataset
from evaluation.eval_dataset.cache import save_eval_dataset
from evaluation.eval_dataset.eval_dataset import EvalDataset
from evaluation.eval_dataset.eval_sample import generate_eval_dataset
from evaluation.eval_dataset.knowledge_graph import ReplyDidNotFinishError
from evaluation.eval_dataset.knowledge_graph import check_reply_finished
from rag.build.ingestion import save_documents
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
    field_values["output_path"] = "outputs/"
    field_values["dataset_name"] = "test-dataset"
    field_values["dataset_size"] = 1
    field_values["documents_path"] = "missing/documents/doc_ids_test.json"
    field_values["eval_dataset_llm_args"] = LLMArgs("anthropic/claude-sonnet-5-5", 100, 30)
    field_values["eval_dataset_kg_llm_args"] = LLMArgs("anthropic/claude-haiku-5-5", 100, 30)
    field_values["eval_dataset_embedding_model"] = "fake-model"
    field_values["n_sample_documents"] = 1
    field_values["n_evals_requested"] = 2
    field_values["sample_documents_seed"] = 42
    field_values["min_document_tokens"] = 500
    field_values["passage_match_chars"] = 80
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
    ({"n_evals_requested": 0}, "n_evals_requested"),
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


def save_complete_eval_dataset(output_path, eval_dataset_args):
    """
    Input: output_path; eval_dataset_args.
    Output: the saved eval samples.
    What it does: saves an eval dataset of two samples, an empty knowledge graph file and metadata with the eval
    dataset file's SHA-256, so the eval dataset is complete.
    """
    eval_samples = [make_eval_sample(0), make_eval_sample(1)]
    eval_dataset_path = paths.get_eval_dataset_path(output_path, eval_dataset_args, make_ollama_args())
    save_eval_dataset(eval_dataset_path, eval_samples)

    with open(paths.get_eval_dataset_kg_path(output_path, eval_dataset_args, make_ollama_args()), "w", encoding="utf-8") as f:
        f.write("{}")

    with open(eval_dataset_path, "rb") as f:
        content_hash = hashlib.sha256(f.read()).hexdigest()
    with open(paths.get_eval_dataset_metadata_path(output_path, eval_dataset_args, make_ollama_args()), "w", encoding="utf-8") as f:
        json.dump({"content_hash": content_hash}, f)

    return eval_samples


def save_document_list(output_path, make_document):
    """
    Input: output_path; pytest's make_document fixture.
    Output: the path of the saved document list.
    What it does: saves one long document and its document list doc_ids_test.json, as the build does.
    """
    documents = {}
    documents["doc1"] = make_document("doc1", "word " * 600)
    documents_folder = paths.get_documents_path(output_path)
    save_documents(documents_folder, documents, "test")
    return os.path.join(documents_folder, "doc_ids_test.json")


class GeneratePathReached(Exception):
    """
    Raised by the stand-in for generate_eval_dataset, so a test can see compute_eval_dataset went to the generate path.
    """


def stop_at_generate_path(monkeypatch):
    """
    Input: pytest's monkeypatch fixture.
    Output: nothing.
    What it does: makes generate_eval_dataset, called by compute_eval_dataset, raise GeneratePathReached.
    """
    def raise_generate_path_reached(documents, output_path, eval_dataset_args, ollama_args):
        """
        Input: the arguments of generate_eval_dataset.
        Output: none.
        What it does: raises GeneratePathReached.
        """
        raise GeneratePathReached()

    monkeypatch.setattr(eval_dataset_module, "generate_eval_dataset", raise_generate_path_reached)


def test_compute_eval_dataset_returns_saved_eval_dataset_without_documents_or_llm(tmp_path, monkeypatch, capsys):
    """
    Input: a complete eval dataset (metadata, eval dataset and knowledge graph files, matching SHA-256), and a
    documents_path with no document list.
    Output: the eval samples and what was printed.
    What it does: checks a complete eval dataset is returned, no documents are loaded (the missing list would raise)
    and no LLMProvider is built, so no key or ALLOW_PAID_API is needed.
    """
    eval_dataset_args = make_eval_dataset_args()
    output_path = str(tmp_path)
    eval_samples = save_complete_eval_dataset(output_path, eval_dataset_args)

    def fail_to_build_llm(llm_args, ollama_args):
        """
        Input: the arguments of LLMProvider.
        Output: none.
        What it does: fails the test if an LLM is built.
        """
        raise AssertionError("An LLM was built for a cached eval dataset")

    monkeypatch.setattr(eval_sample_module, "LLMProvider", fail_to_build_llm)
    monkeypatch.delenv("ALLOW_PAID_API", raising=False)

    loaded_eval_samples = EvalDataset(eval_dataset_args, make_ollama_args()).compute_eval_dataset(output_path)

    assert loaded_eval_samples == eval_samples
    assert "Eval dataset : 2 (all from cache)" in capsys.readouterr().out


@pytest.mark.parametrize("missing_or_changed", ["metadata", "eval_dataset", "knowledge_graph", "changed_content"])
def test_compute_eval_dataset_generates_when_the_eval_dataset_is_not_complete(tmp_path, monkeypatch, make_document, missing_or_changed):
    """
    Input: a complete eval dataset with its metadata, eval dataset or knowledge graph file removed, or its eval
    dataset file changed after the metadata was saved, and a saved document list.
    Output: GeneratePathReached.
    What it does: checks each case goes to the generate path instead of loading.
    """
    output_path = str(tmp_path)
    eval_dataset_args = make_eval_dataset_args(documents_path=save_document_list(output_path, make_document))
    save_complete_eval_dataset(output_path, eval_dataset_args)
    eval_dataset_path = paths.get_eval_dataset_path(output_path, eval_dataset_args, make_ollama_args())

    if missing_or_changed == "metadata":
        os.remove(paths.get_eval_dataset_metadata_path(output_path, eval_dataset_args, make_ollama_args()))
    elif missing_or_changed == "eval_dataset":
        os.remove(eval_dataset_path)
    elif missing_or_changed == "knowledge_graph":
        os.remove(paths.get_eval_dataset_kg_path(output_path, eval_dataset_args, make_ollama_args()))
    else:
        save_eval_dataset(eval_dataset_path, [make_eval_sample(5)])
    stop_at_generate_path(monkeypatch)

    with pytest.raises(GeneratePathReached):
        EvalDataset(eval_dataset_args, make_ollama_args()).compute_eval_dataset(output_path)


def test_compute_eval_dataset_without_saved_eval_dataset_stops_before_paid_call(tmp_path, monkeypatch, make_document):
    """
    Input: no saved eval dataset, a saved document list with one long document, and ALLOW_PAID_API not set.
    Output: ValueError, and the eval dataset folder.
    What it does: checks a cache miss loads the listed documents and never calls a paid LLM without the opt-in, and saves nothing.
    """
    monkeypatch.delenv("ALLOW_PAID_API", raising=False)
    output_path = str(tmp_path)
    eval_dataset_args = make_eval_dataset_args(documents_path=save_document_list(output_path, make_document))

    with pytest.raises(ValueError, match="ALLOW_PAID_API"):
        EvalDataset(eval_dataset_args, make_ollama_args()).compute_eval_dataset(output_path)

    assert not os.path.exists(os.path.join(output_path, "eval_dataset"))


# ---------- hash and paths ----------


def test_eval_dataset_hash_is_12_hex_characters_and_changes_with_any_arg():
    """
    Input: the default args, and the same args with one setting changed.
    Output: the two hashes.
    What it does: checks the hash is 12 hex characters, stays the same for the same args and changes for a changed arg.
    """
    eval_dataset_hash = paths.get_eval_dataset_hash(make_eval_dataset_args(), make_ollama_args())
    same_eval_dataset_hash = paths.get_eval_dataset_hash(make_eval_dataset_args(), make_ollama_args())
    changed_eval_dataset_hash = paths.get_eval_dataset_hash(make_eval_dataset_args(passage_match_chars=81), make_ollama_args())

    assert len(eval_dataset_hash) == 12
    int(eval_dataset_hash, 16)
    assert eval_dataset_hash == same_eval_dataset_hash
    assert eval_dataset_hash != changed_eval_dataset_hash


def test_eval_dataset_hash_uses_ollama_context_length_only_for_an_ollama_model():
    """
    Input: two Ollama context lengths, first with Claude models, then with an Ollama model.
    Output: the hashes.
    What it does: checks the context length changes the hash only when one of the LLMs is an Ollama model.
    """
    short_context = OllamaArgs(ollama_api_base="http://ollama-test:11434", ollama_context_length=4096)
    long_context = OllamaArgs(ollama_api_base="http://ollama-test:11434", ollama_context_length=32768)
    claude_args = make_eval_dataset_args()
    ollama_args_with_ollama_model = make_eval_dataset_args(eval_dataset_kg_llm_args=LLMArgs("ollama_chat/llama3.1:8b", 100, 30))

    assert paths.get_eval_dataset_hash(claude_args, short_context) == paths.get_eval_dataset_hash(claude_args, long_context)
    assert paths.get_eval_dataset_hash(ollama_args_with_ollama_model, short_context) != paths.get_eval_dataset_hash(ollama_args_with_ollama_model, long_context)


def test_eval_dataset_metadata_and_kg_paths_share_the_eval_dataset_hash(tmp_path):
    """
    Input: the default args.
    Output: the three paths.
    What it does: checks the eval dataset, metadata and knowledge graph files are named by the same hash.
    """
    eval_dataset_args = make_eval_dataset_args()
    output_path = str(tmp_path)
    eval_dataset_hash = paths.get_eval_dataset_hash(eval_dataset_args, make_ollama_args())

    assert paths.get_eval_dataset_path(output_path, eval_dataset_args, make_ollama_args()) == os.path.join(output_path, "eval_dataset", f"eval_dataset_{eval_dataset_hash}.json")
    assert paths.get_eval_dataset_metadata_path(output_path, eval_dataset_args, make_ollama_args()) == os.path.join(output_path, "eval_dataset", f"eval_dataset_{eval_dataset_hash}_metadata.json")
    assert paths.get_eval_dataset_kg_path(output_path, eval_dataset_args, make_ollama_args()) == os.path.join(output_path, "eval_dataset", f"kg_{eval_dataset_hash}.json")


# ---------- make_eval_dataset_metadata ----------


def test_make_eval_dataset_metadata_fills_every_field(tmp_path, make_document):
    """
    Input: a saved eval dataset of two samples and the one document it came from.
    Output: the EvalDatasetMetadata.
    What it does: checks the identity, integrity, source and generation fields.
    """
    eval_dataset_args = make_eval_dataset_args()
    output_path = str(tmp_path)
    eval_samples = [make_eval_sample(0), make_eval_sample(1)]
    eval_dataset_path = paths.get_eval_dataset_path(output_path, eval_dataset_args, make_ollama_args())
    save_eval_dataset(eval_dataset_path, eval_samples)
    documents = {}
    documents["doc1"] = make_document("doc1", "word " * 600)

    eval_dataset_metadata = EvalDataset(eval_dataset_args, make_ollama_args()).make_eval_dataset_metadata(documents, output_path, eval_samples)

    with open(eval_dataset_path, "rb") as f:
        expected_content_hash = hashlib.sha256(f.read()).hexdigest()
    assert eval_dataset_metadata.eval_dataset_hash == paths.get_eval_dataset_hash(eval_dataset_args, make_ollama_args())
    assert eval_dataset_metadata.eval_dataset_path == eval_dataset_path
    assert eval_dataset_metadata.knowledge_graph_path == paths.get_eval_dataset_kg_path(output_path, eval_dataset_args, make_ollama_args())
    assert eval_dataset_metadata.content_hash == expected_content_hash
    assert eval_dataset_metadata.n_evals_generated == 2
    assert eval_dataset_metadata.dataset_name == "test-dataset"
    assert eval_dataset_metadata.dataset_size == 1
    assert eval_dataset_metadata.documents_path == eval_dataset_args.documents_path
    assert eval_dataset_metadata.documents_hash == hashlib.sha256("doc1".encode("utf-8")).hexdigest()
    assert eval_dataset_metadata.ollama_context_length is None
    assert eval_dataset_metadata.sample_documents_seed == 42
    assert eval_dataset_metadata.min_document_tokens == 500
    assert eval_dataset_metadata.passage_match_chars == 80


# ---------- load_saved_eval_dataset ----------


def test_load_saved_eval_dataset_returns_saved_eval_dataset(tmp_path):
    """
    Input: a complete saved eval dataset.
    Output: the eval samples.
    What it does: checks the saved eval dataset is returned.
    """
    eval_dataset_args = make_eval_dataset_args()
    output_path = str(tmp_path)
    eval_samples = save_complete_eval_dataset(output_path, eval_dataset_args)

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


def test_load_saved_eval_dataset_raises_when_the_eval_dataset_changed(tmp_path):
    """
    Input: a complete saved eval dataset whose eval dataset file is changed after its metadata was saved.
    Output: ValueError.
    What it does: checks a changed eval dataset is not loaded.
    """
    eval_dataset_args = make_eval_dataset_args()
    output_path = str(tmp_path)
    save_complete_eval_dataset(output_path, eval_dataset_args)
    save_eval_dataset(paths.get_eval_dataset_path(output_path, eval_dataset_args, make_ollama_args()), [make_eval_sample(5)])

    with pytest.raises(ValueError, match="scripts/eval_dataset.sh"):
        EvalDataset(eval_dataset_args, make_ollama_args()).load_saved_eval_dataset(output_path)


def test_generate_eval_dataset_raises_when_too_few_documents_are_long_enough(tmp_path, make_document):
    """
    Input: one short document with n_sample_documents 1.
    Output: ValueError.
    What it does: checks RAGAS is only given documents long enough for it, before any LLM is built.
    """
    documents = {}
    documents["doc1"] = make_document("doc1", "too short")

    with pytest.raises(ValueError, match="n_sample_documents"):
        generate_eval_dataset(documents, str(tmp_path), make_eval_dataset_args(), make_ollama_args())


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
