"""
test_retrieval.py

Tests for l6_retrieval.retrieval.Retrieval: which retrievals run for which index, the cache file of each, the
LangChain hand-off and reranking. The FAISS and BM25 indexes are real and small; the cross-encoder is a fake.
"""

import dataclasses
import os

import pytest

from common import paths
from l6_retrieval import retrieval as retrieval_module
from l6_retrieval.retrievars import dense
from l6_retrieval.retrieval import Retrieval

CROSS_ENCODER_FILE_NAME = "cross-encoder-ms-marco-MiniLM-L-6-v2"


@pytest.fixture
def no_cross_encoder_model(monkeypatch, fake_cross_encoder):
    """
    Input: pytest's monkeypatch fixture and the FakeCrossEncoder class.
    Output: a list of the (model name, device) each CrossEncoder was created with.
    What it does: makes Retrieval create a FakeCrossEncoder, so no model is downloaded or loaded.
    """
    created = []

    def make_cross_encoder(model_name, device):
        """
        Input: model_name and device, as CrossEncoder takes them.
        Output: a FakeCrossEncoder scoring every text 0.
        What it does: records the model and builds the fake.
        """
        created.append((model_name, device))
        return fake_cross_encoder({})

    monkeypatch.setattr(retrieval_module, "CrossEncoder", make_cross_encoder)
    return created


@pytest.fixture
def make_retrieval(tmp_path, chunking_args, index_args, make_retrieval_args, no_cross_encoder_model):
    """
    Input: tmp_path and the args fixtures.
    Output: a function that builds a Retrieval writing its caches under tmp_path.
    What it does: returns a builder where any RetrievalArgs field can be overridden by keyword.
    """
    def build_retrieval(**overrides):
        """
        Input: any RetrievalArgs field as a keyword argument.
        Output: a Retrieval with 150 tuning queries and tmp_path as the output folder.
        What it does: builds the orchestrator the tests call.
        """
        return Retrieval(make_retrieval_args(**overrides), chunking_args, index_args, 150, str(tmp_path))

    return build_retrieval


@pytest.fixture
def small_indexes(make_faiss_index, make_bm25_index):
    """
    Input: the index builder fixtures.
    Output: a (FaissIndex, Bm25Index) pair over the same three chunks.
    What it does: gives the orchestrator tests one dense and one BM25 index.
    """
    chunk_ids = ["doc_a_0", "doc_b_0", "doc_c_0"]
    faiss_index = make_faiss_index(chunk_ids, [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    bm25_index = make_bm25_index(chunk_ids, ["refund policy", "shipping days", "office hours"])
    return faiss_index, bm25_index


def retrieved_folder(tmp_path, chunking_args, index_args, index_type):
    """
    Input: tmp_path, the chunking and index args, and an index type.
    Output: the retrieved folder of that index with 150 tuning queries.
    What it does: works out where Retrieval writes that index's caches.
    """
    index_type_args = dataclasses.replace(index_args, index_type=index_type)
    return paths.get_retrieved_path(str(tmp_path), chunking_args, index_type_args, 150)


def hybrid_folder_of(tmp_path, chunking_args, index_args, dense_index_type):
    """
    Input: tmp_path, the chunking and index args, and the dense index type of a hybrid pair.
    Output: the retrieved folder of that pair with 150 tuning queries.
    What it does: works out where Retrieval writes the pair's hybrid caches.
    """
    dense_index_args = dataclasses.replace(index_args, index_type=dense_index_type)
    bm25_index_args = dataclasses.replace(index_args, index_type="bm25")
    return paths.get_hybrid_retrieved_path(str(tmp_path), chunking_args, dense_index_args, bm25_index_args, 150)


# ------ Setup

def test_invalid_args_raise_when_created(tmp_path, chunking_args, index_args, make_retrieval_args):
    """
    Input: rerank_top_k above top_k.
    Output: ValueError when the Retrieval is created.
    What it does: checks the settings are validated before any search.
    """
    with pytest.raises(ValueError, match="rerank_top_k"):
        Retrieval(make_retrieval_args(top_k=2, rerank_top_k=5), chunking_args, index_args, 150, str(tmp_path))


def test_cross_encoder_is_loaded_only_with_reranking(make_retrieval, no_cross_encoder_model):
    """
    Input: one Retrieval without reranking, one with.
    Output: no cross-encoder for the first; the second loads CROSS_ENCODER_MODEL once.
    What it does: checks the model is only loaded when it is used.
    """
    assert make_retrieval(reranking=False).cross_encoder is None
    assert no_cross_encoder_model == []

    assert make_retrieval(reranking=True).cross_encoder is not None
    assert len(no_cross_encoder_model) == 1
    assert no_cross_encoder_model[0][0] == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_langchain_is_used_only_for_the_langchain_framework(make_retrieval):
    """
    Input: one built-in Retrieval and one with framework langchain.
    Output: no Langchain object for the first; one with top_k for the second.
    What it does: checks the framework decides which retrieval code runs.
    """
    assert make_retrieval(framework="").langchain is None
    assert make_retrieval(framework="langchain", top_k=3).langchain.top_k == 3


# ------ Which retrieval runs for which index

def test_dense_runs_on_faiss_and_qdrant_only(make_retrieval, make_query, small_indexes):
    """
    Input: dense retrieval asked for on a faiss_flatip index and on the bm25 index.
    Output: outputs for faiss_flatip, none for bm25.
    What it does: checks dense only searches vector indexes.
    """
    faiss_index, bm25_index = small_indexes
    retrieval = make_retrieval()

    assert len(retrieval.compute_dense_retrieval([make_query()], "fixed", "faiss_flatip", faiss_index)) == 4
    assert retrieval.compute_dense_retrieval([make_query()], "fixed", "bm25", bm25_index) == []


def test_bm25_runs_on_the_bm25_index_only(make_retrieval, make_query, small_indexes):
    """
    Input: BM25 retrieval asked for on the bm25 index and on a faiss_flatip index.
    Output: outputs for bm25, none for faiss_flatip.
    What it does: checks BM25 only searches the BM25 index.
    """
    faiss_index, bm25_index = small_indexes
    retrieval = make_retrieval()

    assert len(retrieval.compute_bm25_retrieval([make_query()], "fixed", "bm25", bm25_index)) == 3
    assert retrieval.compute_bm25_retrieval([make_query()], "fixed", "faiss_flatip", faiss_index) == []


def test_retrieval_types_not_asked_for_do_not_run(make_retrieval, make_query, small_indexes, make_retrieved):
    """
    Input: retrieval_types ["dense"] only.
    Output: BM25 and hybrid return no outputs, while dense still runs.
    What it does: checks --retrieval_types decides which retrievals run.
    """
    faiss_index, bm25_index = small_indexes
    retrieval = make_retrieval(retrieval_types=["dense"])

    assert len(retrieval.compute_dense_retrieval([make_query()], "fixed", "faiss_flatip", faiss_index)) == 4
    assert retrieval.compute_bm25_retrieval([make_query()], "fixed", "bm25", bm25_index) == []
    assert retrieval.compute_hybrid_retrieval("fixed", "faiss_flatip", [make_retrieved()], [make_retrieved()]) == []


# ------ Cache files

def test_dense_outputs_are_saved_then_loaded_without_searching(tmp_path, monkeypatch, make_retrieval, make_query, small_indexes, chunking_args, index_args):
    """
    Input: two dense retrievals on the same index.
    Output: the first writes dense_top3.json in the index's retrieved folder; the second returns the same outputs
    without searching.
    What it does: checks the dense cache file name and that a cached run never searches again.
    """
    faiss_index, bm25_index = small_indexes
    retrieval = make_retrieval()
    first_outputs = retrieval.compute_dense_retrieval([make_query()], "fixed", "faiss_flatip", faiss_index)

    dense_file = os.path.join(retrieved_folder(tmp_path, chunking_args, index_args, "faiss_flatip"), "dense_top3.json")
    assert os.path.exists(dense_file)

    def fail_search(*arguments):
        """
        Input: anything.
        Output: never returns.
        What it does: fails the test if the cached run searches.
        """
        raise AssertionError("a cached dense retrieval must not search")

    monkeypatch.setattr(dense, "get_dense_outputs", fail_search)
    assert retrieval.compute_dense_retrieval([make_query()], "fixed", "faiss_flatip", faiss_index) == first_outputs


def test_bm25_outputs_are_saved_in_the_bm25_folder(tmp_path, make_retrieval, make_query, small_indexes, chunking_args, index_args):
    """
    Input: one BM25 retrieval.
    Output: bm25_top3.json in the bm25 index's retrieved folder, holding the outputs.
    What it does: checks the BM25 cache file name and folder, which holds the BM25 settings.
    """
    faiss_index, bm25_index = small_indexes
    bm25_outputs = make_retrieval().compute_bm25_retrieval([make_query()], "fixed", "bm25", bm25_index)

    bm25_folder = retrieved_folder(tmp_path, chunking_args, index_args, "bm25")
    assert "bm25_1.5_0.75_whitespace" in bm25_folder
    assert retrieval_module.cache.load_cached_retrieved(os.path.join(bm25_folder, "bm25_top3.json")) == bm25_outputs


def test_hybrid_outputs_are_saved_in_the_pair_folder_and_reused(tmp_path, make_retrieval, make_retrieved, chunking_args, index_args):
    """
    Input: hybrid retrieval of one dense and one BM25 output, run twice, the second time with no inputs.
    Output: hybrid_top3.json in the "flatip+bm25_..." folder; the second run returns the saved outputs.
    What it does: checks the hybrid cache file is named by both indexes and reused.
    """
    retrieval = make_retrieval()
    dense_outputs = [make_retrieved(chunk_ids=["doc_a_0"], chunk_scores=[0.9])]
    bm25_outputs = [make_retrieved(retrieval_type="bm25", index_type="bm25", chunk_ids=["doc_b_0"], chunk_scores=[3.0])]

    first_outputs = retrieval.compute_hybrid_retrieval("fixed", "faiss_flatip", dense_outputs, bm25_outputs)

    hybrid_folder = hybrid_folder_of(tmp_path, chunking_args, index_args, "faiss_flatip")
    assert "flatip+bm25_1.5_0.75_whitespace" in hybrid_folder
    assert os.path.exists(os.path.join(hybrid_folder, "hybrid_top3.json"))
    assert retrieval.compute_hybrid_retrieval("fixed", "faiss_flatip", [], []) == first_outputs


def test_each_index_has_its_own_dense_cache(tmp_path, make_retrieval, make_query, make_qdrant_index, small_indexes, chunking_args, index_args):
    """
    Input: dense retrieval on faiss_flatip and then on qdrant.
    Output: two different files, and the qdrant outputs are not the faiss ones.
    What it does: checks one index's cache is never loaded for another.
    """
    faiss_index, bm25_index = small_indexes
    retrieval = make_retrieval()
    retrieval.compute_dense_retrieval([make_query()], "fixed", "faiss_flatip", faiss_index)
    qdrant_outputs = retrieval.compute_dense_retrieval([make_query()], "fixed", "qdrant", make_qdrant_index([("doc_z_9", 0.7)]))

    assert qdrant_outputs[0].chunk_ids == ["doc_z_9"]
    assert qdrant_outputs[0].index_type == "qdrant"
    assert os.path.exists(os.path.join(retrieved_folder(tmp_path, chunking_args, index_args, "qdrant"), "dense_top3.json"))


# ------ LangChain hand-off

def test_langchain_framework_hands_dense_to_langchain(make_retrieval, make_query, monkeypatch):
    """
    Input: framework langchain, and Langchain.get_dense_outputs replaced to return a marker.
    Output: compute_dense_retrieval returns that marker list.
    What it does: checks the LangChain indexes are searched by the Langchain class, not the built-in code.
    """
    retrieval = make_retrieval(framework="langchain")
    calls = []

    def fake_langchain_dense(queries, chunk_type, index_type, index):
        """
        Input: the arguments Retrieval passes.
        Output: an empty list.
        What it does: records the call.
        """
        calls.append(index_type)
        return []

    monkeypatch.setattr(retrieval.langchain, "get_dense_outputs", fake_langchain_dense)
    assert retrieval.compute_dense_retrieval([make_query()], "fixed", "faiss_hnsw", object()) == []
    assert calls == ["faiss_hnsw"]


# ------ Reranking

@pytest.mark.parametrize("retrieval_type", ["dense", "bm25", "hybrid"])
def test_reranked_outputs_are_saved_with_the_cross_encoder_in_the_name(tmp_path, make_retrieval, make_query, make_chunk, make_retrieved,
                                                                       fake_cross_encoder, chunking_args, index_args, retrieval_type):
    """
    Input: reranking on, rerank_top_k 1, and one output with two chunks the cross-encoder scores 1 and 9.
    Output: one reranked output keeping the chunk scored 9, saved as
    {retrieval_type}_top3_rerank1_cross-encoder-ms-marco-MiniLM-L-6-v2.json; a second run returns it from the file.
    What it does: checks each reranked cache file is named by top_k, rerank_top_k and the cross-encoder, and reused.
    """
    retrieval = make_retrieval(reranking=True, rerank_top_k=1)
    retrieval.cross_encoder = fake_cross_encoder({"low": 1.0, "high": 9.0})
    chunks = {"doc_a_0": make_chunk("doc_a_0", "low"), "doc_b_0": make_chunk("doc_b_0", "high")}
    outputs = [make_retrieved(retrieval_type=retrieval_type, chunk_ids=["doc_a_0", "doc_b_0"], chunk_scores=[0.9, 0.1])]

    if retrieval_type == "dense":
        reranked_outputs = retrieval.get_reranked_dense_retrieval([make_query()], chunks, "faiss_flatip", outputs)
        folder = retrieved_folder(tmp_path, chunking_args, index_args, "faiss_flatip")
        again = retrieval.get_reranked_dense_retrieval([make_query()], {}, "faiss_flatip", [])
    elif retrieval_type == "bm25":
        reranked_outputs = retrieval.get_reranked_bm25_retrieval([make_query()], chunks, outputs)
        folder = retrieved_folder(tmp_path, chunking_args, index_args, "bm25")
        again = retrieval.get_reranked_bm25_retrieval([make_query()], {}, [])
    else:
        reranked_outputs = retrieval.get_reranked_hybrid_retrieval([make_query()], chunks, "faiss_flatip", outputs)
        folder = hybrid_folder_of(tmp_path, chunking_args, index_args, "faiss_flatip")
        again = retrieval.get_reranked_hybrid_retrieval([make_query()], {}, "faiss_flatip", [])

    assert reranked_outputs[0].chunk_ids == ["doc_b_0"]
    assert reranked_outputs[0].reranking is True
    reranked_file = os.path.join(folder, f"{retrieval_type}_top3_rerank1_{CROSS_ENCODER_FILE_NAME}.json")
    assert os.path.exists(reranked_file)
    assert again == reranked_outputs
