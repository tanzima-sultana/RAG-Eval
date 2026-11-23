"""
test_retrieval.py

Tests for rag.serving.retrieval (retrieval, single_retrieval and cache) and paths.get_retrieved_file_path. No model is loaded: the vector stores,
the BM25 model and the cross-encoder are fakes that return fixed scores. Each result is saved as its own file in tmp_path.
"""

import os
import types

import numpy as np
import pytest

from common import paths
from common.args import QueryArgs
from common.args import RetrievalArgs
from common.constants import I_DONT_KNOW
from common.constants import RRF_K
from rag.serving.retrieval import retrieval as retrieval_module
from rag.serving.retrieval.cache import load_retrieved_entry
from rag.serving.retrieval.cache import save_retrieved_entry
from rag.serving.retrieval.retrieval import Retrieval
from rag.serving.retrieval.single_retrieval import bm25_search
from rag.serving.retrieval.single_retrieval import dense_search
from rag.serving.retrieval.single_retrieval import hybrid_search
from rag.serving.retrieval.single_retrieval import rerank_result
from schema.setup import Setup
from schema.chunk import Document
from schema.index import Bm25Index
from schema.llm import LLMArgs
from schema.query import Ans
from schema.query import Query
from schema.query import Qus
from schema.retrieved import Retrieved


class FakeVectorStore:
    """
    Stand-in for a LangChain FAISS or Qdrant vector store. results_by_first_value maps the first number of a query
    vector to the (chunk_id, score) pairs it finds, best first. Every search is recorded in searched_vectors.
    """

    def __init__(self, results_by_first_value):
        """
        Input: results_by_first_value, a dict of float to a list of (chunk_id, score).
        Output: nothing.
        What it does: stores the results and starts an empty search record.
        """
        self.results_by_first_value = results_by_first_value
        self.searched_vectors = []

    def similarity_search_with_score_by_vector(self, embedding, k):
        """
        Input: embedding, the query vector as a list; k, the number of chunks to return.
        Output: up to k (Document-like object, score) pairs.
        What it does: records the search and returns the stored results of the vector's first number.
        """
        self.searched_vectors.append(embedding)
        found = []
        for chunk_id, score in self.results_by_first_value[embedding[0]][:k]:
            document = types.SimpleNamespace(metadata={"chunk_id": chunk_id})
            found.append((document, score))
        return found


class FakeBm25Model:
    """
    Stand-in for the rank_bm25 model of a BM25Retriever. scores_by_first_token maps the first token of a query to
    one score per chunk position.
    """

    def __init__(self, scores_by_first_token):
        """
        Input: scores_by_first_token, a dict of token to a list of scores.
        Output: nothing.
        What it does: stores the scores.
        """
        self.scores_by_first_token = scores_by_first_token

    def get_scores(self, query_tokens):
        """
        Input: query_tokens, the tokens of one query.
        Output: a numpy array with one score per chunk position.
        What it does: returns the stored scores of the first token.
        """
        return np.array(self.scores_by_first_token[query_tokens[0]], dtype=np.float64)


class FakeCrossEncoder:
    """
    Stand-in for sentence_transformers.CrossEncoder. A (question, chunk text) pair scores the number in the chunk
    text. Every scored question is recorded in scored_questions.
    """

    scored_questions = []

    def __init__(self, model_name, device=None):
        """
        Input: model_name and device, as the real class takes them.
        Output: nothing.
        What it does: nothing; the fake needs no model.
        """
        return None

    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        """
        Input: pairs, a list of (question, chunk text); batch_size and show_progress_bar (ignored).
        Output: one score per pair.
        What it does: records each question and scores each pair by the number in the chunk text.
        """
        scores = []
        for question, chunk_text in pairs:
            FakeCrossEncoder.scored_questions.append(question)
            scores.append(float(chunk_text.split()[-1]))
        return np.array(scores)


@pytest.fixture
def fake_cross_encoder(monkeypatch):
    """
    Input: pytest's monkeypatch fixture.
    Output: the FakeCrossEncoder class.
    What it does: makes Retrieval load the fake cross-encoder.
    """
    monkeypatch.setattr(retrieval_module, "CrossEncoder", FakeCrossEncoder)
    FakeCrossEncoder.scored_questions = []
    return FakeCrossEncoder


@pytest.fixture
def make_retrieval_args(make_index_args):
    """
    Input: the make_index_args fixture.
    Output: a function that builds a RetrievalArgs.
    What it does: returns a builder with small defaults (top_k 3, rerank_top_k 2) where any field can be overridden.
    """
    def build_retrieval_args(**overrides):
        """
        Input: any RetrievalArgs field as a keyword argument.
        Output: a RetrievalArgs with the defaults plus the overrides.
        What it does: fills in defaults for every field not overridden. The index type is empty, as in the run.
        """
        field_values = {}
        field_values["index_args"] = make_index_args(index_type="")
        field_values["retrieval_types"] = ["dense", "bm25", "hybrid"]
        field_values["top_k"] = 3
        field_values["reranking"] = False
        field_values["rerank_top_k"] = 2
        for field_name in overrides:
            field_values[field_name] = overrides[field_name]
        return RetrievalArgs(**field_values)

    return build_retrieval_args


def make_query_args(**overrides):
    """
    Input: any QueryArgs field as a keyword argument.
    Output: a QueryArgs with small defaults plus the overrides.
    What it does: fills in defaults for every field not overridden.
    """
    field_values = {}
    field_values["query_llm_args"] = LLMArgs("ollama_chat/llama3.1:8b", 100, 30)
    field_values["embedding_model"] = "fake-model"
    field_values["n_multi_qus"] = 2
    field_values["query_types"] = ["rewritten", "multi", "hyde"]
    field_values["query_split"] = True
    field_values["val_queries_size"] = 1
    field_values["test_queries_size"] = 1
    field_values["ollama_api_base"] = "http://ollama-test:11434"
    field_values["ollama_context_length"] = 32768
    for field_name in overrides:
        field_values[field_name] = overrides[field_name]
    return QueryArgs(**field_values)


def make_qus(first_value, first_token):
    """
    Input: first_value, the first number of the vector; first_token, the first BM25 token.
    Output: a Qus whose vector and tokens start with those values.
    What it does: builds one searchable question for the fakes.
    """
    return Qus(qus=f"question {first_token}", tokens=[first_token, "x"], embeddings=np.array([first_value, 0.0], dtype=np.float32))


def make_query(hyde_text="a passage", query_id="q1"):
    """
    Input: hyde_text, the HyDE answer text; query_id, the id of the query.
    Output: a Query with that id (q1 by default) and vectors 1.0 (original), 2.0 (rewritten), 3.0 and 4.0 (multi) and 5.0 (hyde).
    What it does: builds one query with all four query types.
    """
    return Query(
        id=query_id,
        original_qus=make_qus(1.0, "original"),
        rewritten_qus=make_qus(2.0, "rewritten"),
        multi_qus=[make_qus(3.0, "multi_a"), make_qus(4.0, "multi_b")],
        hyde_ans=Ans(ans=hyde_text, embeddings=np.array([5.0, 0.0], dtype=np.float32)),
        normalized=True,
        embedding_model="fake-model",
        llm="ollama_chat",
        llm_model="llama3.1:8b",
    )


def make_dense_results():
    """
    Input: none.
    Output: the (chunk_id, score) pairs each query vector finds in the fake vector store.
    What it does: gives multi two phrasings that share doc1_1, so the merge can be checked.
    """
    return {
        1.0: [("doc1_0", 0.9), ("doc1_1", 0.8), ("doc2_0", 0.7)],
        2.0: [("doc2_0", 0.6)],
        3.0: [("doc1_1", 0.5), ("doc2_1", 0.4)],
        4.0: [("doc1_1", 0.95), ("doc1_0", 0.3)],
        5.0: [("doc2_1", 0.2)],
    }


def make_bm25_index():
    """
    Input: none.
    Output: a Bm25Index over 4 chunks with fixed scores per query.
    What it does: gives original a chunk with score 0, which must be dropped.
    """
    chunk_ids = ["doc1_0", "doc1_1", "doc2_0", "doc2_1"]
    documents = []
    for chunk_id in chunk_ids:
        documents.append(types.SimpleNamespace(metadata={"chunk_id": chunk_id}))
    scores_by_first_token = {
        "original": [0.0, 3.0, 1.0, 2.0],
        "rewritten": [5.0, 0.0, 0.0, 0.0],
        "multi_a": [1.0, 0.0, 0.0, 0.0],
        "multi_b": [0.0, 0.0, 0.0, 4.0],
    }
    retriever = types.SimpleNamespace(docs=documents, vectorizer=FakeBm25Model(scores_by_first_token))
    return Bm25Index(retriever=retriever, index_path="unused")


def make_retrieved(query_type, retrieval_type, index_type, chunk_ids):
    """
    Input: query_type; retrieval_type; index_type; chunk_ids, best first.
    Output: a plain Retrieved for query q1 with falling scores.
    What it does: builds one plain result for the hybrid and rerank tests.
    """
    chunk_scores = []
    doc_ids = []
    for position in range(len(chunk_ids)):
        chunk_scores.append(1.0 - position * 0.1)
        doc_ids.append(chunk_ids[position].rsplit("_", 1)[0])
    return Retrieved(
        id="q1",
        setup=Setup(chunk_type="fixed", index_type=index_type, query_type=query_type, retrieval_type=retrieval_type, reranking=False),
        top_k=3,
        rerank_top_k=0,
        chunk_ids=chunk_ids,
        chunk_scores=chunk_scores,
        doc_ids=doc_ids,
    )


def make_chunks():
    """
    Input: none.
    Output: a dict of chunk_id to Document whose texts end with their cross-encoder score.
    What it does: gives the chunks the rerank tests read.
    """
    chunks = {}
    chunks["doc1_0"] = Document(id="doc1_0", text="chunk 0.1", size=2, type="fixed")
    chunks["doc1_1"] = Document(id="doc1_1", text="chunk 0.9", size=2, type="fixed")
    chunks["doc2_0"] = Document(id="doc2_0", text="chunk 0.5", size=2, type="fixed")
    chunks["doc2_1"] = Document(id="doc2_1", text="chunk 0.7", size=2, type="fixed")
    return chunks


def make_retrieval(retrieval_args, tmp_path, query_type="original", retrieval_type="dense", reranking=False, vector_store=None, bm25_index=None, chunks=None, index_type="faiss_flatip"):
    """
    Input: retrieval_args; tmp_path, the folder the cache files go in; and the setup: query_type, retrieval_type,
    reranking, vector_store, bm25_index, chunks and index_type.
    Output: a Retrieval for that setup. The index objects the retrieval type needs default to the fakes.
    What it does: fills in the fake index objects a test does not give, so each test names only what it checks.
    """
    if vector_store is None:
        vector_store = FakeVectorStore(make_dense_results())
    if bm25_index is None:
        bm25_index = make_bm25_index()
    if retrieval_type == "bm25":
        vector_store = None
        index_type = None
    if retrieval_type == "dense":
        bm25_index = None
    return Retrieval(retrieval_args, make_query_args(), query_type, retrieval_type, reranking, index_type, vector_store, bm25_index, chunks, str(tmp_path))


# ---------- __init__ and validate ----------


@pytest.mark.parametrize("overrides, message", [
    ({"retrieval_types": ["dense", "keyword"]}, "retrieval_type"),
    ({"top_k": 0}, "top_k"),
    ({"rerank_top_k": 0}, "rerank_top_k"),
    ({"rerank_top_k": 4}, "must not be more than top_k"),
])
def test_constructor_raises_for_invalid_setting(make_retrieval_args, tmp_path, overrides, message):
    """
    Input: RetrievalArgs with one invalid setting.
    Output: ValueError.
    What it does: checks the constructor validates the retrieval settings and names the invalid one.
    """
    with pytest.raises(ValueError, match=message):
        make_retrieval(make_retrieval_args(**overrides), tmp_path)


@pytest.mark.parametrize("setup, message", [
    ({"query_type": "keyword"}, "query_type"),
    ({"retrieval_type": "keyword"}, "retrieval_type"),
    ({"query_type": "hyde", "retrieval_type": "bm25"}, "no BM25 tokens"),
    ({"query_type": "hyde", "retrieval_type": "hybrid"}, "no BM25 tokens"),
    ({"reranking": True}, "Reranking needs chunks"),
])
def test_constructor_raises_for_invalid_setup(make_retrieval_args, fake_cross_encoder, tmp_path, setup, message):
    """
    Input: one invalid setup: an unknown query or retrieval type, HyDE with BM25 or hybrid, or reranking without chunks.
    Output: ValueError.
    What it does: checks the constructor rejects the setup and names what is wrong.
    """
    with pytest.raises(ValueError, match=message):
        make_retrieval(make_retrieval_args(), tmp_path, **setup)


@pytest.mark.parametrize("retrieval_type, missing, message", [
    ("dense", "vector_store", "Dense retrieval needs"),
    ("bm25", "bm25_index", "BM25 retrieval needs"),
    ("hybrid", "vector_store", "Hybrid retrieval needs"),
    ("hybrid", "bm25_index", "Hybrid retrieval needs"),
])
def test_constructor_raises_when_an_index_the_retrieval_type_needs_is_missing(make_retrieval_args, tmp_path, retrieval_type, missing, message):
    """
    Input: a retrieval type and one index object it needs left out.
    Output: ValueError.
    What it does: checks each retrieval type gets the index objects it searches.
    """
    vector_store = FakeVectorStore(make_dense_results())
    bm25_index = make_bm25_index()
    if missing == "vector_store":
        vector_store = None
    if missing == "bm25_index":
        bm25_index = None

    with pytest.raises(ValueError, match=message):
        Retrieval(make_retrieval_args(), make_query_args(), "original", retrieval_type, False, "faiss_flatip", vector_store, bm25_index, None, str(tmp_path))


def test_constructor_loads_the_cross_encoder_only_with_reranking(make_retrieval_args, fake_cross_encoder, tmp_path):
    """
    Input: the same setup with reranking off and on.
    Output: the cross-encoder of each.
    What it does: checks the cross-encoder is loaded once in the constructor, and only with reranking.
    """
    plain_retrieval = make_retrieval(make_retrieval_args(), tmp_path)
    reranked_retrieval = make_retrieval(make_retrieval_args(), tmp_path, reranking=True, chunks=make_chunks())

    assert plain_retrieval.cross_encoder is None
    assert isinstance(reranked_retrieval.cross_encoder, FakeCrossEncoder)


# ---------- get_retrieved_file_path ----------


def get_retrieved_file_name(retrieval_args, query_args=None, query_type="original", question="What is X?", index_type="faiss_flatip", retrieval_type="dense", reranking=False):
    """
    Input: the arguments of get_retrieved_file_path, with defaults; query_args defaults to make_query_args().
    Output: the file name of that setup's result for that question.
    What it does: builds the retrieved file path under "out" and keeps only its name.
    """
    if query_args is None:
        query_args = make_query_args()
    return os.path.basename(paths.get_retrieved_file_path("out", retrieval_args, query_args, query_type, question, index_type, retrieval_type, reranking))


def test_get_retrieved_file_path_is_a_12_character_hash_in_the_given_folder(make_retrieval_args):
    """
    Input: the default settings, twice.
    Output: the retrieved file path.
    What it does: checks the file is {hash}.json directly in the given folder, with a 12 hex character hash, the same
    for the same inputs.
    """
    retrieved_file = paths.get_retrieved_file_path("out", make_retrieval_args(), make_query_args(), "original", "What is X?", "faiss_flatip", "dense", False)
    same_retrieved_file = paths.get_retrieved_file_path("out", make_retrieval_args(), make_query_args(), "original", "What is X?", "faiss_flatip", "dense", False)

    file_name = os.path.basename(retrieved_file)
    assert os.path.dirname(retrieved_file) == "out"
    assert file_name.endswith(".json")
    assert len(file_name) == 12 + len(".json")
    int(file_name[:12], 16)
    assert retrieved_file == same_retrieved_file


def test_get_retrieved_file_path_changes_with_each_hashed_setting(make_retrieval_args, make_index_args, make_chunking_args):
    """
    Input: the default settings, and the same with one of them changed: the question, the query type, a query setting
    of that query type, the input hash, a chunk setting, an index setting, the index type, the retrieval type, top_k
    or reranking.
    Output: the file names.
    What it does: checks each changed setting gives a different file.
    """
    default_name = get_retrieved_file_name(make_retrieval_args(), query_type="rewritten")

    changed_names = [
        get_retrieved_file_name(make_retrieval_args(), query_type="rewritten", question="What is Y?"),
        get_retrieved_file_name(make_retrieval_args(), query_type="original"),
        get_retrieved_file_name(make_retrieval_args(), query_args=make_query_args(query_llm_args=LLMArgs("ollama_chat/other-model", 100, 30)), query_type="rewritten"),
        get_retrieved_file_name(make_retrieval_args(index_args=make_index_args(index_type="", input_hash="otherinputhash")), query_type="rewritten"),
        get_retrieved_file_name(make_retrieval_args(index_args=make_index_args(index_type="", chunking_args=make_chunking_args(max_chunk_size=8))), query_type="rewritten"),
        get_retrieved_file_name(make_retrieval_args(index_args=make_index_args(index_type="", hnsw_m=8)), query_type="rewritten"),
        get_retrieved_file_name(make_retrieval_args(), query_type="rewritten", index_type="qdrant"),
        get_retrieved_file_name(make_retrieval_args(), query_type="rewritten", retrieval_type="hybrid"),
        get_retrieved_file_name(make_retrieval_args(top_k=5), query_type="rewritten"),
        get_retrieved_file_name(make_retrieval_args(), query_type="rewritten", reranking=True),
    ]

    for changed_name in changed_names:
        assert changed_name != default_name


def test_get_retrieved_file_path_uses_rerank_top_k_only_with_reranking(make_retrieval_args):
    """
    Input: two rerank_top_k values, with reranking off and on.
    Output: the file names.
    What it does: checks rerank_top_k changes the file only of a reranked result.
    """
    assert get_retrieved_file_name(make_retrieval_args(rerank_top_k=1)) == get_retrieved_file_name(make_retrieval_args(rerank_top_k=2))
    assert get_retrieved_file_name(make_retrieval_args(rerank_top_k=1), reranking=True) != get_retrieved_file_name(make_retrieval_args(rerank_top_k=2), reranking=True)


def test_get_retrieved_file_path_does_not_use_the_query_types_list(make_retrieval_args):
    """
    Input: the same setup with two different query_types lists in the query settings.
    Output: the file names.
    What it does: checks the result of a query type is reused whatever other query types the run makes.
    """
    one_list = get_retrieved_file_name(make_retrieval_args(), query_args=make_query_args(query_types=[]))
    other_list = get_retrieved_file_name(make_retrieval_args(), query_args=make_query_args(query_types=["hyde"]))

    assert one_list == other_list


# ---------- save_retrieved_entry / load_retrieved_entry ----------


def test_save_and_load_retrieved_entry_round_trip(tmp_path):
    """
    Input: one Retrieved, saved and loaded back.
    Output: the loaded Retrieved.
    What it does: checks every field but the id comes back, the id is empty, and no temporary file is left.
    """
    retrieved = make_retrieved("multi", "hybrid", "faiss_flatip+bm25", ["doc1_0", "doc2_1"])
    retrieved_file = os.path.join(str(tmp_path), "retrieved", "entry.json")

    save_retrieved_entry(retrieved_file, "What is X?", retrieved)
    loaded = load_retrieved_entry(retrieved_file)

    assert loaded.id == ""
    assert loaded.setup == retrieved.setup
    assert loaded.top_k == retrieved.top_k
    assert loaded.rerank_top_k == retrieved.rerank_top_k
    assert loaded.chunk_ids == retrieved.chunk_ids
    assert loaded.chunk_scores == pytest.approx(retrieved.chunk_scores)
    assert loaded.doc_ids == retrieved.doc_ids
    assert not os.path.exists(retrieved_file + ".tmp")


def test_load_retrieved_entry_gives_none_when_the_file_is_missing(tmp_path):
    """
    Input: a path with no file.
    Output: None.
    What it does: checks a missing entry is reported as None.
    """
    assert load_retrieved_entry(os.path.join(str(tmp_path), "missing.json")) is None


# ---------- dense_search ----------


def test_dense_search_keeps_top_k_of_one_query_type(make_retrieval_args):
    """
    Input: one query, searched with original.
    Output: its dense Retrieved.
    What it does: checks the chunks, doc ids and setup of the one query type searched.
    """
    retrieved = dense_search(make_retrieval_args(), FakeVectorStore(make_dense_results()), "faiss_flatip", make_query(), "original")

    assert retrieved.id == "q1"
    assert retrieved.chunk_ids == ["doc1_0", "doc1_1", "doc2_0"]
    assert retrieved.doc_ids == ["doc1", "doc1", "doc2"]
    assert retrieved.setup == Setup(chunk_type="fixed", index_type="faiss_flatip", query_type="original", retrieval_type="dense", reranking=False)


def test_dense_search_merges_multi_by_best_score(make_retrieval_args):
    """
    Input: one query, searched with multi, whose two phrasings both find doc1_1.
    Output: its dense Retrieved.
    What it does: checks multi keeps doc1_1 once with its best score.
    """
    retrieved = dense_search(make_retrieval_args(), FakeVectorStore(make_dense_results()), "faiss_flatip", make_query(), "multi")

    assert retrieved.chunk_ids == ["doc1_1", "doc2_1", "doc1_0"]
    assert retrieved.chunk_scores == pytest.approx([0.95, 0.4, 0.3])


@pytest.mark.parametrize("query_type, first_value", [
    ("original", 1.0),
    ("rewritten", 2.0),
    ("hyde", 5.0),
])
def test_dense_search_searches_only_the_given_query_type(make_retrieval_args, query_type, first_value):
    """
    Input: one query, searched with one query type.
    Output: the vectors searched.
    What it does: checks only that query type's vector is searched.
    """
    vector_store = FakeVectorStore(make_dense_results())

    dense_search(make_retrieval_args(), vector_store, "faiss_flatip", make_query(), query_type)

    searched_first_values = []
    for searched_vector in vector_store.searched_vectors:
        searched_first_values.append(searched_vector[0])
    assert searched_first_values == [first_value]


def test_dense_search_does_not_search_a_hyde_i_dont_know(make_retrieval_args):
    """
    Input: a query whose HyDE answer is I_DONT_KNOW, searched with hyde.
    Output: its dense Retrieved and the searched vectors.
    What it does: checks nothing is searched but a Retrieved with no chunks is still given, so it counts as a miss.
    """
    vector_store = FakeVectorStore(make_dense_results())

    retrieved = dense_search(make_retrieval_args(), vector_store, "faiss_flatip", make_query(hyde_text=I_DONT_KNOW), "hyde")

    assert retrieved.chunk_ids == []
    assert vector_store.searched_vectors == []


# ---------- bm25_search ----------


def test_bm25_search_drops_zero_scores(make_retrieval_args):
    """
    Input: one query, searched with original, whose doc1_0 scores 0.
    Output: its BM25 Retrieved.
    What it does: checks the chunk with score 0 is dropped and the setup is BM25.
    """
    retrieved = bm25_search(make_retrieval_args(), make_bm25_index(), make_query(), "original")

    assert retrieved.chunk_ids == ["doc1_1", "doc2_1", "doc2_0"]
    assert retrieved.setup.index_type == "bm25"
    assert retrieved.setup.retrieval_type == "bm25"


def test_bm25_search_merges_multi_by_best_score(make_retrieval_args):
    """
    Input: one query, searched with multi.
    Output: its BM25 Retrieved.
    What it does: checks each phrasing's chunks are merged with their best score.
    """
    retrieved = bm25_search(make_retrieval_args(), make_bm25_index(), make_query(), "multi")

    assert retrieved.chunk_ids == ["doc2_1", "doc1_0"]
    assert retrieved.chunk_scores == pytest.approx([4.0, 1.0])


# ---------- hybrid_search ----------


def test_hybrid_search_fuses_the_dense_and_bm25_ranks(make_retrieval_args):
    """
    Input: one query, searched with original on a dense index and the BM25 index.
    Output: its hybrid Retrieved.
    What it does: checks reciprocal rank fusion, top_k and the "+" index type. Dense finds doc1_0, doc1_1, doc2_0 and
    BM25 finds doc1_1, doc2_1, doc2_0, so doc1_1 (ranks 2 and 1) beats doc2_0 (ranks 3 and 3), which beats doc1_0.
    """
    retrieved = hybrid_search(make_retrieval_args(), FakeVectorStore(make_dense_results()), "faiss_flatip", make_bm25_index(), make_query(), "original")

    assert retrieved.chunk_ids == ["doc1_1", "doc2_0", "doc1_0"]
    assert retrieved.chunk_scores[0] == pytest.approx(1.0 / (RRF_K + 2) + 1.0 / (RRF_K + 1))
    assert retrieved.setup.index_type == "faiss_flatip+bm25"
    assert retrieved.setup.retrieval_type == "hybrid"


# ---------- rerank_result ----------


def test_rerank_result_keeps_the_best_rerank_top_k(make_retrieval_args):
    """
    Input: one plain result whose chunk texts end with their cross-encoder score, and its question.
    Output: the reranked Retrieved.
    What it does: checks the chunks are sorted by score, rerank_top_k are kept and the question is the one given.
    """
    FakeCrossEncoder.scored_questions = []
    retrieved = make_retrieved("rewritten", "dense", "faiss_flatip", ["doc1_0", "doc1_1", "doc2_0"])

    reranked = rerank_result(make_retrieval_args(), FakeCrossEncoder("unused"), retrieved, "What is X?", make_chunks())

    assert reranked.chunk_ids == ["doc1_1", "doc2_0"]
    assert reranked.chunk_scores == pytest.approx([0.9, 0.5])
    assert reranked.doc_ids == ["doc1", "doc2"]
    assert reranked.rerank_top_k == 2
    assert reranked.setup.reranking is True
    assert reranked.setup.query_type == "rewritten"
    assert set(FakeCrossEncoder.scored_questions) == {"What is X?"}


# ---------- retrieve ----------


@pytest.mark.parametrize("query_type, missing_field", [
    ("rewritten", "rewritten_qus"),
    ("multi", "multi_qus"),
    ("hyde", "hyde_ans"),
])
def test_retrieve_raises_when_the_query_does_not_hold_the_query_type(make_retrieval_args, tmp_path, query_type, missing_field):
    """
    Input: a query whose query type of this setup is None.
    Output: ValueError.
    What it does: checks the input is rejected and names the missing query type.
    """
    query = make_query()
    setattr(query, missing_field, None)
    retrieval = make_retrieval(make_retrieval_args(), tmp_path, query_type=query_type)

    with pytest.raises(ValueError, match=query_type):
        retrieval.retrieve(query)


@pytest.mark.parametrize("retrieval_type, index_type, chunk_ids", [
    ("dense", "faiss_flatip", ["doc1_0", "doc1_1", "doc2_0"]),
    ("bm25", "bm25", ["doc1_1", "doc2_1", "doc2_0"]),
    ("hybrid", "faiss_flatip+bm25", ["doc1_1", "doc2_0", "doc1_0"]),
])
def test_retrieve_searches_with_the_retrieval_type_of_the_setup(make_retrieval_args, tmp_path, retrieval_type, index_type, chunk_ids):
    """
    Input: one setup of each retrieval type, nothing saved.
    Output: the Retrieved.
    What it does: checks retrieve gives the one result of its own retrieval type.
    """
    retrieval = make_retrieval(make_retrieval_args(), tmp_path, retrieval_type=retrieval_type)

    retrieved = retrieval.retrieve(make_query())

    assert retrieved.setup.retrieval_type == retrieval_type
    assert retrieved.setup.index_type == index_type
    assert retrieved.chunk_ids == chunk_ids


def test_retrieve_saves_the_result_and_loads_it_without_searching(make_retrieval_args, tmp_path, capsys):
    """
    Input: the same setup retrieved twice, the second time with a query of the same question and another id.
    Output: both results, the searches of the second run and what was printed.
    What it does: checks the first call searches and saves one file, and the second loads it without searching and
    gives it the second query's id.
    """
    first_retrieved = make_retrieval(make_retrieval_args(), tmp_path).retrieve(make_query())
    assert "made, 3 chunks" in capsys.readouterr().out
    assert len(os.listdir(str(tmp_path))) == 1

    second_vector_store = FakeVectorStore(make_dense_results())
    second_retrieved = make_retrieval(make_retrieval_args(), tmp_path, vector_store=second_vector_store).retrieve(make_query(query_id="q2"))

    assert second_vector_store.searched_vectors == []
    assert "loaded from cache, 3 chunks" in capsys.readouterr().out
    assert second_retrieved.id == "q2"
    assert second_retrieved.chunk_ids == first_retrieved.chunk_ids
    assert second_retrieved.setup == first_retrieved.setup


def test_retrieve_reranks_against_the_original_question(make_retrieval_args, fake_cross_encoder, tmp_path):
    """
    Input: a reranked setup searched with rewritten, nothing saved.
    Output: the Retrieved and the questions the cross-encoder scored.
    What it does: checks the result is reranked, keeps rerank_top_k chunks and is scored against the original
    question, not the rewritten one.
    """
    retrieval = make_retrieval(make_retrieval_args(), tmp_path, query_type="rewritten", reranking=True, chunks=make_chunks(), vector_store=FakeVectorStore({2.0: [("doc1_0", 0.6), ("doc1_1", 0.5), ("doc2_0", 0.4)]}))

    retrieved = retrieval.retrieve(make_query())

    assert retrieved.setup.reranking is True
    assert retrieved.setup.query_type == "rewritten"
    assert retrieved.chunk_ids == ["doc1_1", "doc2_0"]
    assert set(fake_cross_encoder.scored_questions) == {"question original"}
