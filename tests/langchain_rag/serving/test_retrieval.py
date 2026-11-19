"""
test_retrieval.py

Tests for langchain_rag.serving.retrieval and paths_temp.get_retrieved_path. No model is loaded: the vector stores, the BM25
model and the cross-encoder are fakes that return fixed scores.
"""

import os
import types

import numpy as np
import pytest

from common import paths_temp
from common.args_temp import QueryArgs
from common.args_temp import RetrievalArgs
from common.constants import I_DONT_KNOW
from common.constants import RRF_K
from langchain_rag.serving import retrieval as retrieval_module
from langchain_rag.serving.retrieval import Retrieval
from langchain_rag.serving.retrieval import load_retrieved
from schema.chunk import Chunk
from schema.index import Bm25Index
from schema.index import FaissIndex
from schema.llm import LLMArgs
from schema.qna import Qna
from schema.query import Ans
from schema.query import Query
from schema.query import Qus
from schema.retrieved import Retrieved
from schema.vectordb import VectorDb


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
    field_values["query_types"] = ["original", "rewritten", "multi", "hyde"]
    field_values["query_split"] = True
    field_values["val_queries_size"] = 1
    field_values["test_queries_size"] = 1
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


def make_query(hyde_text="a passage"):
    """
    Input: hyde_text, the HyDE answer text.
    Output: a Query with id q1 and vectors 1.0 (original), 2.0 (rewritten), 3.0 and 4.0 (multi) and 5.0 (hyde).
    What it does: builds one query with all four query types.
    """
    return Query(
        id="q1",
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


def make_faiss_index(vector_store):
    """
    Input: vector_store, a FakeVectorStore.
    Output: a FaissIndex holding it.
    What it does: wraps the fake in the index record the run passes to retrieval.
    """
    return FaissIndex(index_type="faiss_flatip", vector_store=vector_store, index_path="unused", dimension=2)


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
        query_type=query_type,
        chunk_type="fixed",
        retrieval_type=retrieval_type,
        index_type=index_type,
        top_k=3,
        reranking=False,
        rerank_top_k=0,
        chunk_ids=chunk_ids,
        chunk_scores=chunk_scores,
        doc_ids=doc_ids,
    )


def find_retrieved(retrieved_items, query_type):
    """
    Input: retrieved_items; query_type.
    Output: the one Retrieved of that query type.
    What it does: finds a result by its query type, so a test can read it.
    """
    found_items = []
    for retrieved in retrieved_items:
        if retrieved.query_type == query_type:
            found_items.append(retrieved)
    assert len(found_items) == 1
    return found_items[0]


# ---------- validate ----------


@pytest.mark.parametrize("overrides, message", [
    ({"retrieval_types": ["dense", "keyword"]}, "retrieval_type"),
    ({"top_k": 0}, "top_k"),
    ({"rerank_top_k": 0}, "rerank_top_k"),
    ({"rerank_top_k": 4}, "must not be more than top_k"),
])
def test_validate_raises_for_invalid_setting(make_retrieval_args, overrides, message):
    """
    Input: RetrievalArgs with one invalid setting.
    Output: ValueError.
    What it does: checks validate names the invalid setting.
    """
    with pytest.raises(ValueError, match=message):
        Retrieval(make_retrieval_args(**overrides), make_query_args(), 1)


# ---------- get_retrieved_path ----------


def test_get_retrieved_path_names_the_index_split_and_query_types(make_retrieval_args):
    """
    Input: the default settings, a hybrid index type, 150 val queries and two query types.
    Output: the retrieved folder.
    What it does: checks each index settings folder, the "+" of hybrid, the val count and the query types are in the
    name.
    """
    query_args = make_query_args(query_types=["original", "hyde"])

    retrieved_path = paths_temp.get_retrieved_path("out", make_retrieval_args(), query_args, 150, "faiss_hnsw+bm25")

    assert retrieved_path == os.path.join(
        "out", "retrieved", "fixed", "fake-model", "2_6_2_0.5", "hnsw_4_8+bm25_1.5_0.75_whitespace",
        "val_150", "original_hyde"
    )


# ---------- compute_retrieval / compute_vectordb_retrieval ----------


@pytest.mark.parametrize("retrieval_types, index_kind, expected_count", [
    (["dense"], "bm25", 0),
    (["bm25"], "faiss", 0),
    (["hybrid"], "bm25", 3),
    (["hybrid"], "faiss", 4),
])
def test_compute_retrieval_runs_only_what_the_retrieval_types_need(make_retrieval_args, tmp_path, retrieval_types, index_kind, expected_count):
    """
    Input: retrieval_types and one BM25 or FAISS index.
    Output: the number of results.
    What it does: checks a retrieval type not requested gives [], and hybrid alone still runs dense and BM25.
    """
    retrieval = Retrieval(make_retrieval_args(retrieval_types=retrieval_types), make_query_args(), 1)

    if index_kind == "bm25":
        retrieved_items = retrieval.compute_retrieval(make_bm25_index(), "bm25", [make_query()], str(tmp_path))
    else:
        index = make_faiss_index(FakeVectorStore(make_dense_results()))
        retrieved_items = retrieval.compute_retrieval(index, "faiss_flatip", [make_query()], str(tmp_path))

    assert len(retrieved_items) == expected_count


def test_compute_vectordb_retrieval_searches_the_collection(make_retrieval_args, tmp_path):
    """
    Input: one fake Qdrant collection.
    Output: the dense results.
    What it does: checks the collection is searched as dense with index type qdrant.
    """
    vectordb = VectorDb(vectordb_type="qdrant", vector_store=FakeVectorStore(make_dense_results()), collection_name="c", dimension=2)

    retrieved_items = Retrieval(make_retrieval_args(), make_query_args(), 1).compute_vectordb_retrieval(vectordb, [make_query()], str(tmp_path))

    assert len(retrieved_items) == 4
    assert retrieved_items[0].retrieval_type == "dense"
    assert retrieved_items[0].index_type == "qdrant"


# ---------- dense_retrieval ----------


def test_dense_retrieval_keeps_top_k_and_merges_multi_by_best_score(make_retrieval_args, tmp_path):
    """
    Input: one query with all four query types.
    Output: the dense results.
    What it does: checks each query type's chunks, scores and doc ids, and that multi keeps doc1_1 once with its best score.
    """
    retrieved_items = Retrieval(make_retrieval_args(), make_query_args(), 1).dense_retrieval(
        FakeVectorStore(make_dense_results()), "faiss_flatip", [make_query()], str(tmp_path)
    )

    original = find_retrieved(retrieved_items, "original")
    assert original.chunk_ids == ["doc1_0", "doc1_1", "doc2_0"]
    assert original.doc_ids == ["doc1", "doc1", "doc2"]
    assert original.chunk_type == "fixed"
    assert original.reranking is False
    multi = find_retrieved(retrieved_items, "multi")
    assert multi.chunk_ids == ["doc1_1", "doc2_1", "doc1_0"]
    assert multi.chunk_scores == pytest.approx([0.95, 0.4, 0.3])
    assert find_retrieved(retrieved_items, "hyde").chunk_ids == ["doc2_1"]


def test_dense_retrieval_does_not_search_a_hyde_i_dont_know(make_retrieval_args, tmp_path):
    """
    Input: a query whose HyDE answer is I_DONT_KNOW.
    Output: the dense results and the searched vectors.
    What it does: checks the HyDE vector is not searched but still gets an empty result, so it counts as a miss.
    """
    vector_store = FakeVectorStore(make_dense_results())

    retrieved_items = Retrieval(make_retrieval_args(), make_query_args(), 1).dense_retrieval(
        vector_store, "faiss_flatip", [make_query(hyde_text=I_DONT_KNOW)], str(tmp_path)
    )

    assert find_retrieved(retrieved_items, "hyde").chunk_ids == []
    searched_first_values = []
    for searched_vector in vector_store.searched_vectors:
        searched_first_values.append(searched_vector[0])
    assert 5.0 not in searched_first_values


def test_dense_retrieval_searches_only_the_query_types_the_query_holds(make_retrieval_args, tmp_path):
    """
    Input: a query whose rewritten and multi query types are None.
    Output: the dense results.
    What it does: checks only original and hyde are searched.
    """
    query = make_query()
    query.rewritten_qus = None
    query.multi_qus = None

    retrieved_items = Retrieval(make_retrieval_args(), make_query_args(), 1).dense_retrieval(
        FakeVectorStore(make_dense_results()), "faiss_flatip", [query], str(tmp_path)
    )

    query_types = []
    for retrieved in retrieved_items:
        query_types.append(retrieved.query_type)
    assert query_types == ["original", "hyde"]


def test_dense_retrieval_loads_saved_results_without_searching(make_retrieval_args, tmp_path):
    """
    Input: the same dense retrieval run twice.
    Output: the second run's results and the searches it made.
    What it does: checks the results are saved and the second run loads them without searching.
    """
    retrieval = Retrieval(make_retrieval_args(), make_query_args(), 1)
    first_items = retrieval.dense_retrieval(FakeVectorStore(make_dense_results()), "faiss_flatip", [make_query()], str(tmp_path))

    second_vector_store = FakeVectorStore(make_dense_results())
    second_items = retrieval.dense_retrieval(second_vector_store, "faiss_flatip", [make_query()], str(tmp_path))

    assert second_vector_store.searched_vectors == []
    assert second_items == first_items


# ---------- bm25_retrieval ----------


def test_bm25_retrieval_drops_zero_scores_and_has_no_hyde(make_retrieval_args, tmp_path):
    """
    Input: one query with all four query types.
    Output: the BM25 results.
    What it does: checks chunks with score 0 are dropped, multi is merged and HyDE has no result.
    """
    retrieved_items = Retrieval(make_retrieval_args(), make_query_args(), 1).bm25_retrieval(make_bm25_index(), [make_query()], str(tmp_path))

    query_types = []
    for retrieved in retrieved_items:
        query_types.append(retrieved.query_type)
    assert query_types == ["original", "rewritten", "multi"]
    assert find_retrieved(retrieved_items, "original").chunk_ids == ["doc1_1", "doc2_1", "doc2_0"]
    assert find_retrieved(retrieved_items, "rewritten").chunk_ids == ["doc1_0"]
    multi = find_retrieved(retrieved_items, "multi")
    assert multi.chunk_ids == ["doc2_1", "doc1_0"]
    assert multi.chunk_scores == pytest.approx([4.0, 1.0])
    assert multi.index_type == "bm25"


# ---------- hybrid_retrieval ----------


def test_hybrid_retrieval_fuses_ranks_for_each_dense_index(make_retrieval_args, tmp_path):
    """
    Input: dense results of two indexes and BM25 results; hyde has no BM25 result.
    Output: the hybrid results and the saved files.
    What it does: checks reciprocal rank fusion, the "+" index type, that hyde gets no hybrid result, and one saved
    file per dense index.
    """
    retrieved_items = [
        make_retrieved("original", "dense", "faiss_flatip", ["doc1_0", "doc1_1"]),
        make_retrieved("hyde", "dense", "faiss_flatip", ["doc2_0"]),
        make_retrieved("original", "dense", "qdrant", ["doc2_1"]),
        make_retrieved("original", "bm25", "bm25", ["doc1_1", "doc2_0"]),
    ]
    retrieval_args = make_retrieval_args()
    query_args = make_query_args()

    hybrid_items = Retrieval(retrieval_args, query_args, 1).hybrid_retrieval(retrieved_items, str(tmp_path))

    assert len(hybrid_items) == 2
    flatip_hybrid = hybrid_items[0]
    assert flatip_hybrid.index_type == "faiss_flatip+bm25"
    assert flatip_hybrid.retrieval_type == "hybrid"
    # doc1_1 is rank 2 in dense and rank 1 in BM25, so it beats doc1_0 (rank 1 in dense only)
    assert flatip_hybrid.chunk_ids == ["doc1_1", "doc1_0", "doc2_0"]
    assert flatip_hybrid.chunk_scores[0] == pytest.approx(1.0 / (RRF_K + 2) + 1.0 / (RRF_K + 1))
    assert hybrid_items[1].index_type == "qdrant+bm25"
    for hybrid_index_type in ["faiss_flatip+bm25", "qdrant+bm25"]:
        retrieved_path = paths_temp.get_retrieved_path(str(tmp_path), retrieval_args, query_args, 1, hybrid_index_type)
        assert load_retrieved(os.path.join(retrieved_path, "hybrid_top3.json")) is not None


# ---------- rerank_retrieval ----------


def test_rerank_retrieval_sorts_by_cross_encoder_score_against_the_eval_question(make_retrieval_args, fake_cross_encoder, tmp_path):
    """
    Input: a dense and a BM25 result whose chunk texts end with their cross-encoder score.
    Output: the reranked results and the questions the cross-encoder scored.
    What it does: checks each result is reranked on its own, keeps rerank_top_k chunks, is marked reranked, and every
    query type is scored against the eval question from the QnA set.
    """
    chunks = {
        "doc1_0": Chunk(id="doc1_0", text="chunk 0.1", size=2, type="fixed"),
        "doc1_1": Chunk(id="doc1_1", text="chunk 0.9", size=2, type="fixed"),
        "doc2_0": Chunk(id="doc2_0", text="chunk 0.5", size=2, type="fixed"),
    }
    retrieved_items = [
        make_retrieved("rewritten", "dense", "faiss_flatip", ["doc1_0", "doc1_1", "doc2_0"]),
        make_retrieved("original", "bm25", "bm25", ["doc2_0", "doc1_0"]),
    ]
    qna_items = [Qna(
        id="q1", qus="What is X?", ans="X.", context=["c"], qus_type="single_hop_specific_query_synthesizer",
        doc_ids=["doc1"], doc_paths=["doc1.pdf"], llm="anthropic", llm_model="claude-sonnet-5-5",
    )]

    reranked_items = Retrieval(make_retrieval_args(reranking=True), make_query_args(), 1).rerank_retrieval(
        retrieved_items, chunks, qna_items, str(tmp_path)
    )

    assert len(reranked_items) == 2
    assert reranked_items[0].chunk_ids == ["doc1_1", "doc2_0"]
    assert reranked_items[0].chunk_scores == pytest.approx([0.9, 0.5])
    assert reranked_items[0].reranking is True
    assert reranked_items[0].rerank_top_k == 2
    assert reranked_items[0].query_type == "rewritten"
    assert reranked_items[1].chunk_ids == ["doc2_0", "doc1_0"]
    assert set(fake_cross_encoder.scored_questions) == {"What is X?"}
    # The plain results are not changed
    assert retrieved_items[0].chunk_ids == ["doc1_0", "doc1_1", "doc2_0"]
