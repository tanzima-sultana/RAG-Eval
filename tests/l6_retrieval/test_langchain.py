"""
test_langchain.py

Tests for l6_retrieval.langchain.Langchain: retrieval on the LangChain indexes. The vector store is a fake, and
the BM25 retriever holds a real in-memory BM25 model.
"""

from types import SimpleNamespace

import pytest
from langchain_core.documents import Document as LangchainDocument
from rank_bm25 import BM25Okapi

from common.constants import I_DONT_KNOW
from l6_retrieval.langchain import Langchain


class FakeVectorStore:
    """
    Stand-in for a LangChain FAISS or Qdrant vector store: returns set (Document, score) pairs.
    """

    def __init__(self, hits):
        """
        Input: hits, a list of (chunk_id, score) pairs every search returns.
        Output: nothing.
        What it does: stores the hits; records each search.
        """
        self.hits = hits
        self.searches = []

    def similarity_search_with_score_by_vector(self, embedding, k):
        """
        Input: embedding, the query vector as a list; k, the number of results.
        Output: at most k (Document, score) pairs, each Document holding its chunk_id in metadata.
        What it does: records the search and returns the set hits.
        """
        self.searches.append((embedding, k))
        documents_and_scores = []
        for chunk_id, score in self.hits[:k]:
            documents_and_scores.append((LangchainDocument(page_content="text", metadata={"chunk_id": chunk_id}), score))
        return documents_and_scores


def test_dense_outputs_search_each_embedding_and_read_chunk_ids(make_query):
    """
    Input: a vector store returning doc_a_0 (0.9) and doc_b_1 (0.5); a query with two multi phrasings and a
    refused HyDE answer; top_k 2.
    Output: original, rewritten, multi and hyde outputs; the store searched 1 + 1 + 2 times with k 2 and the
    vector as a list; hyde has no chunks.
    What it does: checks the LangChain dense path searches each query type's embeddings and merges the phrasings.
    """
    vector_store = FakeVectorStore([("doc_a_0", 0.9), ("doc_b_1", 0.5)])
    index = SimpleNamespace(vector_store=vector_store)
    query = make_query(multi=((1, 0, 0), (0, 1, 0)), multi_tokens=(("a",), ("b",)), hyde_text=I_DONT_KNOW)

    dense_outputs = Langchain(2, ["original", "rewritten", "multi", "hyde"]).get_dense_outputs([query], "fixed", "faiss_flatip", index)

    query_types = []
    for retrieved in dense_outputs:
        query_types.append(retrieved.query_type)
    assert query_types == ["original", "rewritten", "multi", "hyde"]
    assert dense_outputs[0].chunk_ids == ["doc_a_0", "doc_b_1"]
    assert dense_outputs[0].chunk_scores == pytest.approx([0.9, 0.5])
    assert dense_outputs[0].doc_ids == ["doc_a", "doc_b"]
    assert dense_outputs[2].chunk_ids == ["doc_a_0", "doc_b_1"]
    assert dense_outputs[3].chunk_ids == []
    assert len(vector_store.searches) == 4
    assert vector_store.searches[0][1] == 2
    assert isinstance(vector_store.searches[0][0], list)


def test_bm25_outputs_score_with_the_retrievers_model(make_query):
    """
    Input: a LangChain BM25 retriever over six chunks, and the tokens "refund policy", top_k 3.
    Output: original, rewritten and multi outputs; original holds doc_a_0 then doc_b_0, read from the
    retriever's documents by position, and no chunk with score 0.
    What it does: checks the LangChain BM25 path scores with the retriever's own BM25 model and maps positions
    to chunk ids.
    """
    texts = [
        "refund policy refund within thirty days", "shipping takes five days", "the refund needs a receipt",
        "office hours are nine to five", "parking is behind the building", "the cafe opens at eight",
    ]
    chunk_ids = ["doc_a_0", "doc_a_1", "doc_b_0", "doc_c_0", "doc_d_0", "doc_e_0"]
    documents = []
    tokenized_texts = []
    for position in range(len(texts)):
        documents.append(LangchainDocument(page_content=texts[position], metadata={"chunk_id": chunk_ids[position]}))
        tokenized_texts.append(texts[position].split())
    index = SimpleNamespace(retriever=SimpleNamespace(docs=documents, vectorizer=BM25Okapi(tokenized_texts)))

    bm25_outputs = Langchain(3, ["original", "rewritten", "multi", "hyde"]).get_bm25_outputs([make_query(original_tokens=("refund", "policy"))], "fixed", "bm25", index)

    query_types = []
    for retrieved in bm25_outputs:
        query_types.append(retrieved.query_type)
    assert query_types == ["original", "rewritten", "multi"]
    assert bm25_outputs[0].chunk_ids == ["doc_a_0", "doc_b_0"]
    assert bm25_outputs[0].retrieval_type == "bm25"
