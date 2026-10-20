"""
conftest.py

Shared fixtures for the l6_retrieval tests.

FAISS and BM25 are real, small, in-memory indexes built from a few chunks with hand-made vectors and texts, so
the scores and ranks can be worked out. Qdrant, the LangChain stores and the cross-encoder are replaced by fakes,
so no server or model is needed.
"""

from types import SimpleNamespace

import faiss
import numpy as np
import pytest
from rank_bm25 import BM25Okapi

from common.args import ChunkingArgs
from common.args import IndexArgs
from common.args import RetrievalArgs
from stages.build.chunking.schema import Chunk
from stages.build.indexing.schema import Bm25Index
from stages.build.indexing.schema import FaissIndex
from stages.build.indexing.schema import QdrantIndex
from l5_query.schema import Ans
from l5_query.schema import Qus
from l5_query.schema import Query
from l6_retrieval.schema import Retrieved


def unit_vector(values):
    """
    Input: values, a list of numbers.
    Output: a float32 numpy vector of length 1 in the same direction.
    What it does: normalizes the vector like embedding and query transformation do, so a dot product is the cosine similarity.
    """
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


@pytest.fixture
def make_vector():
    """
    Input: none.
    Output: the unit_vector function.
    What it does: lets a test build normalized vectors.
    """
    return unit_vector


@pytest.fixture
def make_query():
    """
    Input: none.
    Output: a function that builds a Query.
    What it does: returns a builder where each query type's vector and tokens can be given, with 3-dimensional
    defaults.
    """
    def build_query(query_id="eval_qna_0", original=(1, 0, 0), rewritten=(0, 1, 0), multi=((0, 0, 1),), hyde=(1, 1, 0),
                    hyde_text="A hypothetical passage.", question="What is the refund policy?",
                    original_tokens=("refund", "policy"), rewritten_tokens=("refund", "rules"), multi_tokens=(("money", "back"),)):
        """
        Input: the query id; the vector of each query type; hyde_text, the HyDE answer (I_DONT_KNOW if refused);
        question, the original question; and the tokens of the original, rewritten and multi questions.
        Output: a Query.
        What it does: builds the Query l5 would give, with one multi phrasing per multi vector.
        """
        multi_qus = []
        for position in range(len(multi)):
            multi_qus.append(Qus(qus=f"phrasing {position}", tokens=list(multi_tokens[position]), embeddings=unit_vector(multi[position])))
        return Query(
            id=query_id,
            original_qus=Qus(qus=question, tokens=list(original_tokens), embeddings=unit_vector(original)),
            rewritten_qus=Qus(qus="rewritten " + question, tokens=list(rewritten_tokens), embeddings=unit_vector(rewritten)),
            multi_qus=multi_qus,
            hyde_ans=Ans(ans=hyde_text, embeddings=unit_vector(hyde)),
            normalized=True,
            embedding_model="all-MiniLM-L6-v2",
            llm="ollama",
            llm_model="llama3.1:8b"
        )

    return build_query


@pytest.fixture
def make_faiss_index():
    """
    Input: none.
    Output: a function that builds a real in-memory FaissIndex.
    What it does: returns a builder that adds the given vectors to a flat inner-product index, like the faiss_flatip index.
    """
    def build_faiss_index(chunk_ids, vectors):
        """
        Input: chunk_ids, and one vector (a list of numbers) per chunk.
        Output: a FaissIndex holding the normalized vectors, position i being chunk_ids[i].
        What it does: builds the index the dense retriever searches.
        """
        matrix = []
        for values in vectors:
            matrix.append(unit_vector(values))
        matrix = np.array(matrix, dtype=np.float32)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        return FaissIndex(faiss_index_type="faiss_flatip", faiss_index=index, chunk_ids=list(chunk_ids),
                          index_file_path="unused.faiss", dimension=matrix.shape[1])

    return build_faiss_index


@pytest.fixture
def make_bm25_index():
    """
    Input: none.
    Output: a function that builds a real in-memory Bm25Index.
    What it does: returns a builder that indexes the given texts like BM25 indexing: lowercased and split on whitespace.
    """
    def build_bm25_index(chunk_ids, texts):
        """
        Input: chunk_ids, and one text per chunk.
        Output: a Bm25Index, position i being chunk_ids[i].
        What it does: builds the index the BM25 retriever searches.
        """
        tokenized_texts = []
        for text in texts:
            tokenized_texts.append(text.lower().split())
        return Bm25Index(bm25_model=BM25Okapi(tokenized_texts), chunk_ids=list(chunk_ids), index_file_path="unused.pkl")

    return build_bm25_index


class FakeQdrantClient:
    """
    Stand-in for QdrantClient: query_points returns set points and records each search.
    """

    def __init__(self, points):
        """
        Input: points, a list of (chunk_id, score) pairs every search returns.
        Output: nothing.
        What it does: stores the points.
        """
        self.points = points
        self.searches = []

    def query_points(self, collection_name, query, limit, with_payload):
        """
        Input: collection_name, query (the vector as a list), limit and with_payload, as the real client takes them.
        Output: an object whose points have a chunk_id payload and a score, at most limit of them.
        What it does: records the search and returns the set points.
        """
        self.searches.append({"collection_name": collection_name, "query": query, "limit": limit, "with_payload": with_payload})
        points = []
        for chunk_id, score in self.points[:limit]:
            points.append(SimpleNamespace(payload={"chunk_id": chunk_id}, score=score))
        return SimpleNamespace(points=points)


@pytest.fixture
def make_qdrant_index():
    """
    Input: none.
    Output: a function that builds a QdrantIndex with a FakeQdrantClient.
    What it does: lets a test search "Qdrant" without a server.
    """
    def build_qdrant_index(points):
        """
        Input: points, the (chunk_id, score) pairs every search returns.
        Output: a QdrantIndex on collection test_collection.
        What it does: builds the index the dense retriever searches.
        """
        return QdrantIndex(qdrant_client=FakeQdrantClient(points), collection_name="test_collection", dimension=3)

    return build_qdrant_index


class FakeCrossEncoder:
    """
    Stand-in for sentence_transformers.CrossEncoder: scores a pair by a set score for its chunk text.
    """

    def __init__(self, score_by_text):
        """
        Input: score_by_text, a dict of chunk text to score.
        Output: nothing.
        What it does: stores the scores; records each predict call.
        """
        self.score_by_text = score_by_text
        self.predict_calls = []

    def predict(self, pairs, batch_size, show_progress_bar):
        """
        Input: pairs, (question, chunk text) tuples; batch_size and show_progress_bar.
        Output: one score per pair, in order.
        What it does: records the pairs and looks up each chunk text's score.
        """
        self.predict_calls.append(list(pairs))
        scores = []
        for question, text in pairs:
            scores.append(self.score_by_text[text])
        return np.array(scores, dtype=np.float32)


@pytest.fixture
def fake_cross_encoder():
    """
    Input: none.
    Output: the FakeCrossEncoder class.
    What it does: lets a test rerank without loading a model.
    """
    return FakeCrossEncoder


@pytest.fixture
def make_chunk():
    """
    Input: none.
    Output: a function that builds a Chunk.
    What it does: returns a builder for a fixed-type chunk with an id and text.
    """
    def build_chunk(chunk_id, text):
        """
        Input: chunk_id and text.
        Output: a Chunk.
        What it does: builds the chunk the reranker reads the text of.
        """
        return Chunk(id=chunk_id, text=text, size=len(text.split()), type="fixed")

    return build_chunk


@pytest.fixture
def make_retrieved():
    """
    Input: none.
    Output: a function that builds a Retrieved.
    What it does: returns a builder where any field can be overridden by keyword.
    """
    def build_retrieved(**overrides):
        """
        Input: any Retrieved field as a keyword argument.
        Output: a Retrieved with the defaults plus the overrides. doc_ids follow chunk_ids unless given.
        What it does: fills in a not reranked dense output for every field not overridden.
        """
        field_values = {
            "id": "eval_qna_0", "query_type": "original", "chunk_type": "fixed", "retrieval_type": "dense",
            "index_type": "faiss_flatip", "top_k": 3, "reranking": False, "rerank_top_k": 0,
            "chunk_ids": ["doc_a_0", "doc_b_1"], "chunk_scores": [0.9, 0.5],
        }
        for field_name in overrides:
            field_values[field_name] = overrides[field_name]
        if "doc_ids" not in field_values:
            doc_ids = []
            for chunk_id in field_values["chunk_ids"]:
                doc_ids.append(chunk_id.rsplit("_", 1)[0])
            field_values["doc_ids"] = doc_ids
        return Retrieved(**field_values)

    return build_retrieved


@pytest.fixture
def make_retrieval_args():
    """
    Input: none.
    Output: a function that builds a RetrievalArgs.
    What it does: returns a builder where any field can be overridden by keyword.
    """
    def build_retrieval_args(**overrides):
        """
        Input: any RetrievalArgs field as a keyword argument.
        Output: a RetrievalArgs with the defaults plus the overrides.
        What it does: fills in a built-in, not reranked run of all three retrieval types.
        """
        field_values = {"framework": "", "retrieval_types": ["dense", "bm25", "hybrid"], "query_types": ["original", "rewritten", "multi", "hyde"], "top_k": 3, "reranking": False, "rerank_top_k": 2}
        for field_name in overrides:
            field_values[field_name] = overrides[field_name]
        return RetrievalArgs(**field_values)

    return build_retrieval_args


@pytest.fixture
def chunking_args():
    """
    Input: none.
    Output: the ChunkingArgs of a built-in fixed chunking run.
    What it does: gives the retrieval cache folders their chunk settings.
    """
    return ChunkingArgs(framework="", model_name="all-MiniLM-L6-v2", chunk_type="fixed", min_chunk_size=128,
                        max_chunk_size=256, fixed_chunk_overlap=32, semantic_threshold=0.2)


@pytest.fixture
def index_args():
    """
    Input: none.
    Output: the IndexArgs of the run, with index_type left empty as eval.py gives it.
    What it does: gives the retrieval cache folders their index settings.
    """
    return IndexArgs(index_type="", ivf_nlist=256, hnsw_m=32, hnsw_ef_construction=200, bm25_k1=1.5, bm25_b=0.75,
                     bm25_tokenizer="whitespace", qdrant_batch_size=256)
