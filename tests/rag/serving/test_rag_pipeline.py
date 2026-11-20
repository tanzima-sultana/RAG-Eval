"""
test_rag_pipeline.py

Tests for rag.serving.rag_pipeline. Nothing is loaded from disk and no model or LLM is called: the build layers, the
embedding model, the query LLM and the cross-encoder are replaced with fakes. The fake embedding model embeds a text
as [len(text), 1, 0, 0], so the fake vector store finds chunks by the length of the searched text.
"""

import types

import numpy as np
import pytest

from common.args_temp import QueryArgs
from common.args_temp import RetrievalArgs
from rag.serving import rag_pipeline as rag_pipeline_module
from rag.serving.rag_pipeline import RagPipeline
from rag.serving.retrieval import single_retrieval as single_retrieval_module
from schema.chunk import Chunk
from schema.index import Bm25Index
from schema.index import FaissIndex
from schema.llm import LLMArgs
from schema.setup import Setup
from schema.vectordb import VectorDb

QUESTION = "What is X?"
REWRITTEN_QUESTION = "What does X mean?"


class FakeVectorStore:
    """
    Stand-in for a LangChain FAISS or Qdrant vector store. A vector whose first number is the length of the question
    finds FIRST_CHUNKS; the length of the rewritten question finds REWRITTEN_CHUNKS.
    """

    def __init__(self):
        """
        Input: none.
        Output: nothing.
        What it does: starts an empty search record.
        """
        self.searched_first_values = []

    def similarity_search_with_score_by_vector(self, embedding, k):
        """
        Input: embedding, the query vector as a list; k, the number of chunks to return.
        Output: up to k (Document-like object, score) pairs.
        What it does: records the search and returns the chunks of the searched text's length.
        """
        self.searched_first_values.append(embedding[0])
        if embedding[0] == float(len(REWRITTEN_QUESTION)):
            results = [("doc2_0", 0.9)]
        else:
            results = [("doc1_0", 0.9), ("doc1_1", 0.8), ("doc2_0", 0.7)]
        found = []
        for chunk_id, score in results[:k]:
            found.append((types.SimpleNamespace(metadata={"chunk_id": chunk_id}), score))
        return found


class FakeBm25Model:
    """
    Stand-in for the rank_bm25 model of a BM25Retriever over the 3 chunks doc1_0, doc1_1, doc2_0.
    """

    def get_scores(self, query_tokens):
        """
        Input: query_tokens, the tokens of one query.
        Output: one score per chunk position.
        What it does: scores doc2_0 highest and doc1_1 second for any query.
        """
        return np.array([0.0, 2.0, 3.0], dtype=np.float64)


class FakeLLMProvider:
    """
    Stand-in for LLMProvider. It rewrites any question to REWRITTEN_QUESTION. Every built model is recorded.
    """

    built_models = []

    def __init__(self, llm_args, ollama_args):
        """
        Input: llm_args and ollama_args, as LLMProvider takes them.
        Output: nothing.
        What it does: records the model.
        """
        FakeLLMProvider.built_models.append(llm_args.llm_model)

    def generate(self, system_prompt, user_prompt):
        """
        Input: system_prompt and user_prompt (ignored).
        Output: an object with the rewrite in .text.
        What it does: returns the rewritten question.
        """
        return types.SimpleNamespace(text=REWRITTEN_QUESTION)


class FakeCrossEncoder:
    """
    Stand-in for sentence_transformers.CrossEncoder. A (question, chunk text) pair scores the number in the chunk text.
    Every scored question is recorded.
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
def fake_build(monkeypatch, make_chunking_args, make_index_args, fake_huggingface_embeddings):
    """
    Input: pytest's monkeypatch fixture; the args builders; the fake_huggingface_embeddings fixture.
    Output: a record of what the pipeline loaded: loaded_index_types and loaded_vectordb_types.
    What it does: replaces every loader RagPipeline calls with a fake, so a pipeline can be built with no files.
    """
    loaded = types.SimpleNamespace(loaded_index_types=[], loaded_vectordb_types=[], ingestion_calls=0)
    chunks = {
        "doc1_0": Chunk(id="doc1_0", text="chunk 0.1", size=2, type="fixed"),
        "doc1_1": Chunk(id="doc1_1", text="chunk 0.9", size=2, type="fixed"),
        "doc2_0": Chunk(id="doc2_0", text="chunk 0.5", size=2, type="fixed"),
    }

    def fake_ingestion(ingestion_args):
        """
        Input: ingestion_args (ignored).
        Output: no documents.
        What it does: counts the call; the fake chunking needs no documents.
        """
        loaded.ingestion_calls = loaded.ingestion_calls + 1
        return {}

    class FakeChunking:
        """
        Stand-in for Chunking. compute_chunks returns the 3 fixed chunks.
        """

        def __init__(self, chunking_args):
            """
            Input: chunking_args (ignored).
            Output: nothing.
            What it does: nothing.
            """
            return None

        def compute_chunks(self, documents, output_path):
            """
            Input: documents and output_path (ignored).
            Output: the 3 chunks.
            What it does: returns the fixed chunks.
            """
            return chunks

    class FakeEmbedding:
        """
        Stand-in for Embedding. compute_embeddings returns no embeddings; the fake indexes need none.
        """

        def __init__(self, embedding_args):
            """
            Input: embedding_args (ignored).
            Output: nothing.
            What it does: nothing.
            """
            return None

        def compute_embeddings(self, chunks, output_path):
            """
            Input: chunks and output_path (ignored).
            Output: an empty dict.
            What it does: nothing.
            """
            return {}

    class FakeIndexing:
        """
        Stand-in for Indexing. The index args are the index type itself, so compute_index knows which to return.
        """

        def __init__(self, index_type):
            """
            Input: index_type, as the fake make_index_args returns it.
            Output: nothing.
            What it does: stores it.
            """
            self.index_type = index_type

        def compute_index(self, chunks, embeddings, output_path):
            """
            Input: chunks, embeddings and output_path (ignored).
            Output: a Bm25Index for bm25, otherwise a FaissIndex holding a FakeVectorStore.
            What it does: records the index type and returns its fake index.
            """
            loaded.loaded_index_types.append(self.index_type)
            if self.index_type == "bm25":
                documents = []
                for chunk_id in ["doc1_0", "doc1_1", "doc2_0"]:
                    documents.append(types.SimpleNamespace(metadata={"chunk_id": chunk_id}))
                retriever = types.SimpleNamespace(docs=documents, vectorizer=FakeBm25Model())
                return Bm25Index(retriever=retriever, index_path="unused")
            return FaissIndex(index_type=self.index_type, vector_store=FakeVectorStore(), index_path="unused", dimension=4)

    class FakeIndexingVectorDb:
        """
        Stand-in for IndexingVectorDb. compute_vectordb returns a VectorDb holding a FakeVectorStore.
        """

        def __init__(self, vectordb_type):
            """
            Input: vectordb_type, as the fake make_vectordb_args returns it.
            Output: nothing.
            What it does: stores it.
            """
            self.vectordb_type = vectordb_type

        def compute_vectordb(self, chunks, embeddings, output_path):
            """
            Input: chunks, embeddings and output_path (ignored).
            Output: a VectorDb holding a FakeVectorStore.
            What it does: records the vector DB type and returns its fake collection.
            """
            loaded.loaded_vectordb_types.append(self.vectordb_type)
            return VectorDb(vectordb_type=self.vectordb_type, vector_store=FakeVectorStore(), collection_name="c", dimension=4)

    def fake_make_chunking_args(args, chunk_type):
        """
        Input: args (ignored); chunk_type.
        Output: a small ChunkingArgs of that chunk type.
        What it does: builds the chunking settings without parsed args.
        """
        return make_chunking_args(chunk_type=chunk_type)

    def fake_make_index_args(args, chunking_args, index_type):
        """
        Input: args and chunking_args (ignored); index_type.
        Output: the index type itself, which FakeIndexing reads.
        What it does: passes the index type through.
        """
        return index_type

    def fake_make_vectordb_args(args, chunking_args, vectordb_type):
        """
        Input: args and chunking_args (ignored); vectordb_type.
        Output: the vector DB type itself, which FakeIndexingVectorDb reads.
        What it does: passes the vector DB type through.
        """
        return vectordb_type

    def fake_make_query_args(args):
        """
        Input: args (ignored).
        Output: a QueryArgs with the fake embedding model and an Ollama query model.
        What it does: builds the query settings without parsed args.
        """
        return QueryArgs(
            query_llm_args=LLMArgs("ollama_chat/llama3.1:8b", 100, 30),
            embedding_model="fake-model",
            n_multi_qus=2,
            query_types=["original", "rewritten", "multi", "hyde"],
            query_split=True,
            val_queries_size=1,
            test_queries_size=1,
        )

    def fake_make_retrieval_args(args, chunking_args):
        """
        Input: args (ignored); chunking_args.
        Output: a RetrievalArgs with top_k 3 and rerank_top_k 2. reranking is false; the pipeline sets it from the setup.
        What it does: builds the retrieval settings without parsed args.
        """
        return RetrievalArgs(
            index_args=make_index_args(chunking_args=chunking_args, index_type=""),
            retrieval_types=["dense", "bm25", "hybrid"],
            top_k=3,
            reranking=False,
            rerank_top_k=2,
        )

    def fake_make_none(*arguments):
        """
        Input: any arguments (ignored).
        Output: None.
        What it does: stands in for an args builder whose result the fakes do not read.
        """
        return None

    monkeypatch.setattr(rag_pipeline_module, "ingestion", fake_ingestion)
    monkeypatch.setattr(rag_pipeline_module, "Chunking", FakeChunking)
    monkeypatch.setattr(rag_pipeline_module, "Embedding", FakeEmbedding)
    monkeypatch.setattr(rag_pipeline_module, "Indexing", FakeIndexing)
    monkeypatch.setattr(rag_pipeline_module, "IndexingVectorDb", FakeIndexingVectorDb)
    monkeypatch.setattr(rag_pipeline_module, "HuggingFaceEmbeddings", fake_huggingface_embeddings)
    monkeypatch.setattr(rag_pipeline_module, "LLMProvider", FakeLLMProvider)
    monkeypatch.setattr(single_retrieval_module, "CrossEncoder", FakeCrossEncoder)
    monkeypatch.setattr(rag_pipeline_module, "make_ingestion_args", fake_make_none)
    monkeypatch.setattr(rag_pipeline_module, "make_embedding_args", fake_make_none)
    monkeypatch.setattr(rag_pipeline_module, "make_ollama_args", fake_make_none)
    monkeypatch.setattr(rag_pipeline_module, "make_chunking_args", fake_make_chunking_args)
    monkeypatch.setattr(rag_pipeline_module, "make_index_args", fake_make_index_args)
    monkeypatch.setattr(rag_pipeline_module, "make_vectordb_args", fake_make_vectordb_args)
    monkeypatch.setattr(rag_pipeline_module, "make_query_args", fake_make_query_args)
    monkeypatch.setattr(rag_pipeline_module, "make_retrieval_args", fake_make_retrieval_args)
    FakeLLMProvider.built_models = []
    FakeCrossEncoder.scored_questions = []
    return loaded


def make_args():
    """
    Input: none.
    Output: the parsed args the pipeline reads directly: output_path and dataset_name.
    What it does: builds the two args the pipeline reads itself; the fakes need no others.
    """
    return types.SimpleNamespace(output_path="out", dataset_name="dataset")


def make_setup(retrieval_type, index_type, query_type="original", reranking=False):
    """
    Input: retrieval_type; index_type; query_type; reranking.
    Output: a Setup of the fixed chunk type.
    What it does: builds one setup for the pipeline.
    """
    return Setup(chunk_type="fixed", index_type=index_type, query_type=query_type, retrieval_type=retrieval_type, reranking=reranking)


def test_dense_original_loads_its_index_and_no_llm(fake_build):
    """
    Input: a dense faiss_flatip original setup and one question.
    Output: the Retrieved of run.
    What it does: checks only its FAISS index is loaded, no query LLM is built, and the result carries the setup and id.
    """
    setup = make_setup("dense", "faiss_flatip")
    pipeline = RagPipeline(setup, make_args())

    retrieved = pipeline.run("call_1", QUESTION)

    assert fake_build.loaded_index_types == ["faiss_flatip"]
    assert fake_build.loaded_vectordb_types == []
    assert FakeLLMProvider.built_models == []
    assert retrieved.id == "call_1"
    assert retrieved.setup == setup
    assert retrieved.chunk_ids == ["doc1_0", "doc1_1", "doc2_0"]


def test_qdrant_loads_the_collection(fake_build):
    """
    Input: a dense qdrant setup.
    Output: the Retrieved of run.
    What it does: checks the Qdrant collection is loaded instead of a FAISS index.
    """
    pipeline = RagPipeline(make_setup("dense", "qdrant"), make_args())

    retrieved = pipeline.run("call_1", QUESTION)

    assert fake_build.loaded_index_types == []
    assert fake_build.loaded_vectordb_types == ["qdrant"]
    assert retrieved.setup.index_type == "qdrant"


def test_rewritten_builds_the_llm_and_searches_the_rewrite(fake_build):
    """
    Input: a dense rewritten setup.
    Output: the Query of transform and the Retrieved of run.
    What it does: checks the query LLM is built, the Query holds only rewritten, and the rewrite is what is searched.
    """
    pipeline = RagPipeline(make_setup("dense", "faiss_flatip", query_type="rewritten"), make_args())

    query = pipeline.transform("call_1", QUESTION)
    retrieved = pipeline.run("call_1", QUESTION)

    assert FakeLLMProvider.built_models == ["ollama_chat/llama3.1:8b"]
    assert query.original_qus is None
    assert query.rewritten_qus.qus == REWRITTEN_QUESTION
    assert retrieved.setup.query_type == "rewritten"
    assert retrieved.chunk_ids == ["doc2_0"]


def test_bm25_loads_only_the_bm25_index(fake_build):
    """
    Input: a bm25 setup.
    Output: the Retrieved of run.
    What it does: checks only the BM25 index is loaded and chunks are ranked by their BM25 score.
    """
    pipeline = RagPipeline(make_setup("bm25", "bm25"), make_args())

    retrieved = pipeline.run("call_1", QUESTION)

    assert fake_build.loaded_index_types == ["bm25"]
    assert retrieved.chunk_ids == ["doc2_0", "doc1_1"]


def test_hybrid_loads_both_indexes_and_fuses_them(fake_build):
    """
    Input: a hybrid faiss_flatip+bm25 setup.
    Output: the Retrieved of run.
    What it does: checks both indexes are loaded and the result is the fusion of the dense and BM25 results.
    """
    pipeline = RagPipeline(make_setup("hybrid", "faiss_flatip+bm25"), make_args())
    query = pipeline.transform("call_1", QUESTION)

    retrieved = pipeline.run("call_1", QUESTION)

    single_retrieval = pipeline.single_retrieval
    dense_retrieved = single_retrieval.dense_search(pipeline.vector_store, "faiss_flatip", query)[0]
    bm25_retrieved = single_retrieval.bm25_search(pipeline.bm25_index, query)[0]
    assert fake_build.loaded_index_types == ["faiss_flatip", "bm25"]
    assert retrieved == single_retrieval.hybrid_fuse(dense_retrieved, bm25_retrieved)
    assert retrieved.setup.index_type == "faiss_flatip+bm25"


def test_reranking_reranks_against_the_question(fake_build):
    """
    Input: a reranked dense setup whose chunk texts end with their cross-encoder score.
    Output: the Retrieved of run.
    What it does: checks the cross-encoder is loaded, scores against the question and rerank_top_k chunks are kept.
    """
    pipeline = RagPipeline(make_setup("dense", "faiss_flatip", reranking=True), make_args())

    retrieved = pipeline.run("call_1", QUESTION)

    assert pipeline.single_retrieval.cross_encoder is not None
    assert retrieved.setup.reranking is True
    assert retrieved.chunk_ids == ["doc1_1", "doc2_0"]
    assert set(FakeCrossEncoder.scored_questions) == {QUESTION}


def test_no_reranking_loads_no_cross_encoder(fake_build):
    """
    Input: a plain dense setup.
    Output: the pipeline.
    What it does: checks the cross-encoder is not loaded when the setup does not rerank.
    """
    pipeline = RagPipeline(make_setup("dense", "faiss_flatip"), make_args())

    assert pipeline.single_retrieval.cross_encoder is None


@pytest.mark.parametrize("setup, message", [
    (make_setup("bm25", "bm25", query_type="hyde"), "no hyde"),
    (make_setup("hybrid", "faiss_flatip+bm25", query_type="hyde"), "no hyde"),
    (make_setup("hybrid", "faiss_flatip"), "joined with"),
])
def test_validate_raises_before_loading_anything(fake_build, setup, message):
    """
    Input: a setup retrieval cannot run.
    Output: ValueError.
    What it does: checks the setup is refused before anything is loaded.
    """
    with pytest.raises(ValueError, match=message):
        RagPipeline(setup, make_args())

    assert fake_build.ingestion_calls == 0
