"""
test_pipeline.py

Tests for rag.serving.pipeline. Every layer and loader the Pipeline uses is a fake that records how it was built and
called, so no model, index or LLM is loaded.
"""

import types

import pytest

from common.args import PipelineArgs
from rag.serving import pipeline as pipeline_module
from rag.serving.pipeline import Pipeline


class PipelineRecord:
    """
    What every fake was given, in order.
    """

    def __init__(self):
        """
        Input: none.
        Output: nothing.
        What it does: starts empty records.
        """
        self.built = []
        self.loaded = []
        self.calls = []


@pytest.fixture
def record(monkeypatch):
    """
    Input: pytest's monkeypatch fixture.
    Output: the PipelineRecord the fakes fill.
    What it does: replaces the query layer, the embedding model, the index loaders and the retrieval layer in the
    pipeline module with fakes.
    """
    pipeline_record = PipelineRecord()

    class FakeQueryTransformation:
        """
        Stand-in for QueryTransformation: gives a placeholder query.
        """

        def __init__(self, query_args, query_path):
            """
            Input: the arguments of QueryTransformation.
            Output: nothing.
            What it does: records that the query layer was built.
            """
            pipeline_record.built.append(("query", query_path))

        def query(self, question_id, question):
            """
            Input: the arguments of query.
            Output: a placeholder query holding the question id.
            What it does: records the call.
            """
            pipeline_record.calls.append(("query", question_id, question))
            return types.SimpleNamespace(id=question_id)

    class FakeRetrieval:
        """
        Stand-in for Retrieval: gives a placeholder result.
        """

        def __init__(self, retrieval_args, query_args, faiss_index, vectordb_index, bm25_index, retrieved_path):
            """
            Input: the arguments of Retrieval.
            Output: nothing.
            What it does: records which index records it was given.
            """
            pipeline_record.built.append(("retrieval", faiss_index, vectordb_index, bm25_index, retrieved_path))

        def retrieve(self, query):
            """
            Input: the argument of retrieve.
            Output: a placeholder result.
            What it does: records the call.
            """
            pipeline_record.calls.append(("retrieve", query.id))
            return "retrieved"

    def fake_load_index_from_path(index_type, index_folder, embedding_model):
        """
        Input: the arguments of load_index_from_path.
        Output: a placeholder index record named by its type.
        What it does: records the load.
        """
        pipeline_record.loaded.append((index_type, index_folder))
        return f"{index_type} record"

    def fake_load_vectordb_from_path(collection_name, embedding_model, qdrant_url):
        """
        Input: the arguments of load_vectordb_from_path.
        Output: a placeholder vector DB record.
        What it does: records the load.
        """
        pipeline_record.loaded.append(("qdrant", collection_name, qdrant_url))
        return "qdrant record"

    def fake_huggingface_embeddings(model_name, model_kwargs, encode_kwargs):
        """
        Input: the arguments of HuggingFaceEmbeddings.
        Output: a placeholder embedding model.
        What it does: records the model name.
        """
        pipeline_record.built.append(("embedding_model", model_name))
        return "embedding model"

    monkeypatch.setattr(pipeline_module, "QueryTransformation", FakeQueryTransformation)
    monkeypatch.setattr(pipeline_module, "Retrieval", FakeRetrieval)
    monkeypatch.setattr(pipeline_module, "load_index_from_path", fake_load_index_from_path)
    monkeypatch.setattr(pipeline_module, "load_vectordb_from_path", fake_load_vectordb_from_path)
    monkeypatch.setattr(pipeline_module, "HuggingFaceEmbeddings", fake_huggingface_embeddings)
    return pipeline_record


def make_pipeline_args(index_type, index_paths):
    """
    Input: index_type; index_paths, in its order.
    Output: a PipelineArgs with placeholder sub-args.
    What it does: builds the args the Pipeline reads; the layers are fakes, so the sub-args are placeholders.
    """
    return PipelineArgs(
        index_type=index_type, index_paths=index_paths, embedding_model="fake-model", qdrant_url="http://qdrant-test:6333",
        query_args="query args", query_path="out/queries", retrieval_args="retrieval args", retrieved_path="out/retrieved",
        context_args=None, prompt_args=None, answer_args=None,
    )


@pytest.mark.parametrize("index_type, index_paths, loaded, faiss_index, vectordb_index, bm25_index", [
    ("faiss_flatip", ["out/faiss"], [("faiss_flatip", "out/faiss")], "faiss_flatip record", None, None),
    ("bm25", ["out/bm25"], [("bm25", "out/bm25")], None, None, "bm25 record"),
    ("qdrant", ["qdrant_c"], [("qdrant", "qdrant_c", "http://qdrant-test:6333")], None, "qdrant record", None),
    ("faiss_ivf+bm25", ["out/ivf", "out/bm25"], [("faiss_ivf", "out/ivf"), ("bm25", "out/bm25")], "faiss_ivf record", None, "bm25 record"),
    ("qdrant+bm25", ["qdrant_c", "out/bm25"], [("qdrant", "qdrant_c", "http://qdrant-test:6333"), ("bm25", "out/bm25")], None, "qdrant record", "bm25 record"),
])
def test_init_loads_each_index_by_its_type_and_builds_every_layer_once(record, index_type, index_paths, loaded, faiss_index, vectordb_index, bm25_index):
    """
    Input: each kind of setup: FAISS, BM25, Qdrant, and hybrid on FAISS or Qdrant.
    Output: what the fakes were given.
    What it does: checks each index path is loaded with the loader of its type, in order, and the query layer, the
    embedding model, the query layer and the retrieval layer are built once, in that order, with the matching index
    records.
    """
    pipeline = Pipeline(make_pipeline_args(index_type, index_paths))

    assert record.loaded == loaded
    assert record.built == [
        ("embedding_model", "fake-model"),
        ("query", "out/queries"),
        ("retrieval", faiss_index, vectordb_index, bm25_index, "out/retrieved"),
    ]
    assert pipeline.context is None
    assert pipeline.prompt is None
    assert pipeline.answer is None


def test_invoke_makes_the_query_then_retrieves_it_and_gives_no_answer(record):
    """
    Input: one question sent to a built pipeline, twice.
    Output: the results and the calls.
    What it does: checks each call makes the query and then retrieves it, returns the retrieved result and no answer,
    and builds nothing again.
    """
    pipeline = Pipeline(make_pipeline_args("faiss_flatip", ["out/faiss"]))
    built_count = len(record.built)

    first_result = pipeline.invoke("q1", "What is X?")
    second_result = pipeline.invoke("q2", "What is Y?")

    assert first_result == ("retrieved", None)
    assert second_result == ("retrieved", None)
    assert record.calls == [("query", "q1", "What is X?"), ("retrieve", "q1"), ("query", "q2", "What is Y?"), ("retrieve", "q2")]
    assert len(record.built) == built_count
