"""
test_run_benchmark.py

Tests for run_benchmark and the benchmark hash. No model, LLM or Qdrant server is used: every loader and step the
benchmark calls is replaced with a fake that records what it was given.
"""

import argparse
import dataclasses
import json
import os
import types

import pytest

import run_benchmark as run_benchmark_module
from common import paths
from common.args import BenchmarkArgs
from common.args import OllamaArgs
from common.args import QueryArgs
from schema.eval_dataset_metadata import EvalDatasetMetadata
from schema.llm import LLMArgs
from schema.manifest import BuildManifest
from schema.manifest import ChunkEntry
from schema.manifest import IndexEntry


def make_benchmark_parsed_args(tmp_path, manifest_path, eval_dataset_metadata_path):
    """
    Input: tmp_path; manifest_path; eval_dataset_metadata_path.
    Output: an argparse.Namespace standing in for the parsed benchmark config.
    What it does: holds every benchmark setting, with the output folder in tmp_path.
    """
    return argparse.Namespace(
        manifest_path=manifest_path, eval_dataset_metadata_path=eval_dataset_metadata_path, output_path=str(tmp_path),
        query_llm_model="ollama_chat/llama3.1:8b", n_multi_qus=3, query_types=["original"], query_split="false",
        val_queries_size=1, test_queries_size=1, query_llm_max_tokens=100, query_llm_timeout_seconds=30,
        ollama_api_base="http://ollama-test:11434", ollama_context_length=32768,
        retrieval_types=["dense", "bm25", "hybrid"], top_k=3, reranking="false", rerank_top_k=2, n_top_setups=1,
        qdrant_url="http://qdrant-test:6333", qdrant_batch_size=3,
    )


def save_manifest(tmp_path, dataset_name):
    """
    Input: tmp_path; dataset_name.
    Output: the path of the saved manifest JSON.
    What it does: saves a manifest of two chunk types, each with a FAISS IVF, a BM25 and a Qdrant index.
    """
    chunk_entries = [
        ChunkEntry("fixed", 2, 6, 2, 0.5, "out/chunks/chunks_fixed", "out/embeddings/embeddings_fixed", 4),
        ChunkEntry("sentence", 2, 6, 2, 0.5, "out/chunks/chunks_sentence", "out/embeddings/embeddings_sentence", 4),
    ]
    index_entries = []
    for chunk_type in ["fixed", "sentence"]:
        index_entries.append(IndexEntry(chunk_type, "faiss_ivf", {"ivf_nlist": 7}, f"out/indexes/faiss_ivf_{chunk_type}", 4))
        index_entries.append(IndexEntry(chunk_type, "bm25", {"bm25_k1": 1.5, "bm25_b": 0.75, "bm25_tokenizer": "whitespace"}, f"out/indexes/bm25_{chunk_type}", 4))
        index_entries.append(IndexEntry(chunk_type, "qdrant", {}, f"qdrant_{chunk_type}", 4))
    manifest = BuildManifest(
        build_hash="testbuildhash", input_hash="testinputhash", dataset_name=dataset_name, dataset_size=2,
        documents_path="out/documents", embedding_model="fake-model", built_at="2026-11-20T00:00:00",
        chunk_entries=chunk_entries, index_entries=index_entries,
    )
    manifest_path = os.path.join(str(tmp_path), "manifest_testbuildhash.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(manifest), f)
    return manifest_path


def save_eval_dataset_metadata(tmp_path, dataset_name):
    """
    Input: tmp_path; dataset_name.
    Output: the path of the saved eval dataset metadata JSON.
    What it does: saves metadata with test values and the given dataset name.
    """
    eval_dataset_metadata = EvalDatasetMetadata(
        eval_dataset_hash="testevalhash", eval_dataset_path="out/eval_dataset/eval_dataset_testevalhash.json",
        knowledge_graph_path="out/eval_dataset/kg_testevalhash.json", content_hash="hash", n_evals_generated=2,
        created_at="2026-11-20T00:00:00+00:00", dataset_name=dataset_name, dataset_size=2,
        documents_path="out/documents/doc_ids_testinputhash.json", documents_hash="hash",
        eval_dataset_llm_model="anthropic/claude-sonnet-5-5", eval_dataset_llm_max_tokens=100,
        eval_dataset_kg_llm_model="anthropic/claude-haiku-5-5", eval_dataset_kg_llm_max_tokens=100,
        ollama_context_length=None, embedding_model="fake-model", n_sample_documents=1, n_evals_requested=2,
        sample_documents_seed=42, min_document_tokens=500, passage_match_chars=80, ragas_version="test",
    )
    metadata_path = os.path.join(str(tmp_path), "eval_dataset_testevalhash_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(eval_dataset_metadata), f)
    return metadata_path


class BenchmarkRecord:
    """
    What every fake was given during one run_benchmark call.
    """

    def __init__(self):
        """
        Input: none.
        Output: nothing.
        What it does: starts empty records.
        """
        self.loaded_documents = []
        self.loaded_eval_datasets = []
        self.query_args = []
        self.query_output_paths = []
        self.transformed_questions = []
        self.loaded_chunks = []
        self.loaded_indexes = []
        self.loaded_vectordbs = []
        self.saved_best_setups = []


def use_fake_benchmark_steps(monkeypatch):
    """
    Input: pytest's monkeypatch fixture.
    Output: a BenchmarkRecord the fakes fill.
    What it does: replaces every loader and step run_benchmark calls with a fake that records its arguments.
    """
    record = BenchmarkRecord()

    def fake_load_documents(processed_data_path, input_hash):
        """
        Input: the arguments of load_documents.
        Output: one placeholder document.
        What it does: records the documents folder and input hash.
        """
        record.loaded_documents.append((processed_data_path, input_hash))
        return {"doc1": "document"}

    def fake_load_eval_dataset(eval_dataset_path):
        """
        Input: the argument of load_eval_dataset.
        Output: one placeholder eval sample.
        What it does: records the eval dataset path.
        """
        record.loaded_eval_datasets.append(eval_dataset_path)
        return [types.SimpleNamespace(id="eval_qna_0", qus="What is X?")]

    class FakeQueryTransformation:
        """
        Stand-in for QueryTransformation: records its args, output folder and questions, and gives one query.
        """

        def __init__(self, query_args, output_path):
            """
            Input: query_args; output_path.
            Output: nothing.
            What it does: records query_args and output_path.
            """
            record.query_args.append(query_args)
            record.query_output_paths.append(output_path)

        def transform_query(self, question_ids, questions):
            """
            Input: the arguments of transform_query.
            Output: one placeholder query.
            What it does: records the question ids and questions.
            """
            record.transformed_questions.append((question_ids, questions))
            return ["query"]

        def split_queries(self, queries):
            """
            Input: queries.
            Output: the queries as both val and test.
            What it does: nothing else.
            """
            return queries, queries

    class FakeChunking:
        """
        Stand-in for Chunking: records the chunk settings and folder it loads.
        """

        def __init__(self, chunking_args):
            """
            Input: chunking_args.
            Output: nothing.
            What it does: stores chunking_args.
            """
            self.chunking_args = chunking_args

        def load_saved_chunks(self, documents, chunk_path):
            """
            Input: the arguments of load_saved_chunks.
            Output: one placeholder chunk.
            What it does: records the chunking args and chunk folder.
            """
            record.loaded_chunks.append((self.chunking_args, chunk_path))
            return {"doc1_0": "chunk"}

    class FakeIndexing:
        """
        Stand-in for Indexing: records the index settings and folder it loads.
        """

        def __init__(self, index_args):
            """
            Input: index_args.
            Output: nothing.
            What it does: stores index_args.
            """
            self.index_args = index_args

        def load_saved_index(self, chunks, index_folder):
            """
            Input: the arguments of load_saved_index.
            Output: a placeholder index.
            What it does: records the index args and folder.
            """
            record.loaded_indexes.append((self.index_args, index_folder))
            return "index"

    class FakeIndexingVectorDb:
        """
        Stand-in for IndexingVectorDb: records the vector DB settings and collection it loads.
        """

        def __init__(self, vectordb_args):
            """
            Input: vectordb_args.
            Output: nothing.
            What it does: stores vectordb_args.
            """
            self.vectordb_args = vectordb_args

        def load_saved_vectordb(self, chunks, collection_name):
            """
            Input: the arguments of load_saved_vectordb.
            Output: a placeholder vector DB.
            What it does: records the vector DB args and collection name.
            """
            record.loaded_vectordbs.append((self.vectordb_args, collection_name))
            return "vectordb"

    class FakeRetrieval:
        """
        Stand-in for Retrieval: gives no results.
        """

        def __init__(self, retrieval_args, query_args, val_queries_count):
            """
            Input: the arguments of Retrieval.
            Output: nothing.
            What it does: nothing; the fake keeps no state.
            """
            return None

        def compute_retrieval(self, index, index_type, queries, output_path):
            """
            Input: the arguments of compute_retrieval.
            Output: an empty list.
            What it does: nothing.
            """
            return []

        def compute_vectordb_retrieval(self, vectordb, queries, output_path):
            """
            Input: the arguments of compute_vectordb_retrieval.
            Output: an empty list.
            What it does: nothing.
            """
            return []

        def hybrid_retrieval(self, retrieved_items, output_path):
            """
            Input: the arguments of hybrid_retrieval.
            Output: an empty list.
            What it does: nothing.
            """
            return []

    class FakeRetrievalEvaluation:
        """
        Stand-in for RetrievalEvaluation: gives no evals.
        """

        def __init__(self, eval_samples):
            """
            Input: eval_samples.
            Output: nothing.
            What it does: nothing; the fake keeps no state.
            """
            return None

        def compute_retrieved_evals(self, retrieved_items, chunks):
            """
            Input: the arguments of compute_retrieved_evals.
            Output: an empty list.
            What it does: nothing.
            """
            return []

    def fake_rank_setups(retrieved_evals, n_top_setups):
        """
        Input: the arguments of rank_setups.
        Output: an empty list.
        What it does: nothing.
        """
        return []

    def fake_save_best_setups(best_setups, manifest, output_path, eval_dataset_hash, benchmark_hash):
        """
        Input: the arguments of save_best_setups.
        Output: an empty list.
        What it does: records the build, eval dataset and benchmark hashes and the output folder.
        """
        record.saved_best_setups.append((manifest.build_hash, eval_dataset_hash, benchmark_hash, output_path))
        return []

    monkeypatch.setattr(run_benchmark_module, "load_documents", fake_load_documents)
    monkeypatch.setattr(run_benchmark_module, "load_eval_dataset", fake_load_eval_dataset)
    monkeypatch.setattr(run_benchmark_module, "QueryTransformation", FakeQueryTransformation)
    monkeypatch.setattr(run_benchmark_module, "Chunking", FakeChunking)
    monkeypatch.setattr(run_benchmark_module, "Indexing", FakeIndexing)
    monkeypatch.setattr(run_benchmark_module, "IndexingVectorDb", FakeIndexingVectorDb)
    monkeypatch.setattr(run_benchmark_module, "Retrieval", FakeRetrieval)
    monkeypatch.setattr(run_benchmark_module, "RetrievalEvaluation", FakeRetrievalEvaluation)
    monkeypatch.setattr(run_benchmark_module, "rank_setups", fake_rank_setups)
    monkeypatch.setattr(run_benchmark_module, "save_best_setups", fake_save_best_setups)
    return record


# ---------- the dataset name check ----------


def test_run_benchmark_raises_when_the_dataset_names_differ(tmp_path, monkeypatch):
    """
    Input: a manifest of dataset "dataset" and eval dataset metadata of dataset "other-dataset".
    Output: ValueError, and what was loaded.
    What it does: checks the benchmark stops before loading any documents.
    """
    record = use_fake_benchmark_steps(monkeypatch)
    args = make_benchmark_parsed_args(tmp_path, save_manifest(tmp_path, "dataset"), save_eval_dataset_metadata(tmp_path, "other-dataset"))

    with pytest.raises(ValueError, match="other-dataset"):
        run_benchmark_module.run_benchmark(args)

    assert record.loaded_documents == []


# ---------- run_benchmark ----------


def test_run_benchmark_loads_every_chunk_type_and_index_the_manifest_lists(tmp_path, monkeypatch):
    """
    Input: a manifest of two chunk types, each with a FAISS IVF, a BM25 and a Qdrant index, and metadata of the same
    dataset.
    Output: what every fake was given.
    What it does: checks the documents, eval set, chunks, indexes and collections are loaded from the manifest and
    metadata with the build's settings, and the best setups are saved under the three hashes.
    """
    record = use_fake_benchmark_steps(monkeypatch)
    args = make_benchmark_parsed_args(tmp_path, save_manifest(tmp_path, "dataset"), save_eval_dataset_metadata(tmp_path, "dataset"))

    run_benchmark_module.run_benchmark(args)

    assert record.loaded_documents == [("out/documents", "testinputhash")]
    assert record.loaded_eval_datasets == ["out/eval_dataset/eval_dataset_testevalhash.json"]
    assert record.query_args[0].embedding_model == "fake-model"
    assert record.transformed_questions == [(["eval_qna_0"], ["What is X?"])]

    loaded_chunk_folders = []
    for chunking_args, chunk_path in record.loaded_chunks:
        loaded_chunk_folders.append((chunking_args.chunk_type, chunking_args.model_name, chunk_path))
    assert loaded_chunk_folders == [
        ("fixed", "fake-model", "out/chunks/chunks_fixed"),
        ("sentence", "fake-model", "out/chunks/chunks_sentence"),
    ]

    loaded_index_folders = []
    for index_args, index_folder in record.loaded_indexes:
        loaded_index_folders.append((index_args.chunking_args.chunk_type, index_args.index_type, index_folder))
        assert index_args.ivf_nlist == 7
        assert index_args.bm25_k1 == 1.5
        assert index_args.input_hash == "testinputhash"
    assert loaded_index_folders == [
        ("fixed", "faiss_ivf", "out/indexes/faiss_ivf_fixed"),
        ("fixed", "bm25", "out/indexes/bm25_fixed"),
        ("sentence", "faiss_ivf", "out/indexes/faiss_ivf_sentence"),
        ("sentence", "bm25", "out/indexes/bm25_sentence"),
    ]

    loaded_collections = []
    for vectordb_args, collection_name in record.loaded_vectordbs:
        loaded_collections.append((vectordb_args.chunking_args.chunk_type, collection_name))
        assert vectordb_args.qdrant_url == "http://qdrant-test:6333"
    assert loaded_collections == [("fixed", "qdrant_fixed"), ("sentence", "qdrant_sentence")]

    build_hash, eval_dataset_hash, benchmark_hash, output_path = record.saved_best_setups[0]
    assert build_hash == "testbuildhash"
    assert eval_dataset_hash == "testevalhash"
    assert len(benchmark_hash) == 12
    assert output_path == os.path.join(str(tmp_path), "dataset")


# ---------- get_benchmark_hash ----------


def make_benchmark_args(**overrides):
    """
    Input: any BenchmarkArgs field as a keyword argument.
    Output: a BenchmarkArgs with small defaults plus the overrides.
    What it does: fills in defaults for every field not overridden.
    """
    field_values = {}
    field_values["query_args"] = make_query_args()
    field_values["retrieval_types"] = ["dense", "bm25", "hybrid"]
    field_values["top_k"] = 3
    field_values["reranking"] = False
    field_values["rerank_top_k"] = 2
    field_values["n_top_setups"] = 1
    for field_name in overrides:
        field_values[field_name] = overrides[field_name]
    return BenchmarkArgs(**field_values)


def make_query_args(**overrides):
    """
    Input: any QueryArgs field as a keyword argument.
    Output: a QueryArgs with small defaults plus the overrides.
    What it does: fills in defaults for every field not overridden.
    """
    field_values = {}
    field_values["query_llm_args"] = LLMArgs("ollama_chat/llama3.1:8b", 100, 30)
    field_values["embedding_model"] = "fake-model"
    field_values["n_multi_qus"] = 3
    field_values["query_types"] = ["original"]
    field_values["query_split"] = False
    field_values["val_queries_size"] = 1
    field_values["test_queries_size"] = 1
    field_values["ollama_api_base"] = "http://ollama-test:11434"
    field_values["ollama_context_length"] = 32768
    for field_name in overrides:
        field_values[field_name] = overrides[field_name]
    return QueryArgs(**field_values)


def make_ollama_args(context_length=32768):
    """
    Input: context_length, the Ollama context length.
    Output: an OllamaArgs with test values.
    What it does: builds the Ollama settings the benchmark hash takes.
    """
    return OllamaArgs(ollama_api_base="http://ollama-test:11434", ollama_context_length=context_length)


@pytest.mark.parametrize("overrides", [
    {"query_args": make_query_args(n_multi_qus=4)},
    {"retrieval_types": ["dense"]},
    {"top_k": 5},
    {"reranking": True},
    {"rerank_top_k": 1},
    {"n_top_setups": 2},
])
def test_benchmark_hash_is_12_hex_characters_and_changes_with_any_arg(overrides):
    """
    Input: the default benchmark args, and the same args with one setting changed.
    Output: the hashes.
    What it does: checks the hash is 12 hex characters, stays the same for the same args and changes for a changed arg.
    """
    benchmark_hash = paths.get_benchmark_hash(make_benchmark_args(), make_ollama_args())
    same_benchmark_hash = paths.get_benchmark_hash(make_benchmark_args(), make_ollama_args())
    changed_benchmark_hash = paths.get_benchmark_hash(make_benchmark_args(**overrides), make_ollama_args())

    assert len(benchmark_hash) == 12
    int(benchmark_hash, 16)
    assert benchmark_hash == same_benchmark_hash
    assert benchmark_hash != changed_benchmark_hash


def test_benchmark_hash_ignores_the_query_llm_timeout():
    """
    Input: two benchmark args that differ only in the query LLM timeout.
    Output: the two hashes.
    What it does: checks the timeout, which does not change the results, does not change the hash.
    """
    short_timeout = make_benchmark_args(query_args=make_query_args(query_llm_args=LLMArgs("ollama_chat/llama3.1:8b", 100, 30)))
    long_timeout = make_benchmark_args(query_args=make_query_args(query_llm_args=LLMArgs("ollama_chat/llama3.1:8b", 100, 300)))

    assert paths.get_benchmark_hash(short_timeout, make_ollama_args()) == paths.get_benchmark_hash(long_timeout, make_ollama_args())


def test_benchmark_hash_uses_ollama_context_length_only_for_an_ollama_query_llm():
    """
    Input: two Ollama context lengths, first with an Ollama query LLM, then with a Claude one.
    Output: the hashes.
    What it does: checks the context length changes the hash only when the query LLM is an Ollama model.
    """
    ollama_benchmark_args = make_benchmark_args()
    claude_benchmark_args = make_benchmark_args(query_args=make_query_args(query_llm_args=LLMArgs("anthropic/claude-haiku-5-5", 100, 30)))

    assert paths.get_benchmark_hash(ollama_benchmark_args, make_ollama_args(4096)) != paths.get_benchmark_hash(ollama_benchmark_args, make_ollama_args(32768))
    assert paths.get_benchmark_hash(claude_benchmark_args, make_ollama_args(4096)) == paths.get_benchmark_hash(claude_benchmark_args, make_ollama_args(32768))
