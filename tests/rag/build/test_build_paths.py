"""
test_build_paths.py

Tests for the build path functions in common.paths: each output is named by a prefix and the first 12 hex
characters of the SHA-256 of the args in its path.
"""

import argparse
import hashlib
import json
import os

import pytest

from common import paths

OUTPUT_PATH = "out/dataset"


def hash_of(hashed_args):
    """
    Input: hashed_args, a dict of arg name to value.
    Output: the first 12 hex characters of the SHA-256 of the dict as JSON with sorted keys.
    What it does: computes the expected hash the same way the path functions document it.
    """
    return hashlib.sha256(json.dumps(hashed_args, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def chunk_args_dict():
    """
    Input: none.
    Output: the embedder and chunk args of the make_embedder_args and make_chunking_args defaults as a dict.
    What it does: lists the six args every build path hashes: the embedding provider, the model name and the four
    chunk args.
    """
    chunk_args = {}
    chunk_args["embedding_provider"] = "huggingface"
    chunk_args["model_name"] = "fake-model"
    chunk_args["chunk_type"] = "fixed"
    chunk_args["min_chunk_size"] = 2
    chunk_args["max_chunk_size"] = 6
    chunk_args["fixed_chunk_overlap"] = 2
    return chunk_args


def make_parsed_build_args(**overrides):
    """
    Input: any build arg as a keyword argument.
    Output: an argparse.Namespace with small build defaults plus the overrides.
    What it does: stands in for the parsed build config.
    """
    arg_values = {}
    arg_values["dataset_size"] = 15
    arg_values["dataset_name"] = "dataset"
    arg_values["input_path"] = "data/dataset/"
    arg_values["output_path"] = "out/"
    arg_values["document_type"] = "pdf"
    arg_values["embedding_provider"] = "huggingface"
    arg_values["model_name"] = "fake-model"
    arg_values["batch_size"] = 256
    arg_values["normalize_embeddings"] = "true"
    arg_values["chunk_types"] = ["fixed", "sentence"]
    arg_values["min_chunk_size"] = 2
    arg_values["max_chunk_size"] = 6
    arg_values["fixed_chunk_overlap"] = [2]
    arg_values["index_types"] = ["faiss_flatip", "bm25"]
    arg_values["ivf_nlist_multiplier"] = [1.0]
    arg_values["hnsw_m"] = 4
    arg_values["hnsw_ef_construction"] = 8
    arg_values["bm25_k1"] = 1.5
    arg_values["bm25_b"] = 0.75
    arg_values["bm25_tokenizer"] = "whitespace"
    arg_values["vectordb_types"] = ["qdrant"]
    arg_values["qdrant_url"] = "http://localhost:6333"
    arg_values["qdrant_batch_size"] = 3
    for arg_name in overrides:
        arg_values[arg_name] = overrides[arg_name]
    return argparse.Namespace(**arg_values)


# ---------- chunks and embeddings ----------


def test_chunk_path_is_named_by_the_hash_of_the_embedder_and_chunk_args(make_embedder_args, make_chunking_args):
    """
    Input: the default embedder and chunking args.
    Output: the chunk path.
    What it does: checks the folder is chunks/chunks_{hash of the embedding provider, the model name and the four
    chunk args}.
    """
    chunk_path = paths.get_chunk_path(OUTPUT_PATH, make_embedder_args(), make_chunking_args())

    assert chunk_path == os.path.join(OUTPUT_PATH, "chunks", f"chunks_{hash_of(chunk_args_dict())}")


def test_embedding_path_has_its_own_prefix_and_the_same_hash_as_its_chunks(make_embedder_args, make_chunking_args):
    """
    Input: the default embedder and chunking args.
    Output: the chunk and embedding paths.
    What it does: checks embeddings hash the same args as chunks, so only the prefix differs.
    """
    embedder_args = make_embedder_args()
    chunking_args = make_chunking_args()

    chunk_path = paths.get_chunk_path(OUTPUT_PATH, embedder_args, chunking_args)
    embedding_path = paths.get_embedding_path(OUTPUT_PATH, embedder_args, chunking_args)

    assert embedding_path == os.path.join(OUTPUT_PATH, "embeddings", f"embeddings_{hash_of(chunk_args_dict())}")
    assert os.path.basename(chunk_path)[len("chunks_"):] == os.path.basename(embedding_path)[len("embeddings_"):]


@pytest.mark.parametrize("overrides", [
    {"chunk_type": "sentence"},
    {"min_chunk_size": 3},
    {"max_chunk_size": 7},
    {"fixed_chunk_overlap": 1},
])
def test_chunk_path_changes_when_any_chunk_arg_changes(make_embedder_args, make_chunking_args, overrides):
    """
    Input: the default chunking args, and the same args with one changed.
    Output: the two chunk paths.
    What it does: checks every chunk arg is part of the hash.
    """
    embedder_args = make_embedder_args()

    assert paths.get_chunk_path(OUTPUT_PATH, embedder_args, make_chunking_args()) != paths.get_chunk_path(OUTPUT_PATH, embedder_args, make_chunking_args(**overrides))


@pytest.mark.parametrize("overrides", [
    {"embedding_provider": "other-provider"},
    {"model_name": "other-model"},
])
def test_chunk_path_changes_when_the_embedding_provider_or_model_changes(make_embedder_args, make_chunking_args, overrides):
    """
    Input: the default embedder args, and the same args with the provider or the model changed.
    Output: the two chunk paths.
    What it does: checks the embedding provider and the model name are part of the hash.
    """
    chunking_args = make_chunking_args()

    assert paths.get_chunk_path(OUTPUT_PATH, make_embedder_args(), chunking_args) != paths.get_chunk_path(OUTPUT_PATH, make_embedder_args(**overrides), chunking_args)


@pytest.mark.parametrize("overrides", [
    {"batch_size": 32},
    {"normalize_embeddings": False},
])
def test_chunk_path_does_not_change_with_batch_size_or_normalize_embeddings(make_embedder_args, make_chunking_args, overrides):
    """
    Input: the default embedder args, and the same args with batch_size or normalize_embeddings changed.
    Output: the two chunk paths.
    What it does: checks batch_size and normalize_embeddings are not part of the hash.
    """
    chunking_args = make_chunking_args()

    assert paths.get_chunk_path(OUTPUT_PATH, make_embedder_args(), chunking_args) == paths.get_chunk_path(OUTPUT_PATH, make_embedder_args(**overrides), chunking_args)


# ---------- indexes ----------


@pytest.mark.parametrize("index_type, index_settings", [
    ("faiss_flatip", {}),
    ("faiss_ivf", {"ivf_nlist_multiplier": 1.0}),
    ("faiss_hnsw", {"hnsw_m": 4, "hnsw_ef_construction": 8}),
])
def test_faiss_index_path_is_named_by_its_type_and_the_hash_of_its_args(make_embedder_args, make_chunking_args, make_index_args, index_type, index_settings):
    """
    Input: one FAISS index type with the default settings.
    Output: the index path.
    What it does: checks the folder is indexes/{index_type}_{hash of index type, embedder and chunk args, its own
    settings and the input hash}.
    """
    hashed_args = chunk_args_dict()
    hashed_args["index_type"] = index_type
    hashed_args["input_hash"] = "testinputhash"
    for setting_name in index_settings:
        hashed_args[setting_name] = index_settings[setting_name]

    faiss_index_path = paths.get_faiss_index_path(OUTPUT_PATH, make_embedder_args(), make_chunking_args(), make_index_args(index_type=index_type))

    assert faiss_index_path == os.path.join(OUTPUT_PATH, "indexes", f"{index_type}_{hash_of(hashed_args)}")


def test_faiss_ivf_index_path_changes_with_the_ivf_nlist_multiplier(make_embedder_args, make_chunking_args, make_index_args):
    """
    Input: faiss_ivf index args with ivf_nlist_multiplier 4 and 8.
    Output: the two index paths.
    What it does: checks each multiplier gets its own IVF index folder.
    """
    embedder_args = make_embedder_args()
    chunking_args = make_chunking_args()

    four_path = paths.get_faiss_index_path(OUTPUT_PATH, embedder_args, chunking_args, make_index_args(index_type="faiss_ivf", ivf_nlist_multiplier=4))
    eight_path = paths.get_faiss_index_path(OUTPUT_PATH, embedder_args, chunking_args, make_index_args(index_type="faiss_ivf", ivf_nlist_multiplier=8))

    assert four_path != eight_path


def test_faiss_index_path_raises_for_a_non_faiss_type(make_embedder_args, make_chunking_args, make_index_args):
    """
    Input: index type bm25.
    Output: ValueError.
    What it does: checks only FAISS index types get a FAISS path.
    """
    with pytest.raises(ValueError, match="not a FAISS index type"):
        paths.get_faiss_index_path(OUTPUT_PATH, make_embedder_args(), make_chunking_args(), make_index_args(index_type="bm25"))


def test_bm25_index_path_is_named_by_the_hash_of_the_chunk_and_bm25_args(make_embedder_args, make_chunking_args, make_index_args):
    """
    Input: index type bm25 with the default settings.
    Output: the index path.
    What it does: checks the folder is indexes/bm25_{hash of the embedder and chunk args, bm25_k1, bm25_b,
    bm25_tokenizer and the input hash}.
    """
    hashed_args = chunk_args_dict()
    hashed_args["input_hash"] = "testinputhash"
    hashed_args["bm25_k1"] = 1.5
    hashed_args["bm25_b"] = 0.75
    hashed_args["bm25_tokenizer"] = "whitespace"

    bm25_index_path = paths.get_bm25_index_path(OUTPUT_PATH, make_embedder_args(), make_chunking_args(), make_index_args(index_type="bm25"))

    assert bm25_index_path == os.path.join(OUTPUT_PATH, "indexes", f"bm25_{hash_of(hashed_args)}")


def test_index_path_changes_when_its_chunk_args_change(make_embedder_args, make_chunking_args, make_index_args):
    """
    Input: the default chunk args, and other chunk args, with the same index args.
    Output: the FAISS and BM25 paths.
    What it does: checks an index is named by the chunks it is built from.
    """
    embedder_args = make_embedder_args()
    chunking_args = make_chunking_args()
    other_chunking_args = make_chunking_args(max_chunk_size=7)
    faiss_index_args = make_index_args()
    bm25_index_args = make_index_args(index_type="bm25")

    assert paths.get_faiss_index_path(OUTPUT_PATH, embedder_args, chunking_args, faiss_index_args) != paths.get_faiss_index_path(OUTPUT_PATH, embedder_args, other_chunking_args, faiss_index_args)
    assert paths.get_bm25_index_path(OUTPUT_PATH, embedder_args, chunking_args, bm25_index_args) != paths.get_bm25_index_path(OUTPUT_PATH, embedder_args, other_chunking_args, bm25_index_args)


def test_index_and_qdrant_names_change_with_the_input_hash(make_embedder_args, make_chunking_args, make_index_args, make_vectordb_args):
    """
    Input: the default index and vector DB args, and the same with another input hash.
    Output: the FAISS, BM25 and Qdrant names.
    What it does: checks each input set gets its own indexes and collection.
    """
    embedder_args = make_embedder_args()
    chunking_args = make_chunking_args()

    assert paths.get_faiss_index_path(OUTPUT_PATH, embedder_args, chunking_args, make_index_args()) != paths.get_faiss_index_path(OUTPUT_PATH, embedder_args, chunking_args, make_index_args(input_hash="otherinputhash"))
    assert paths.get_bm25_index_path(OUTPUT_PATH, embedder_args, chunking_args, make_index_args(index_type="bm25")) != paths.get_bm25_index_path(OUTPUT_PATH, embedder_args, chunking_args, make_index_args(index_type="bm25", input_hash="otherinputhash"))
    assert paths.get_qdrant_collection_name(OUTPUT_PATH, embedder_args, chunking_args, make_vectordb_args()) != paths.get_qdrant_collection_name(OUTPUT_PATH, embedder_args, chunking_args, make_vectordb_args(input_hash="otherinputhash"))


# ---------- Qdrant ----------


def test_qdrant_collection_name_changes_with_the_root_folder(make_embedder_args, make_chunking_args, make_vectordb_args):
    """
    Input: the same dataset under the out/ and temp/ root folders.
    Output: the two collection names.
    What it does: checks the name is qdrant_{hash}, and a temp run and an outputs run never share a collection.
    """
    embedder_args = make_embedder_args()
    chunking_args = make_chunking_args()

    out_collection_name = paths.get_qdrant_collection_name("out/dataset", embedder_args, chunking_args, make_vectordb_args())
    temp_collection_name = paths.get_qdrant_collection_name("temp/dataset", embedder_args, chunking_args, make_vectordb_args())

    assert out_collection_name.startswith("qdrant_")
    assert len(out_collection_name) == len("qdrant_") + 12
    assert out_collection_name != temp_collection_name


# ---------- build manifest ----------


def test_build_manifest_path_is_named_by_the_dataset_and_the_hash_of_all_the_build_args():
    """
    Input: the default parsed build args.
    Output: the manifest path.
    What it does: checks the file is build/manifest_{dataset_name}_{dataset_size}_{hash of every build arg and the
    input hash}.json.
    """
    parsed_args = make_parsed_build_args()
    hashed_args = dict(vars(parsed_args))
    hashed_args["input_hash"] = "testinputhash"

    manifest_path = paths.get_build_manifest_path(OUTPUT_PATH, parsed_args, "testinputhash")

    assert manifest_path == os.path.join(OUTPUT_PATH, "build", f"manifest_dataset_15_{hash_of(hashed_args)}.json")
    assert paths.get_build_hash(parsed_args, "testinputhash") == hash_of(hashed_args)


@pytest.mark.parametrize("overrides", [
    {"dataset_size": 20},
    {"input_path": "data/other/"},
    {"model_name": "other-model"},
    {"chunk_types": ["fixed"]},
    {"fixed_chunk_overlap": [2, 4]},
    {"index_types": ["bm25"]},
    {"ivf_nlist_multiplier": [4.0]},
    {"hnsw_m": 5},
    {"qdrant_batch_size": 4},
])
def test_build_hash_changes_when_any_build_arg_changes(overrides):
    """
    Input: the default parsed build args, and the same args with one changed.
    Output: the two build hashes.
    What it does: checks a change to any build arg gives a new manifest name.
    """
    assert paths.get_build_hash(make_parsed_build_args(), "testinputhash") != paths.get_build_hash(make_parsed_build_args(**overrides), "testinputhash")


def test_build_hash_changes_when_the_input_hash_changes():
    """
    Input: the same parsed build args with two input hashes.
    Output: the two build hashes.
    What it does: checks changed input files give a new manifest name.
    """
    assert paths.get_build_hash(make_parsed_build_args(), "testinputhash") != paths.get_build_hash(make_parsed_build_args(), "otherinputhash")
