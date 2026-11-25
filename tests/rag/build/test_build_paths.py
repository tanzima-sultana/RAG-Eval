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



# ---------- chunks and embeddings ----------


def test_chunking_hash_is_the_hash_of_the_ingestion_hash_and_the_embedder_and_chunk_args(make_embedder_args, make_chunking_args):
    """
    Input: the default embedder and chunking args, and an ingestion hash.
    Output: the chunking hash.
    What it does: checks it is the hash of the ingestion hash, the embedding provider, the model name and the four
    chunk args.
    """
    hashed_args = chunk_args_dict()
    hashed_args["ingestion_hash"] = "testingestionhash"

    chunking_hash = paths.get_chunking_hash(make_embedder_args(), make_chunking_args(), "testingestionhash")

    assert chunking_hash == hash_of(hashed_args)


def test_chunk_path_is_named_by_the_hash_of_the_embedder_and_chunk_args(make_embedder_args, make_chunking_args):
    """
    Input: the default embedder and chunking args.
    Output: the chunk path.
    What it does: checks the folder is chunks/chunks_{hash of the embedding provider, the model name and the four
    chunk args}, with no ingestion hash, so every run with these settings shares it.
    """
    chunk_path = paths.get_chunk_path(OUTPUT_PATH, make_embedder_args(), make_chunking_args())

    assert chunk_path == os.path.join(OUTPUT_PATH, "chunks", f"chunks_{hash_of(chunk_args_dict())}")


def test_embedding_path_is_named_by_the_same_hash_as_the_chunk_path(make_embedder_args, make_chunking_args):
    """
    Input: the default embedder and chunking args.
    Output: the chunk and embedding paths.
    What it does: checks embeddings are named by the same settings as chunks, so only the prefix differs.
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
    What it does: checks every chunk arg is part of the folder name.
    """
    embedder_args = make_embedder_args()

    assert paths.get_chunk_path(OUTPUT_PATH, embedder_args, make_chunking_args()) != paths.get_chunk_path(OUTPUT_PATH, embedder_args, make_chunking_args(**overrides))


def test_chunking_hash_is_the_hash_of_the_ingestion_hash_and_the_embedder_and_chunk_args(make_embedder_args, make_chunking_args):
    """
    Input: the default embedder and chunking args, and an ingestion hash.
    Output: the chunking hash.
    What it does: checks it is the hash of the ingestion hash, the embedding provider, the model name and the four
    chunk args.
    """
    hashed_args = chunk_args_dict()
    hashed_args["ingestion_hash"] = "testingestionhash"

    chunking_hash = paths.get_chunking_hash(make_embedder_args(), make_chunking_args(), "testingestionhash")

    assert chunking_hash == hash_of(hashed_args)


@pytest.mark.parametrize("overrides", [
    {"chunk_type": "sentence"},
    {"min_chunk_size": 3},
    {"max_chunk_size": 7},
    {"fixed_chunk_overlap": 1},
])
def test_chunking_hash_changes_when_any_chunk_arg_changes(make_embedder_args, make_chunking_args, overrides):
    """
    Input: the default chunking args, and the same args with one changed.
    Output: the two chunking hashes.
    What it does: checks every chunk arg is part of the hash.
    """
    embedder_args = make_embedder_args()

    assert paths.get_chunking_hash(embedder_args, make_chunking_args(), "testingestionhash") != paths.get_chunking_hash(embedder_args, make_chunking_args(**overrides), "testingestionhash")


@pytest.mark.parametrize("overrides", [
    {"embedding_provider": "other-provider"},
    {"model_name": "other-model"},
])
def test_chunking_hash_changes_when_the_embedding_provider_or_model_changes(make_embedder_args, make_chunking_args, overrides):
    """
    Input: the default embedder args, and the same args with the provider or the model changed.
    Output: the two chunking hashes.
    What it does: checks the embedding provider and the model name are part of the hash.
    """
    chunking_args = make_chunking_args()

    assert paths.get_chunking_hash(make_embedder_args(), chunking_args, "testingestionhash") != paths.get_chunking_hash(make_embedder_args(**overrides), chunking_args, "testingestionhash")


def test_chunking_hash_changes_when_the_ingestion_hash_changes(make_embedder_args, make_chunking_args):
    """
    Input: the default embedder and chunking args, with two ingestion hashes.
    Output: the two chunking hashes.
    What it does: checks the ingestion hash is part of the hash, so a change in the input files changes the chunking hash.
    """
    embedder_args = make_embedder_args()
    chunking_args = make_chunking_args()

    assert paths.get_chunking_hash(embedder_args, chunking_args, "testingestionhash") != paths.get_chunking_hash(embedder_args, chunking_args, "otheringestionhash")


@pytest.mark.parametrize("overrides", [
    {"batch_size": 32},
    {"normalize_embeddings": False},
])
def test_chunking_hash_does_not_change_with_batch_size_or_normalize_embeddings(make_embedder_args, make_chunking_args, overrides):
    """
    Input: the default embedder args, and the same args with batch_size or normalize_embeddings changed.
    Output: the two chunking hashes.
    What it does: checks batch_size and normalize_embeddings are not part of the hash.
    """
    chunking_args = make_chunking_args()

    assert paths.get_chunking_hash(make_embedder_args(), chunking_args, "testingestionhash") == paths.get_chunking_hash(make_embedder_args(**overrides), chunking_args, "testingestionhash")


def test_embedding_hash_is_the_hash_of_the_chunking_hash_and_the_embedding_provider_and_model(make_embedder_args):
    """
    Input: the default embedder args and a chunking hash.
    Output: the embedding hash.
    What it does: checks it is the hash of the chunking hash, the embedding provider and the model name.
    """
    hashed_args = {}
    hashed_args["chunking_hash"] = "testchunkinghash"
    hashed_args["embedding_provider"] = "huggingface"
    hashed_args["model_name"] = "fake-model"

    assert paths.get_embedding_hash(make_embedder_args(), "testchunkinghash") == hash_of(hashed_args)


def test_embedding_hash_changes_when_the_chunking_hash_changes(make_embedder_args):
    """
    Input: the default embedder args, with two chunking hashes.
    Output: the two embedding hashes.
    What it does: checks the chunking hash is part of the hash, so a change upstream changes the embedding hash.
    """
    assert paths.get_embedding_hash(make_embedder_args(), "testchunkinghash") != paths.get_embedding_hash(make_embedder_args(), "otherchunkinghash")


@pytest.mark.parametrize("overrides", [
    {"embedding_provider": "other-provider"},
    {"model_name": "other-model"},
])
def test_embedding_hash_changes_when_the_embedding_provider_or_model_changes(make_embedder_args, overrides):
    """
    Input: the default embedder args, and the same args with the provider or the model changed.
    Output: the two embedding hashes.
    What it does: checks the embedding provider and the model name are part of the hash.
    """
    assert paths.get_embedding_hash(make_embedder_args(), "testchunkinghash") != paths.get_embedding_hash(make_embedder_args(**overrides), "testchunkinghash")


# ---------- indexes ----------


@pytest.mark.parametrize("faiss_type, expected_settings", [
    ("faiss_flatip", {}),
    ("faiss_ivf", {"ivf_nlist_multiplier": 1.0}),
    ("faiss_hnsw", {"hnsw_m": 4, "hnsw_ef_construction": 8}),
])
def test_faiss_hash_is_the_hash_of_the_embedding_hash_the_type_and_its_own_settings(make_faiss_ivf_args, make_faiss_hnsw_args, faiss_type, expected_settings):
    """
    Input: each FAISS type with the default args of that type.
    Output: the FAISS hash.
    What it does: checks it is the hash of the embedding hash, the FAISS type and that type's own settings.
    """
    faiss_args = None
    if faiss_type == "faiss_ivf":
        faiss_args = make_faiss_ivf_args()
    if faiss_type == "faiss_hnsw":
        faiss_args = make_faiss_hnsw_args()
    hashed_args = {}
    hashed_args["embedding_hash"] = "testembeddinghash"
    hashed_args["faiss_type"] = faiss_type
    for setting_name in expected_settings:
        hashed_args[setting_name] = expected_settings[setting_name]

    assert paths.get_faiss_hash(faiss_type, faiss_args, "testembeddinghash") == hash_of(hashed_args)


def test_faiss_hash_changes_with_the_embedding_hash_and_the_settings(make_faiss_ivf_args, make_faiss_hnsw_args):
    """
    Input: the same FAISS type with another embedding hash, and with other settings.
    Output: the FAISS hashes.
    What it does: checks the embedding hash and each type's own settings are part of its hash.
    """
    assert paths.get_faiss_hash("faiss_flatip", None, "firsthash") != paths.get_faiss_hash("faiss_flatip", None, "secondhash")
    assert paths.get_faiss_hash("faiss_ivf", make_faiss_ivf_args(ivf_nlist_multiplier=4), "testhash") != paths.get_faiss_hash("faiss_ivf", make_faiss_ivf_args(ivf_nlist_multiplier=8), "testhash")
    assert paths.get_faiss_hash("faiss_hnsw", make_faiss_hnsw_args(hnsw_m=4), "testhash") != paths.get_faiss_hash("faiss_hnsw", make_faiss_hnsw_args(hnsw_m=5), "testhash")
    assert paths.get_faiss_hash("faiss_hnsw", make_faiss_hnsw_args(hnsw_ef_construction=8), "testhash") != paths.get_faiss_hash("faiss_hnsw", make_faiss_hnsw_args(hnsw_ef_construction=9), "testhash")


def test_faiss_hash_raises_for_a_non_faiss_type():
    """
    Input: type bm25.
    Output: ValueError.
    What it does: checks only FAISS types get a FAISS hash.
    """
    with pytest.raises(ValueError, match="not a FAISS index type"):
        paths.get_faiss_hash("bm25", None, "testhash")


def test_faiss_index_path_is_named_by_its_type_and_hash():
    """
    Input: a FAISS type and a FAISS hash.
    Output: the index path.
    What it does: checks the folder is indexes/{faiss_type}_{faiss_hash}.
    """
    assert paths.get_faiss_index_path(OUTPUT_PATH, "faiss_ivf", "abc123") == os.path.join(OUTPUT_PATH, "indexes", "faiss_ivf_abc123")


def test_bm25_hash_is_the_hash_of_the_chunking_hash_and_the_bm25_settings(make_bm25_indexing_args):
    """
    Input: the default BM25 args and a chunking hash.
    Output: the BM25 hash.
    What it does: checks it is the hash of the chunking hash, bm25_k1, bm25_b and bm25_tokenizer, with no embedding hash.
    """
    hashed_args = {}
    hashed_args["chunking_hash"] = "testchunkinghash"
    hashed_args["bm25_k1"] = 1.5
    hashed_args["bm25_b"] = 0.75
    hashed_args["bm25_tokenizer"] = "whitespace"

    assert paths.get_bm25_hash(make_bm25_indexing_args(), "testchunkinghash") == hash_of(hashed_args)


@pytest.mark.parametrize("overrides", [
    {"bm25_k1": 1.2},
    {"bm25_b": 0.5},
    {"bm25_tokenizer": "other"},
])
def test_bm25_hash_changes_with_the_chunking_hash_and_each_setting(make_bm25_indexing_args, overrides):
    """
    Input: the default BM25 args, with another chunking hash, and with one setting changed.
    Output: the BM25 hashes.
    What it does: checks the chunking hash and each BM25 setting are part of its hash.
    """
    assert paths.get_bm25_hash(make_bm25_indexing_args(), "firsthash") != paths.get_bm25_hash(make_bm25_indexing_args(), "secondhash")
    assert paths.get_bm25_hash(make_bm25_indexing_args(), "testhash") != paths.get_bm25_hash(make_bm25_indexing_args(**overrides), "testhash")


def test_bm25_index_path_is_named_by_its_hash():
    """
    Input: a BM25 hash.
    Output: the index path.
    What it does: checks the folder is indexes/bm25_{bm25_hash}.
    """
    assert paths.get_bm25_index_path(OUTPUT_PATH, "abc123") == os.path.join(OUTPUT_PATH, "indexes", "bm25_abc123")


# ---------- Qdrant ----------


def test_vectordb_hash_is_the_hash_of_the_root_folder_the_dataset_the_type_and_the_embedding_hash():
    """
    Input: the output path out/dataset, the type qdrant and an embedding hash.
    Output: the vector DB hash.
    What it does: checks it is the hash of the root folder name, the dataset name, the vector DB type and the embedding hash.
    """
    hashed_args = {}
    hashed_args["root_folder_name"] = "out"
    hashed_args["dataset_name"] = "dataset"
    hashed_args["vectordb_type"] = "qdrant"
    hashed_args["embedding_hash"] = "testembeddinghash"

    assert paths.get_vectordb_hash("out/dataset", "qdrant", "testembeddinghash") == hash_of(hashed_args)


def test_vectordb_hash_changes_with_the_root_folder_and_the_embedding_hash():
    """
    Input: the same dataset under the out/ and temp/ root folders, and with two embedding hashes.
    Output: the vector DB hashes.
    What it does: checks a temp run and an outputs run never share a collection, and a new embedding hash gives a new one.
    """
    assert paths.get_vectordb_hash("out/dataset", "qdrant", "testhash") != paths.get_vectordb_hash("temp/dataset", "qdrant", "testhash")
    assert paths.get_vectordb_hash("out/dataset", "qdrant", "firsthash") != paths.get_vectordb_hash("out/dataset", "qdrant", "secondhash")


def test_qdrant_collection_name_is_the_type_and_the_hash():
    """
    Input: the type qdrant and a vector DB hash.
    Output: the collection name.
    What it does: checks the name is qdrant_{vectordb_hash}.
    """
    assert paths.get_qdrant_collection_name("qdrant", "abc123") == "qdrant_abc123"


# ---------- build manifest ----------


def test_build_hash_is_the_hash_of_the_six_layer_hashes():
    """
    Input: six layer hashes.
    Output: the build hash.
    What it does: checks it is the hash of the ingestion, chunking, embedding, BM25, FAISS and vector DB hashes.
    """
    hashed_args = {}
    hashed_args["ingestion_hash"] = "ingestionhash"
    hashed_args["chunking_hash"] = "chunkinghash"
    hashed_args["embedding_hash"] = "embeddinghash"
    hashed_args["bm25_hash"] = "bm25hash"
    hashed_args["faiss_hash"] = "faisshash"
    hashed_args["vectordb_hash"] = ""

    build_hash = paths.get_build_hash("ingestionhash", "chunkinghash", "embeddinghash", "bm25hash", "faisshash", "")

    assert build_hash == hash_of(hashed_args)


@pytest.mark.parametrize("position", [0, 1, 2, 3, 4, 5])
def test_build_hash_changes_when_any_layer_hash_changes(position):
    """
    Input: the six layer hashes, and the same with one of them changed.
    Output: the two build hashes.
    What it does: checks every layer's hash is part of the build hash.
    """
    layer_hashes = ["ingestionhash", "chunkinghash", "embeddinghash", "bm25hash", "faisshash", "vectordbhash"]
    changed_layer_hashes = list(layer_hashes)
    changed_layer_hashes[position] = "otherhash"

    assert paths.get_build_hash(*layer_hashes) != paths.get_build_hash(*changed_layer_hashes)


def test_build_manifest_path_is_named_by_the_dataset_its_size_and_the_hash_of_the_build_hashes():
    """
    Input: a dataset name, a size and a list of build hashes.
    Output: the manifest path.
    What it does: checks the file is build/{dataset_name}_{dataset_size}_{hash of the list of build hashes}.json.
    """
    manifest_path = paths.get_build_manifest_path(OUTPUT_PATH, "dataset", 15, ["firsthash", "secondhash"])

    expected_hash = hashlib.sha256(json.dumps(["firsthash", "secondhash"]).encode("utf-8")).hexdigest()[:12]
    assert manifest_path == os.path.join(OUTPUT_PATH, "build", f"dataset_15_{expected_hash}.json")


def test_build_manifest_path_changes_with_the_set_and_the_order_of_build_hashes():
    """
    Input: lists of build hashes with one more hash, and in another order.
    Output: the manifest paths.
    What it does: checks another set of setups, or the same setups in another order, gives another file name.
    """
    two_hashes_path = paths.get_build_manifest_path(OUTPUT_PATH, "dataset", 15, ["firsthash", "secondhash"])

    assert two_hashes_path != paths.get_build_manifest_path(OUTPUT_PATH, "dataset", 15, ["firsthash", "secondhash", "thirdhash"])
    assert two_hashes_path != paths.get_build_manifest_path(OUTPUT_PATH, "dataset", 15, ["secondhash", "firsthash"])
    assert two_hashes_path == paths.get_build_manifest_path(OUTPUT_PATH, "dataset", 15, ["firsthash", "secondhash"])
