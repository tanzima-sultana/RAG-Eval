"""
test_indexing.py

Tests for stages.build.indexing.indexing.Indexing: validating the args, choosing the
index of index_type, and handing the work to Langchain for framework
"langchain". Qdrant runs in memory and HuggingFaceEmbeddings is faked.
"""

import os

import pytest

from common import paths
from stages.build.indexing.indexing import Indexing
from stages.build.indexing.schema import FaissIndex
from stages.build.indexing.schema import Bm25Index
from stages.build.indexing.schema import QdrantIndex
from stages.build.indexing.schema import LangchainFaissIndex
from stages.build.indexing.schema import LangchainBm25Index
from stages.build.indexing.schema import LangchainQdrantIndex


def test_init_with_invalid_args_raises(make_index_args, make_chunking_args):
    """
    Input: index_type "annoy".
    Output: ValueError mentioning index_type.
    What it does: checks Indexing validates its args before anything else.
    """
    with pytest.raises(ValueError, match="index_type"):
        Indexing(make_index_args(index_type="annoy"), make_chunking_args())


def test_init_creates_langchain_only_for_langchain_framework(fake_huggingface_embeddings, make_index_args, make_chunking_args):
    """
    Input: framework "" and framework "langchain".
    Output: the langchain attribute of each Indexing.
    What it does: checks the Langchain class is only created for framework "langchain".
    """
    built_in_indexing = Indexing(make_index_args(), make_chunking_args())
    langchain_indexing = Indexing(make_index_args(), make_chunking_args(framework="langchain"))

    assert built_in_indexing.langchain is None
    assert langchain_indexing.langchain is not None


@pytest.mark.parametrize("index_type", ["faiss_flatip", "faiss_ivf", "faiss_hnsw"])
def test_compute_index_builds_faiss_in_its_folder(tmp_path, make_index_args, make_chunking_args, make_chunks_and_embeddings, index_type):
    """
    Input: each FAISS index type.
    Output: the index compute_index returns and the FAISS index folder.
    What it does: checks a FaissIndex of that type is built and saved under get_faiss_index_path.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    output_path = str(tmp_path / "myset")
    index_args = make_index_args(index_type=index_type)
    chunking_args = make_chunking_args()

    index = Indexing(index_args, chunking_args).compute_index(chunks, embeddings, output_path)

    assert isinstance(index, FaissIndex)
    assert index.faiss_index_type == index_type
    faiss_index_path = paths.get_faiss_index_path(output_path, chunking_args, index_args)
    assert os.path.exists(os.path.join(faiss_index_path, "index.faiss"))


def test_compute_index_builds_bm25_in_its_folder(tmp_path, make_index_args, make_chunking_args, make_chunks_and_embeddings):
    """
    Input: index_type "bm25".
    Output: the index compute_index returns and the BM25 index folder.
    What it does: checks a Bm25Index is built and saved under get_bm25_index_path.
    """
    chunks, embeddings = make_chunks_and_embeddings(6)
    output_path = str(tmp_path / "myset")
    index_args = make_index_args(index_type="bm25")
    chunking_args = make_chunking_args()

    index = Indexing(index_args, chunking_args).compute_index(chunks, embeddings, output_path)

    assert isinstance(index, Bm25Index)
    bm25_index_path = paths.get_bm25_index_path(output_path, chunking_args, index_args)
    assert os.path.exists(os.path.join(bm25_index_path, "bm25.pkl"))


def test_compute_index_builds_qdrant_collection_by_name(tmp_path, memory_qdrant_client, make_index_args, make_chunking_args, make_chunks_and_embeddings):
    """
    Input: index_type "qdrant".
    Output: the index compute_index returns.
    What it does: checks a QdrantIndex is built in the collection named by get_qdrant_collection_name.
    """
    chunks, embeddings = make_chunks_and_embeddings(5)
    output_path = str(tmp_path / "myset")
    chunking_args = make_chunking_args()

    index = Indexing(make_index_args(index_type="qdrant"), chunking_args).compute_index(chunks, embeddings, output_path)

    assert isinstance(index, QdrantIndex)
    assert index.collection_name == paths.get_qdrant_collection_name(output_path, chunking_args)
    assert memory_qdrant_client.collection_exists(index.collection_name)


@pytest.mark.parametrize("index_type, index_class", [
    ("faiss_flatip", LangchainFaissIndex),
    ("faiss_ivf", LangchainFaissIndex),
    ("faiss_hnsw", LangchainFaissIndex),
    ("bm25", LangchainBm25Index),
    ("qdrant", LangchainQdrantIndex),
])
def test_compute_index_builds_langchain_index(tmp_path, memory_qdrant_client, fake_huggingface_embeddings, make_index_args, make_chunking_args, make_chunks_and_embeddings, index_type, index_class):
    """
    Input: framework "langchain" and each index type.
    Output: the index compute_index returns.
    What it does: checks the LangChain index of that type is built.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    output_path = str(tmp_path / "myset")

    index = Indexing(make_index_args(index_type=index_type), make_chunking_args(framework="langchain")).compute_index(chunks, embeddings, output_path)

    assert isinstance(index, index_class)


def test_built_in_and_langchain_indexes_use_separate_folders(tmp_path, make_index_args, make_chunking_args):
    """
    Input: the same index args with framework "" and "langchain".
    Output: the two FAISS and BM25 index folders.
    What it does: checks the two frameworks never share a cache.
    """
    output_path = str(tmp_path / "myset")
    faiss_index_args = make_index_args()
    bm25_index_args = make_index_args(index_type="bm25")
    built_in_chunking_args = make_chunking_args()
    langchain_chunking_args = make_chunking_args(framework="langchain")

    built_in_faiss_path = paths.get_faiss_index_path(output_path, built_in_chunking_args, faiss_index_args)
    langchain_faiss_path = paths.get_faiss_index_path(output_path, langchain_chunking_args, faiss_index_args)
    built_in_bm25_path = paths.get_bm25_index_path(output_path, built_in_chunking_args, bm25_index_args)
    langchain_bm25_path = paths.get_bm25_index_path(output_path, langchain_chunking_args, bm25_index_args)

    assert built_in_faiss_path != langchain_faiss_path
    assert built_in_bm25_path != langchain_bm25_path
