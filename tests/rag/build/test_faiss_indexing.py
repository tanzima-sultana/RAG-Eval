"""
test_faiss_indexing.py

Tests for rag.build.indexing.faiss_indexing. HuggingFaceEmbeddings is replaced with a fake, so no model is downloaded.
FAISS is real.
"""

import os

import pytest

from common import paths
from rag.build.indexing.faiss_indexing import FaissIndexing
from rag.build.indexing.faiss_indexing import load_faiss_index_from_path
from rag.build.indexing.faiss_indexing import load_saved_faiss_index
from schema.index import FaissIndex

EMBEDDING_HASH = "testembeddinghash"


@pytest.fixture
def make_faiss_indexing(fake_huggingface_embeddings):
    """
    Input: the fake_huggingface_embeddings fixture.
    Output: a function that builds a FaissIndexing.
    What it does: returns a builder that creates FaissIndexing with the fake embedding model and a fixed embedding hash.
    """
    def build_faiss_indexing():
        """
        Input: none.
        Output: a FaissIndexing.
        What it does: creates the FaissIndexing.
        """
        return FaissIndexing(EMBEDDING_HASH, fake_huggingface_embeddings())

    return build_faiss_indexing


@pytest.fixture
def build_faiss_index(make_faiss_indexing, make_faiss_ivf_args, make_faiss_hnsw_args):
    """
    Input: the make_faiss_indexing, make_faiss_ivf_args and make_faiss_hnsw_args fixtures.
    Output: a function that builds the FAISS index of a type.
    What it does: returns a builder that calls the indexing method of the FAISS type with the default args of that type.
    """
    def build_index(faiss_type, chunks, embeddings, output_path):
        """
        Input: faiss_type; chunks; embeddings; output_path.
        Output: (faiss_index, faiss_hash, faiss_index_path) of that FAISS type.
        What it does: calls the indexing method of the FAISS type with the default IVF or HNSW args.
        """
        faiss_indexing = make_faiss_indexing()
        if faiss_type == "faiss_flatip":
            return faiss_indexing.faiss_flatip_indexing(chunks, embeddings, output_path)
        if faiss_type == "faiss_ivf":
            return faiss_indexing.faiss_ivf_indexing(chunks, embeddings, output_path, make_faiss_ivf_args())
        return faiss_indexing.faiss_hnsw_indexing(chunks, embeddings, output_path, make_faiss_hnsw_args())

    return build_index


# ---------- validate ----------


@pytest.mark.parametrize("overrides, message", [
    ({"ivf_nlist_multiplier": 0}, "ivf_nlist_multiplier"),
    ({"ivf_nlist_multiplier": -1}, "ivf_nlist_multiplier"),
])
def test_faiss_ivf_indexing_raises_for_invalid_args(make_faiss_indexing, make_faiss_ivf_args, make_chunks_and_embeddings, tmp_path, overrides, message):
    """
    Input: FaissIVFArgs with a multiplier that is not greater than 0.
    Output: ValueError.
    What it does: checks the IVF args are validated when the index is built, and the setting is named.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)

    with pytest.raises(ValueError, match=message):
        make_faiss_indexing().faiss_ivf_indexing(chunks, embeddings, str(tmp_path), make_faiss_ivf_args(**overrides))


@pytest.mark.parametrize("overrides, message", [
    ({"hnsw_m": 0}, "hnsw_m"),
    ({"hnsw_ef_construction": 0}, "hnsw_ef_construction"),
])
def test_faiss_hnsw_indexing_raises_for_invalid_args(make_faiss_indexing, make_faiss_hnsw_args, make_chunks_and_embeddings, tmp_path, overrides, message):
    """
    Input: FaissHNSWArgs with a value that is not greater than 0.
    Output: ValueError.
    What it does: checks the HNSW args are validated when the index is built, and the setting is named.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)

    with pytest.raises(ValueError, match=message):
        make_faiss_indexing().faiss_hnsw_indexing(chunks, embeddings, str(tmp_path), make_faiss_hnsw_args(**overrides))


# ---------- build and save ----------


@pytest.mark.parametrize("faiss_type", ["faiss_flatip", "faiss_ivf", "faiss_hnsw"])
def test_faiss_index_builds_and_saves(build_faiss_index, make_faiss_ivf_args, make_faiss_hnsw_args, make_chunks_and_embeddings, tmp_path, faiss_type):
    """
    Input: 10 chunks and each FAISS type, with no saved index.
    Output: the FaissIndex, its hash, its path and the files in its folder.
    What it does: checks every chunk is in the vector store with its text and chunk_id, the hash and path are the ones
    get_faiss_hash and get_faiss_index_path give, and the store is saved in that folder.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)

    faiss_index, faiss_hash, faiss_index_path = build_faiss_index(faiss_type, chunks, embeddings, str(tmp_path))

    assert faiss_index.index_type == faiss_type
    assert faiss_index.vector_store.index.ntotal == 10
    assert faiss_index.dimension == 4
    document = faiss_index.vector_store.docstore.search("doc0_2")
    assert document.page_content == chunks["doc0_2"].chunk_texts
    assert document.metadata["chunk_id"] == "doc0_2"

    faiss_args = None
    if faiss_type == "faiss_ivf":
        faiss_args = make_faiss_ivf_args()
    if faiss_type == "faiss_hnsw":
        faiss_args = make_faiss_hnsw_args()
    expected_hash = paths.get_faiss_hash(faiss_type, faiss_args, EMBEDDING_HASH)
    expected_path = paths.get_faiss_index_path(str(tmp_path), faiss_type, expected_hash)
    assert faiss_hash == expected_hash
    assert faiss_index_path == expected_path
    assert faiss_index.index_path == expected_path
    assert os.path.exists(os.path.join(expected_path, "index.faiss"))
    assert os.path.exists(os.path.join(expected_path, "index.pkl"))


def test_faiss_search_with_a_chunk_vector_finds_that_chunk(make_faiss_indexing, make_chunks_and_embeddings, tmp_path):
    """
    Input: 10 chunks in a flatip index, searched with chunk "doc1_3"'s vector.
    Output: the top hit and its score.
    What it does: checks vectors are stored at the right positions and the score is cosine similarity.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    faiss_index, faiss_hash, faiss_index_path = make_faiss_indexing().faiss_flatip_indexing(chunks, embeddings, str(tmp_path))

    documents_and_scores = faiss_index.vector_store.similarity_search_with_score_by_vector(
        embeddings["doc1_3"].embeddings.tolist(), k=1
    )

    document, score = documents_and_scores[0]
    assert document.metadata["chunk_id"] == "doc1_3"
    assert score == pytest.approx(1.0, abs=1e-5)


# ---------- cache ----------


@pytest.mark.parametrize("faiss_type", ["faiss_flatip", "faiss_ivf", "faiss_hnsw"])
def test_faiss_index_loads_from_cache_on_second_run(build_faiss_index, make_chunks_and_embeddings, tmp_path, capsys, faiss_type):
    """
    Input: the same chunks indexed twice with each FAISS type.
    Output: what the second run printed and its index.
    What it does: checks the second run loads the saved index and does not build a new one.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    build_faiss_index(faiss_type, chunks, embeddings, str(tmp_path))
    capsys.readouterr()

    faiss_index, faiss_hash, faiss_index_path = build_faiss_index(faiss_type, chunks, embeddings, str(tmp_path))

    printed = capsys.readouterr().out
    assert "loaded from cache : 10 vectors" in printed
    assert "built" not in printed
    assert faiss_index.vector_store.index.ntotal == 10


def test_faiss_index_is_rebuilt_when_chunks_change(make_faiss_indexing, make_chunks_and_embeddings, tmp_path, capsys):
    """
    Input: 10 chunks indexed, then 12 chunks indexed with the same settings and embedding hash.
    Output: what the second run printed and its index.
    What it does: checks a saved index with different chunk ids is replaced by a new one.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    make_faiss_indexing().faiss_flatip_indexing(chunks, embeddings, str(tmp_path))
    capsys.readouterr()

    more_chunks, more_embeddings = make_chunks_and_embeddings(12)
    faiss_index, faiss_hash, faiss_index_path = make_faiss_indexing().faiss_flatip_indexing(more_chunks, more_embeddings, str(tmp_path))

    assert "FAISS flatip index built : 12 vectors" in capsys.readouterr().out
    assert faiss_index.vector_store.index.ntotal == 12


def test_faiss_ivf_indexes_with_different_multipliers_get_different_folders(make_faiss_indexing, make_faiss_ivf_args, make_chunks_and_embeddings, tmp_path):
    """
    Input: 100 chunks indexed with faiss_ivf and multiplier 0.1, then 0.2.
    Output: the two FaissIndex, with their hashes and paths.
    What it does: checks each multiplier has its own hash and folder, so one does not replace the other.
    """
    chunks, embeddings = make_chunks_and_embeddings(100)
    faiss_indexing = make_faiss_indexing()

    first_index, first_hash, first_path = faiss_indexing.faiss_ivf_indexing(chunks, embeddings, str(tmp_path), make_faiss_ivf_args(ivf_nlist_multiplier=0.1))
    second_index, second_hash, second_path = faiss_indexing.faiss_ivf_indexing(chunks, embeddings, str(tmp_path), make_faiss_ivf_args(ivf_nlist_multiplier=0.2))

    assert first_hash != second_hash
    assert first_path != second_path
    assert first_index.index_path != second_index.index_path
    assert os.path.exists(first_index.index_path)
    assert os.path.exists(second_index.index_path)


# ---------- nlist ----------


def test_faiss_ivf_nlist_is_multiplier_times_sqrt_of_chunks(make_faiss_indexing, make_faiss_ivf_args, make_chunks_and_embeddings, tmp_path):
    """
    Input: 400 chunks with ivf_nlist_multiplier 0.1.
    Output: the FAISS IVF index.
    What it does: checks nlist is round(0.1 * sqrt(400)) = 2, which is under the cap 400 // 39 = 10.
    """
    chunks, embeddings = make_chunks_and_embeddings(400)

    faiss_index, faiss_hash, faiss_index_path = make_faiss_indexing().faiss_ivf_indexing(chunks, embeddings, str(tmp_path), make_faiss_ivf_args(ivf_nlist_multiplier=0.1))

    assert faiss_index.vector_store.index.nlist == 2


def test_faiss_ivf_nlist_is_capped_at_chunks_divided_by_39(make_faiss_indexing, make_faiss_ivf_args, make_chunks_and_embeddings, tmp_path):
    """
    Input: 100 chunks with ivf_nlist_multiplier 4.
    Output: the FAISS IVF index.
    What it does: checks round(4 * sqrt(100)) = 40 is capped silently at 100 // 39 = 2.
    """
    chunks, embeddings = make_chunks_and_embeddings(100)

    faiss_index, faiss_hash, faiss_index_path = make_faiss_indexing().faiss_ivf_indexing(chunks, embeddings, str(tmp_path), make_faiss_ivf_args(ivf_nlist_multiplier=4))

    assert faiss_index.vector_store.index.nlist == 2


def test_faiss_ivf_nlist_is_at_least_1(make_faiss_indexing, make_faiss_ivf_args, make_chunks_and_embeddings, tmp_path):
    """
    Input: 3 chunks with ivf_nlist_multiplier 4.
    Output: the FAISS IVF index.
    What it does: checks the cap 3 // 39 = 0 is raised to 1, so IVF still builds with fewer chunks than 39.
    """
    chunks, embeddings = make_chunks_and_embeddings(3)

    faiss_index, faiss_hash, faiss_index_path = make_faiss_indexing().faiss_ivf_indexing(chunks, embeddings, str(tmp_path), make_faiss_ivf_args(ivf_nlist_multiplier=4))

    assert faiss_index.vector_store.index.nlist == 1
    assert faiss_index.vector_store.index.ntotal == 3


@pytest.mark.parametrize("faiss_type", ["faiss_flatip", "faiss_ivf", "faiss_hnsw"])
def test_faiss_index_raises_for_no_embeddings(build_faiss_index, tmp_path, faiss_type):
    """
    Input: no chunks and no embeddings.
    Output: ValueError.
    What it does: checks an empty index is never built.
    """
    with pytest.raises(ValueError, match="embeddings is empty"):
        build_faiss_index(faiss_type, {}, {}, str(tmp_path))


# ---------- load_faiss_index_from_path / load_saved_faiss_index ----------


@pytest.mark.parametrize("faiss_type", ["faiss_flatip", "faiss_ivf", "faiss_hnsw"])
def test_load_faiss_index_from_path_loads_a_saved_faiss_index(build_faiss_index, fake_huggingface_embeddings, make_chunks_and_embeddings, tmp_path, faiss_type):
    """
    Input: a FAISS index of 10 chunks built and saved, then loaded straight from its folder.
    Output: the loaded FaissIndex.
    What it does: checks the same type, folder and chunks come back, with the chunk text, without any chunks given.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    built_index, built_hash, built_path = build_faiss_index(faiss_type, chunks, embeddings, str(tmp_path))

    loaded_index = load_faiss_index_from_path(faiss_type, built_index.index_path, fake_huggingface_embeddings())

    assert isinstance(loaded_index, FaissIndex)
    assert loaded_index.index_type == faiss_type
    assert loaded_index.index_path == built_index.index_path
    assert loaded_index.vector_store.index.ntotal == 10
    assert loaded_index.vector_store.docstore.search("doc0_2").page_content == chunks["doc0_2"].chunk_texts


def test_load_faiss_index_from_path_raises_when_nothing_is_saved(fake_huggingface_embeddings, tmp_path):
    """
    Input: an empty folder.
    Output: FileNotFoundError.
    What it does: checks serving is told to run the build first.
    """
    with pytest.raises(FileNotFoundError, match="build"):
        load_faiss_index_from_path("faiss_flatip", str(tmp_path), fake_huggingface_embeddings())


@pytest.mark.parametrize("faiss_type", ["faiss_flatip", "faiss_ivf", "faiss_hnsw"])
def test_load_saved_faiss_index_loads_the_index_with_the_same_chunks(build_faiss_index, fake_huggingface_embeddings, make_chunks_and_embeddings, tmp_path, faiss_type):
    """
    Input: a FAISS index of 10 chunks built and saved, then loaded with the same chunks.
    Output: the loaded FaissIndex.
    What it does: checks the saved index comes back with its type, folder and every vector.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    built_index, built_hash, built_path = build_faiss_index(faiss_type, chunks, embeddings, str(tmp_path))

    loaded_index = load_saved_faiss_index(faiss_type, chunks, built_index.index_path, fake_huggingface_embeddings())

    assert isinstance(loaded_index, FaissIndex)
    assert loaded_index.index_type == faiss_type
    assert loaded_index.index_path == built_index.index_path
    assert loaded_index.vector_store.index.ntotal == 10


def test_load_saved_faiss_index_raises_when_the_chunks_differ(make_faiss_indexing, fake_huggingface_embeddings, make_chunks_and_embeddings, tmp_path):
    """
    Input: a FAISS index of 10 chunks built and saved, then loaded with 12 chunks.
    Output: FileNotFoundError.
    What it does: checks an index holding other chunks is not used, and the build is named.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    built_index, built_hash, built_path = make_faiss_indexing().faiss_flatip_indexing(chunks, embeddings, str(tmp_path))
    more_chunks, more_embeddings = make_chunks_and_embeddings(12)

    with pytest.raises(FileNotFoundError, match="build"):
        load_saved_faiss_index("faiss_flatip", more_chunks, built_index.index_path, fake_huggingface_embeddings())


def test_load_saved_faiss_index_raises_when_nothing_is_saved(fake_huggingface_embeddings, make_chunks_and_embeddings, tmp_path):
    """
    Input: an empty folder.
    Output: FileNotFoundError.
    What it does: checks the build is named when no FAISS index is saved.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)

    with pytest.raises(FileNotFoundError, match="build"):
        load_saved_faiss_index("faiss_flatip", chunks, str(tmp_path), fake_huggingface_embeddings())
