"""
test_indexing.py

Tests for rag.build.indexing (indexing.py, indexers/faiss_indexer.py and indexers/bm25_indexer.py). HuggingFaceEmbeddings is replaced with a fake, so no model is downloaded.
FAISS and BM25 are real.
"""

import os

import pytest

from common import paths
from rag.build.embedding import embedding as embedding_module
from rag.build.indexing.indexers.bm25_indexer import load_bm25_index_from_path
from rag.build.indexing.indexers.bm25_indexer import load_saved_bm25_index
from rag.build.indexing.indexers.bm25_indexer import lowercase_whitespace_tokenize
from rag.build.indexing.indexers.faiss_indexer import load_faiss_index_from_path
from rag.build.indexing.indexers.faiss_indexer import load_saved_faiss_index
from rag.build.indexing.indexing import Indexing
from schema.index import Bm25Index
from schema.index import FaissIndex


@pytest.fixture
def patched_embeddings(monkeypatch, fake_huggingface_embeddings):
    """
    Input: pytest's monkeypatch fixture; the fake_huggingface_embeddings fixture.
    Output: the FakeHuggingFaceEmbeddings class.
    What it does: makes the embedding model factory, which Indexing uses, load the fake instead of the real model.
    """
    monkeypatch.setattr(embedding_module, "HuggingFaceEmbeddings", fake_huggingface_embeddings)
    return fake_huggingface_embeddings


@pytest.fixture
def make_indexing(make_embedder_args, make_chunking_args):
    """
    Input: the make_embedder_args and make_chunking_args fixtures.
    Output: a function that builds an Indexing from IndexArgs.
    What it does: returns a builder that creates Indexing with the default embedder and chunking args.
    """
    def build_indexing(index_args):
        """
        Input: index_args.
        Output: an Indexing with the default embedder and chunking args and these index args.
        What it does: creates the Indexing.
        """
        return Indexing(make_embedder_args(), make_chunking_args(), index_args)

    return build_indexing


def test_lowercase_whitespace_tokenize():
    """
    Input: text with upper case, tabs and new lines.
    Output: the words.
    What it does: checks text is lowercased and split on whitespace.
    """
    assert lowercase_whitespace_tokenize("The Cat\tsat\non  MAT") == ["the", "cat", "sat", "on", "mat"]


# ---------- validate ----------


@pytest.mark.parametrize("overrides, message", [
    ({"index_type": "qdrant"}, "index_type"),
    ({"index_type": "faiss_ivf", "ivf_nlist_multiplier": 0}, "ivf_nlist_multiplier"),
    ({"index_type": "faiss_hnsw", "hnsw_m": 0}, "hnsw_m"),
    ({"index_type": "faiss_hnsw", "hnsw_ef_construction": 0}, "hnsw_ef_construction"),
    ({"index_type": "bm25", "bm25_tokenizer": "nltk"}, "bm25_tokenizer"),
])
def test_init_raises_for_invalid_setting(patched_embeddings, make_indexing, make_index_args, overrides, message):
    """
    Input: IndexArgs with one invalid setting of its own index type.
    Output: ValueError.
    What it does: checks Indexing validates the settings when it is created, and names the invalid one.
    """
    with pytest.raises(ValueError, match=message):
        make_indexing(make_index_args(**overrides))


@pytest.mark.parametrize("overrides, message", [
    ({"index_type": "faiss_ivf", "ivf_nlist_multiplier": None}, "ivf_nlist_multiplier is missing"),
    ({"index_type": "faiss_hnsw", "hnsw_m": None}, "hnsw_m is missing"),
    ({"index_type": "faiss_hnsw", "hnsw_ef_construction": None}, "hnsw_ef_construction is missing"),
    ({"index_type": "bm25", "bm25_k1": None}, "bm25_k1 is missing"),
    ({"index_type": "bm25", "bm25_b": None}, "bm25_b is missing"),
])
def test_init_raises_when_a_setting_of_its_index_type_is_missing(patched_embeddings, make_indexing, make_index_args, overrides, message):
    """
    Input: IndexArgs with one setting of its own index type missing (None).
    Output: ValueError.
    What it does: checks a missing setting the index type needs is named, with its config block.
    """
    with pytest.raises(ValueError, match=message):
        make_indexing(make_index_args(**overrides))


@pytest.mark.parametrize("index_type", ["faiss_flatip", "faiss_ivf", "faiss_hnsw", "bm25"])
def test_init_checks_only_the_settings_of_its_index_type(patched_embeddings, make_indexing, make_index_args, index_type):
    """
    Input: IndexArgs of one index type, with every setting of the other index types missing (None).
    Output: nothing.
    What it does: checks Indexing is created: settings of index types not being built are not needed. faiss_flatip
    needs none at all.
    """
    settings_by_index_type = {
        "faiss_flatip": [],
        "faiss_ivf": ["ivf_nlist_multiplier"],
        "faiss_hnsw": ["hnsw_m", "hnsw_ef_construction"],
        "bm25": ["bm25_k1", "bm25_b", "bm25_tokenizer"],
    }
    all_settings = ["ivf_nlist_multiplier", "hnsw_m", "hnsw_ef_construction", "bm25_k1", "bm25_b", "bm25_tokenizer"]
    overrides = {"index_type": index_type}
    for setting_name in all_settings:
        if setting_name not in settings_by_index_type[index_type]:
            overrides[setting_name] = None

    indexing = make_indexing(make_index_args(**overrides))

    assert indexing.index_args.index_type == index_type


# ---------- compute_index ----------


@pytest.mark.parametrize("index_type, index_class", [
    ("faiss_flatip", FaissIndex),
    ("faiss_ivf", FaissIndex),
    ("faiss_hnsw", FaissIndex),
    ("bm25", Bm25Index),
])
def test_compute_index_returns_the_index_of_the_index_type(patched_embeddings, make_indexing, make_index_args, make_chunks_and_embeddings, tmp_path, index_type, index_class):
    """
    Input: 10 chunks and each index type.
    Output: the index compute_index returns.
    What it does: checks each index type is built by its own method.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)

    index = make_indexing(make_index_args(index_type=index_type)).compute_index(chunks, embeddings, str(tmp_path))

    assert isinstance(index, index_class)


# ---------- FAISS ----------


@pytest.mark.parametrize("index_type", ["faiss_flatip", "faiss_ivf", "faiss_hnsw"])
def test_faiss_index_builds_and_saves(patched_embeddings, make_indexing, make_embedder_args, make_chunking_args, make_index_args, make_chunks_and_embeddings, tmp_path, index_type):
    """
    Input: 10 chunks and each FAISS index type, with no saved index.
    Output: the FaissIndex and the files in its folder.
    What it does: checks every chunk is in the vector store with its text and chunk_id, and the store is saved.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    index_args = make_index_args(index_type=index_type)

    faiss_index = make_indexing(index_args).compute_index(chunks, embeddings, str(tmp_path))

    assert faiss_index.index_type == index_type
    assert faiss_index.vector_store.index.ntotal == 10
    assert faiss_index.dimension == 4
    document = faiss_index.vector_store.docstore.search("doc0_2")
    assert document.page_content == chunks["doc0_2"].text
    assert document.metadata["chunk_id"] == "doc0_2"

    faiss_index_path = paths.get_faiss_index_path(str(tmp_path), make_embedder_args(), make_chunking_args(), index_args)
    assert faiss_index.index_path == faiss_index_path
    assert os.path.exists(os.path.join(faiss_index_path, "index.faiss"))
    assert os.path.exists(os.path.join(faiss_index_path, "index.pkl"))


def test_faiss_search_with_a_chunk_vector_finds_that_chunk(patched_embeddings, make_indexing, make_index_args, make_chunks_and_embeddings, tmp_path):
    """
    Input: 10 chunks in a flatip index, searched with chunk "doc1_3"'s vector.
    Output: the top hit and its score.
    What it does: checks vectors are stored at the right positions and the score is cosine similarity.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    faiss_index = make_indexing(make_index_args()).compute_index(chunks, embeddings, str(tmp_path))

    documents_and_scores = faiss_index.vector_store.similarity_search_with_score_by_vector(
        embeddings["doc1_3"].embeddings.tolist(), k=1
    )

    document, score = documents_and_scores[0]
    assert document.metadata["chunk_id"] == "doc1_3"
    assert score == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("index_type", ["faiss_flatip", "faiss_ivf", "faiss_hnsw"])
def test_faiss_index_loads_from_cache_on_second_run(patched_embeddings, make_indexing, make_index_args, make_chunks_and_embeddings, tmp_path, capsys, index_type):
    """
    Input: the same chunks indexed twice with each FAISS index type.
    Output: what the second run printed and its index.
    What it does: checks the second run loads the saved index and does not build a new one.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    index_args = make_index_args(index_type=index_type)
    make_indexing(index_args).compute_index(chunks, embeddings, str(tmp_path))
    capsys.readouterr()

    faiss_index = make_indexing(index_args).compute_index(chunks, embeddings, str(tmp_path))

    printed = capsys.readouterr().out
    assert "loaded from cache : 10 vectors" in printed
    assert "built" not in printed
    assert faiss_index.vector_store.index.ntotal == 10


def test_faiss_index_is_rebuilt_when_chunks_change(patched_embeddings, make_indexing, make_index_args, make_chunks_and_embeddings, tmp_path, capsys):
    """
    Input: 10 chunks indexed, then 12 chunks indexed with the same settings.
    Output: what the second run printed and its index.
    What it does: checks a saved index with different chunk ids is replaced by a new one.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    make_indexing(make_index_args()).compute_index(chunks, embeddings, str(tmp_path))
    capsys.readouterr()

    more_chunks, more_embeddings = make_chunks_and_embeddings(12)
    faiss_index = make_indexing(make_index_args()).compute_index(more_chunks, more_embeddings, str(tmp_path))

    assert "FAISS flatip index built : 12 vectors" in capsys.readouterr().out
    assert faiss_index.vector_store.index.ntotal == 12


def test_faiss_ivf_nlist_is_multiplier_times_sqrt_of_chunks(patched_embeddings, make_indexing, make_index_args, make_chunks_and_embeddings, tmp_path):
    """
    Input: 400 chunks with ivf_nlist_multiplier 0.1.
    Output: the FAISS IVF index.
    What it does: checks nlist is round(0.1 * sqrt(400)) = 2, which is under the cap 400 // 39 = 10.
    """
    chunks, embeddings = make_chunks_and_embeddings(400)

    faiss_index = make_indexing(make_index_args(index_type="faiss_ivf", ivf_nlist_multiplier=0.1)).compute_index(chunks, embeddings, str(tmp_path))

    assert faiss_index.vector_store.index.nlist == 2


def test_faiss_ivf_nlist_is_capped_at_chunks_divided_by_39(patched_embeddings, make_indexing, make_index_args, make_chunks_and_embeddings, tmp_path):
    """
    Input: 100 chunks with ivf_nlist_multiplier 4.
    Output: the FAISS IVF index.
    What it does: checks round(4 * sqrt(100)) = 40 is capped silently at 100 // 39 = 2.
    """
    chunks, embeddings = make_chunks_and_embeddings(100)

    faiss_index = make_indexing(make_index_args(index_type="faiss_ivf", ivf_nlist_multiplier=4)).compute_index(chunks, embeddings, str(tmp_path))

    assert faiss_index.vector_store.index.nlist == 2


def test_faiss_ivf_nlist_is_at_least_1(patched_embeddings, make_indexing, make_index_args, make_chunks_and_embeddings, tmp_path):
    """
    Input: 3 chunks with ivf_nlist_multiplier 4.
    Output: the FAISS IVF index.
    What it does: checks the cap 3 // 39 = 0 is raised to 1, so IVF still builds with fewer chunks than 39.
    """
    chunks, embeddings = make_chunks_and_embeddings(3)

    faiss_index = make_indexing(make_index_args(index_type="faiss_ivf", ivf_nlist_multiplier=4)).compute_index(chunks, embeddings, str(tmp_path))

    assert faiss_index.vector_store.index.nlist == 1
    assert faiss_index.vector_store.index.ntotal == 3


@pytest.mark.parametrize("index_type", ["faiss_flatip", "faiss_ivf", "faiss_hnsw"])
def test_faiss_index_raises_for_no_embeddings(patched_embeddings, make_indexing, make_index_args, tmp_path, index_type):
    """
    Input: no chunks and no embeddings.
    Output: ValueError.
    What it does: checks an empty index is never built.
    """
    with pytest.raises(ValueError, match="embeddings is empty"):
        make_indexing(make_index_args(index_type=index_type)).compute_index({}, {}, str(tmp_path))


# ---------- BM25 ----------


def test_bm25_index_builds_and_saves(patched_embeddings, make_indexing, make_embedder_args, make_chunking_args, make_index_args, make_chunks_and_embeddings, tmp_path):
    """
    Input: 10 chunks, with bm25_k1 1.2 and bm25_b 0.5.
    Output: the Bm25Index and its saved file.
    What it does: checks every chunk is in the retriever, k1 and b are used, and the retriever is saved.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    index_args = make_index_args(index_type="bm25", bm25_k1=1.2, bm25_b=0.5)

    bm25_index = make_indexing(index_args).compute_index(chunks, embeddings, str(tmp_path))

    assert len(bm25_index.retriever.docs) == 10
    assert bm25_index.retriever.vectorizer.k1 == 1.2
    assert bm25_index.retriever.vectorizer.b == 0.5
    bm25_index_path = paths.get_bm25_index_path(str(tmp_path), make_embedder_args(), make_chunking_args(), index_args)
    assert bm25_index.index_path == os.path.join(bm25_index_path, "bm25_retriever.pkl")
    assert os.path.exists(bm25_index.index_path)


def test_bm25_search_finds_the_chunk_with_the_word(patched_embeddings, make_indexing, make_index_args, make_chunks_and_embeddings, tmp_path):
    """
    Input: 10 chunks, searched with "WORD3".
    Output: the top hit.
    What it does: checks the query is lowercased like the chunks and the chunk with that word comes first.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    bm25_index = make_indexing(make_index_args(index_type="bm25")).compute_index(chunks, embeddings, str(tmp_path))

    documents = bm25_index.retriever.invoke("WORD3")

    assert documents[0].metadata["chunk_id"] == "doc1_3"


def test_bm25_index_loads_from_cache_on_second_run(patched_embeddings, make_indexing, make_index_args, make_chunks_and_embeddings, tmp_path, capsys):
    """
    Input: the same chunks indexed twice with bm25.
    Output: what the second run printed.
    What it does: checks the second run loads the saved retriever and does not build a new one.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    index_args = make_index_args(index_type="bm25")
    make_indexing(index_args).compute_index(chunks, embeddings, str(tmp_path))
    capsys.readouterr()

    make_indexing(index_args).compute_index(chunks, embeddings, str(tmp_path))

    printed = capsys.readouterr().out
    assert "BM25 index loaded from cache : 10 chunks" in printed
    assert "built" not in printed


def test_bm25_index_is_rebuilt_when_chunks_change(patched_embeddings, make_indexing, make_index_args, make_chunks_and_embeddings, tmp_path, capsys):
    """
    Input: 10 chunks indexed, then 12 chunks indexed with the same settings.
    Output: what the second run printed and its index.
    What it does: checks a saved retriever with different chunk ids is replaced by a new one.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    make_indexing(make_index_args(index_type="bm25")).compute_index(chunks, embeddings, str(tmp_path))
    capsys.readouterr()

    more_chunks, more_embeddings = make_chunks_and_embeddings(12)
    bm25_index = make_indexing(make_index_args(index_type="bm25")).compute_index(more_chunks, more_embeddings, str(tmp_path))

    assert "BM25 index built : 12 chunks" in capsys.readouterr().out
    assert len(bm25_index.retriever.docs) == 12


def test_bm25_index_raises_for_no_chunks(patched_embeddings, make_indexing, make_index_args, tmp_path):
    """
    Input: no chunks.
    Output: ValueError.
    What it does: checks an empty retriever is never built.
    """
    with pytest.raises(ValueError, match="chunks is empty"):
        make_indexing(make_index_args(index_type="bm25")).compute_index({}, {}, str(tmp_path))


# ---------- load_index_from_path ----------


@pytest.mark.parametrize("index_type", ["faiss_flatip", "faiss_ivf", "faiss_hnsw"])
def test_load_faiss_index_from_path_loads_a_saved_faiss_index(patched_embeddings, make_indexing, make_index_args, make_chunks_and_embeddings, tmp_path, index_type):
    """
    Input: a FAISS index of 10 chunks built and saved, then loaded straight from its folder.
    Output: the loaded FaissIndex.
    What it does: checks the same index type, folder and chunks come back, with the chunk text, without any chunks given.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    built_index = make_indexing(make_index_args(index_type=index_type)).compute_index(chunks, embeddings, str(tmp_path))

    loaded_index = load_faiss_index_from_path(index_type, built_index.index_path, patched_embeddings())

    assert isinstance(loaded_index, FaissIndex)
    assert loaded_index.index_type == index_type
    assert loaded_index.index_path == built_index.index_path
    assert loaded_index.vector_store.index.ntotal == 10
    assert loaded_index.vector_store.docstore.search("doc0_2").page_content == chunks["doc0_2"].text


def test_load_bm25_index_from_path_loads_a_saved_bm25_index(patched_embeddings, make_indexing, make_embedder_args, make_chunking_args, make_index_args, make_chunks_and_embeddings, tmp_path):
    """
    Input: a BM25 index of 10 chunks built and saved, then loaded straight from its folder.
    Output: the loaded Bm25Index.
    What it does: checks every chunk comes back, and the index path is the same pickle path the build gives.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    index_args = make_index_args(index_type="bm25")
    built_index = make_indexing(index_args).compute_index(chunks, embeddings, str(tmp_path))

    loaded_index = load_bm25_index_from_path(paths.get_bm25_index_path(str(tmp_path), make_embedder_args(), make_chunking_args(), index_args))

    assert isinstance(loaded_index, Bm25Index)
    assert len(loaded_index.retriever.docs) == 10
    assert loaded_index.index_path == built_index.index_path


def test_load_faiss_index_from_path_raises_when_nothing_is_saved(patched_embeddings, tmp_path):
    """
    Input: an empty folder.
    Output: FileNotFoundError.
    What it does: checks serving is told to run the build first.
    """
    with pytest.raises(FileNotFoundError, match="build"):
        load_faiss_index_from_path("faiss_flatip", str(tmp_path), patched_embeddings())


def test_load_bm25_index_from_path_raises_when_nothing_is_saved(tmp_path):
    """
    Input: an empty folder.
    Output: FileNotFoundError.
    What it does: checks serving is told to run the build first.
    """
    with pytest.raises(FileNotFoundError, match="build"):
        load_bm25_index_from_path(str(tmp_path))


def test_init_raises_for_unknown_embedding_provider(patched_embeddings, make_embedder_args, make_chunking_args, make_index_args):
    """
    Input: EmbedderArgs with embedding_provider "unknown".
    Output: ValueError.
    What it does: checks Indexing validates the embedding provider when it is created, and names it.
    """
    with pytest.raises(ValueError, match="embedding_provider"):
        Indexing(make_embedder_args(embedding_provider="unknown"), make_chunking_args(), make_index_args())


# ---------- load_saved_faiss_index / load_saved_bm25_index ----------


@pytest.mark.parametrize("index_type", ["faiss_flatip", "faiss_ivf", "faiss_hnsw"])
def test_load_saved_faiss_index_loads_the_index_with_the_same_chunks(patched_embeddings, make_indexing, make_index_args, make_chunks_and_embeddings, tmp_path, index_type):
    """
    Input: a FAISS index of 10 chunks built and saved, then loaded with the same chunks.
    Output: the loaded FaissIndex.
    What it does: checks the saved index comes back with its type, folder and every vector.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    built_index = make_indexing(make_index_args(index_type=index_type)).compute_index(chunks, embeddings, str(tmp_path))

    loaded_index = load_saved_faiss_index(index_type, chunks, built_index.index_path, patched_embeddings())

    assert isinstance(loaded_index, FaissIndex)
    assert loaded_index.index_type == index_type
    assert loaded_index.index_path == built_index.index_path
    assert loaded_index.vector_store.index.ntotal == 10


def test_load_saved_faiss_index_raises_when_the_chunks_differ(patched_embeddings, make_indexing, make_index_args, make_chunks_and_embeddings, tmp_path):
    """
    Input: a FAISS index of 10 chunks built and saved, then loaded with 12 chunks.
    Output: FileNotFoundError.
    What it does: checks an index holding other chunks is not used, and the build is named.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    built_index = make_indexing(make_index_args()).compute_index(chunks, embeddings, str(tmp_path))
    more_chunks, more_embeddings = make_chunks_and_embeddings(12)

    with pytest.raises(FileNotFoundError, match="build"):
        load_saved_faiss_index("faiss_flatip", more_chunks, built_index.index_path, patched_embeddings())


def test_load_saved_faiss_index_raises_when_nothing_is_saved(patched_embeddings, make_chunks_and_embeddings, tmp_path):
    """
    Input: an empty folder.
    Output: FileNotFoundError.
    What it does: checks the build is named when no FAISS index is saved.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)

    with pytest.raises(FileNotFoundError, match="build"):
        load_saved_faiss_index("faiss_flatip", chunks, str(tmp_path), patched_embeddings())


def test_load_saved_bm25_index_loads_the_index_with_the_same_chunks(patched_embeddings, make_indexing, make_index_args, make_chunks_and_embeddings, tmp_path):
    """
    Input: a BM25 index of 10 chunks built and saved, then loaded with the same chunks from its folder.
    Output: the loaded Bm25Index.
    What it does: checks every chunk comes back, and the index path is the same pickle path the build gives.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    built_index = make_indexing(make_index_args(index_type="bm25")).compute_index(chunks, embeddings, str(tmp_path))

    loaded_index = load_saved_bm25_index(chunks, os.path.dirname(built_index.index_path))

    assert isinstance(loaded_index, Bm25Index)
    assert len(loaded_index.retriever.docs) == 10
    assert loaded_index.index_path == built_index.index_path


def test_load_saved_bm25_index_raises_when_the_chunks_differ(patched_embeddings, make_indexing, make_index_args, make_chunks_and_embeddings, tmp_path):
    """
    Input: a BM25 index of 10 chunks built and saved, then loaded with 12 chunks.
    Output: FileNotFoundError.
    What it does: checks an index holding other chunks is not used, and the build is named.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)
    built_index = make_indexing(make_index_args(index_type="bm25")).compute_index(chunks, embeddings, str(tmp_path))
    more_chunks, more_embeddings = make_chunks_and_embeddings(12)

    with pytest.raises(FileNotFoundError, match="build"):
        load_saved_bm25_index(more_chunks, os.path.dirname(built_index.index_path))


def test_load_saved_bm25_index_raises_when_nothing_is_saved(make_chunks_and_embeddings, tmp_path):
    """
    Input: an empty folder.
    Output: FileNotFoundError.
    What it does: checks the build is named when no BM25 index is saved.
    """
    chunks, embeddings = make_chunks_and_embeddings(10)

    with pytest.raises(FileNotFoundError, match="build"):
        load_saved_bm25_index(chunks, str(tmp_path))
