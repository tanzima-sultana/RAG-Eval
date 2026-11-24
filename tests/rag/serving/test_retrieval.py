"""
test_retrieval.py

Tests for rag.serving.retrieval (retrieval, single_retrieval and cache) and paths.get_retrieved_file_path. The indexes
are tiny real LangChain ones: a FAISS vector store over a keyword-count embedding, used as the FAISS and the Qdrant
record, and a BM25Retriever. The cross-encoder is a fake that scores each chunk by the number at the end of its text.
Each result is saved as its own file in tmp_path.
"""

import os

import pytest

from langchain_community.cross_encoders.base import BaseCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LangchainDocument
from langchain_core.embeddings import Embeddings

from common import paths
from common.args import QueryArgs
from common.args import RetrievalArgs
from common.constants import I_DONT_KNOW
from rag.serving.retrieval import retrieval as retrieval_module
from rag.serving.retrieval.cache import load_retrieved_entry
from rag.serving.retrieval.cache import save_retrieved_entry
from rag.serving.retrieval.retrieval import Retrieval
from rag.serving.retrieval.single_retrieval import make_retrieved
from schema.index import Bm25Index
from schema.index import FaissIndex
from schema.llm import LLMArgs
from schema.query import Ans
from schema.query import Query
from schema.query import Qus
from schema.setup import Setup
from schema.vectordb import VectorDb

# Each chunk text ends with the score the fake cross-encoder gives it
CHUNK_TEXTS = {
    "doc1_0": "apple banana 0.1",
    "doc1_1": "banana cherry 0.9",
    "doc2_0": "cherry date 0.5",
    "doc2_1": "date apple 0.7",
    "doc3_0": "egg fig 0.2",
    "doc3_1": "fig grape 0.3",
}
VOCABULARY = ["apple", "banana", "cherry", "date", "egg", "fig", "grape"]


class KeywordEmbeddings(Embeddings):
    """
    Stand-in for the embedding model: a text is embedded as the count of each vocabulary word in it, so a query
    vector is closest to the chunks that share its words.
    """

    def embed_documents(self, texts):
        """
        Input: texts, a list of strings.
        Output: one vector per text.
        What it does: counts each vocabulary word in each text.
        """
        vectors = []
        for text in texts:
            vectors.append(self.embed_query(text))
        return vectors

    def embed_query(self, text):
        """
        Input: text, one string.
        Output: its vector.
        What it does: counts each vocabulary word in the text.
        """
        words = text.lower().split()
        vector = []
        for vocabulary_word in VOCABULARY:
            vector.append(float(words.count(vocabulary_word)))
        return vector


class FakeCrossEncoder(BaseCrossEncoder):
    """
    Stand-in for LangChain's HuggingFaceCrossEncoder: a (question, chunk text) pair scores the number at the end of
    the chunk text. Every scored question is recorded in scored_questions.
    """

    scored_questions = []

    def __init__(self, model_name=None):
        """
        Input: model_name, as the real class takes it.
        Output: nothing.
        What it does: nothing; the fake needs no model.
        """
        return None

    def score(self, text_pairs):
        """
        Input: text_pairs, a list of (question, chunk text).
        Output: one score per pair.
        What it does: records each question and scores each pair by the number at the end of the chunk text.
        """
        scores = []
        for question, chunk_text in text_pairs:
            FakeCrossEncoder.scored_questions.append(question)
            scores.append(float(chunk_text.split()[-1]))
        return scores


@pytest.fixture
def fake_cross_encoder(monkeypatch):
    """
    Input: pytest's monkeypatch fixture.
    Output: the FakeCrossEncoder class.
    What it does: makes Retrieval build the fake cross-encoder.
    """
    monkeypatch.setattr(retrieval_module, "HuggingFaceCrossEncoder", FakeCrossEncoder)
    FakeCrossEncoder.scored_questions = []
    return FakeCrossEncoder


def make_langchain_documents():
    """
    Input: none.
    Output: one LangChain Document per chunk, with its chunk id in the metadata, as the build makes them.
    What it does: builds the chunks the tiny indexes hold.
    """
    documents = []
    for chunk_id in CHUNK_TEXTS:
        documents.append(LangchainDocument(page_content=CHUNK_TEXTS[chunk_id], metadata={"chunk_id": chunk_id}))
    return documents


def make_faiss_index():
    """
    Input: none.
    Output: a FaissIndex record holding a real LangChain FAISS store of the chunks.
    What it does: builds the tiny FAISS index.
    """
    vector_store = FAISS.from_documents(make_langchain_documents(), KeywordEmbeddings())
    return FaissIndex(index_type="faiss_flatip", vector_store=vector_store, index_path="unused", dimension=len(VOCABULARY))


def make_vectordb_index():
    """
    Input: none.
    Output: a VectorDb record of type qdrant, holding a real LangChain FAISS store of the chunks in place of Qdrant.
    What it does: builds a tiny vector DB record without a Qdrant server.
    """
    vector_store = FAISS.from_documents(make_langchain_documents(), KeywordEmbeddings())
    return VectorDb(vectordb_type="qdrant", vector_store=vector_store, collection_name="unused", dimension=len(VOCABULARY))


def lowercase_split(text):
    """
    Input: text, one string.
    Output: its lowercased words.
    What it does: tokenizes a text the way the build's BM25 index does.
    """
    return text.lower().split()


def make_bm25_index():
    """
    Input: none.
    Output: a Bm25Index record holding a real LangChain BM25Retriever of the chunks.
    What it does: builds the tiny BM25 index, lowercased and split on whitespace as the build does.
    """
    retriever = BM25Retriever.from_documents(make_langchain_documents(), preprocess_func=lowercase_split)
    return Bm25Index(retriever=retriever, index_path="unused")


@pytest.fixture
def make_retrieval_args(make_index_args):
    """
    Input: the make_index_args fixture.
    Output: a function that builds a RetrievalArgs.
    What it does: returns a builder with small defaults (dense, top_k 3, rerank_top_k 2, no reranking) where any field
    can be overridden.
    """
    def build_retrieval_args(**overrides):
        """
        Input: any RetrievalArgs field as a keyword argument.
        Output: a RetrievalArgs with the defaults plus the overrides.
        What it does: fills in defaults for every field not overridden. The index type is empty, as in the run.
        """
        field_values = {}
        field_values["index_args"] = make_index_args(index_type="")
        field_values["retrieval_types"] = ["dense"]
        field_values["top_k"] = 3
        field_values["reranking"] = False
        field_values["rerank_top_k"] = 2
        for field_name in overrides:
            field_values[field_name] = overrides[field_name]
        return RetrievalArgs(**field_values)

    return build_retrieval_args


def make_query_args(**overrides):
    """
    Input: any QueryArgs field as a keyword argument.
    Output: a QueryArgs with small defaults (no extra query type, so original) plus the overrides.
    What it does: fills in defaults for every field not overridden.
    """
    field_values = {}
    field_values["query_llm_args"] = LLMArgs("ollama_chat/llama3.1:8b", 100, 30)
    field_values["n_multi_qus"] = 2
    field_values["query_types"] = []
    field_values["query_split"] = True
    field_values["val_queries_size"] = 1
    field_values["test_queries_size"] = 1
    field_values["ollama_api_base"] = "http://ollama-test:11434"
    field_values["ollama_context_length"] = 32768
    for field_name in overrides:
        field_values[field_name] = overrides[field_name]
    return QueryArgs(**field_values)


def make_query(query_id="q1", hyde_text="apple banana"):
    """
    Input: query_id; hyde_text, the HyDE answer text.
    Output: a Query of texts: original "apple", rewritten "cherry", multi "banana" and "date", and the HyDE text.
    What it does: builds one query with all four query types.
    """
    return Query(
        id=query_id,
        original_qus=Qus("apple"),
        rewritten_qus=Qus("cherry"),
        multi_qus=[Qus("banana"), Qus("date")],
        hyde_ans=Ans(hyde_text),
        llm="ollama_chat",
        llm_model="llama3.1:8b",
    )


def make_index_records(given):
    """
    Input: given, the names of the records to build: any of "faiss", "vectordb" and "bm25".
    Output: (faiss_index, vectordb_index, bm25_index), with None for each record not given.
    What it does: builds only the named tiny index records.
    """
    faiss_index = None
    vectordb_index = None
    bm25_index = None
    if "faiss" in given:
        faiss_index = make_faiss_index()
    if "vectordb" in given:
        vectordb_index = make_vectordb_index()
    if "bm25" in given:
        bm25_index = make_bm25_index()
    return faiss_index, vectordb_index, bm25_index


# ---------- __init__ and validate ----------


@pytest.mark.parametrize("retrieval_overrides, query_types, message", [
    ({"retrieval_types": []}, [], "exactly one retrieval type"),
    ({"retrieval_types": ["dense", "bm25"]}, [], "exactly one retrieval type"),
    ({"retrieval_types": ["keyword"]}, [], "is not supported"),
    ({"top_k": 0}, [], "top_k"),
    ({"rerank_top_k": 0}, [], "rerank_top_k"),
    ({"rerank_top_k": 4}, [], "must not be more than top_k"),
    ({}, ["rewritten", "multi"], "at most one query type"),
    ({}, ["original"], "is not supported"),
])
def test_constructor_raises_for_invalid_setting(make_retrieval_args, tmp_path, retrieval_overrides, query_types, message):
    """
    Input: one invalid retrieval or query setting of a dense setup.
    Output: ValueError.
    What it does: checks the constructor validates the settings and names what is wrong.
    """
    retrieval_args = make_retrieval_args(**retrieval_overrides)

    with pytest.raises(ValueError, match=message):
        Retrieval(retrieval_args, make_query_args(query_types=query_types), make_faiss_index(), None, None, str(tmp_path))


@pytest.mark.parametrize("retrieval_type", ["bm25", "hybrid"])
def test_constructor_raises_for_hyde_with_bm25_or_hybrid(make_retrieval_args, tmp_path, retrieval_type):
    """
    Input: HyDE as the query type of a BM25 or hybrid setup.
    Output: ValueError.
    What it does: checks HyDE can only be searched with dense.
    """
    retrieval_args = make_retrieval_args(retrieval_types=[retrieval_type])

    with pytest.raises(ValueError, match="can only be searched with"):
        Retrieval(retrieval_args, make_query_args(query_types=["hyde"]), make_faiss_index(), None, make_bm25_index(), str(tmp_path))


@pytest.mark.parametrize("retrieval_type, given, message", [
    ("dense", [], "Dense retrieval needs"),
    ("dense", ["faiss", "vectordb"], "Dense retrieval needs"),
    ("dense", ["faiss", "bm25"], "Dense retrieval needs"),
    ("bm25", ["faiss", "bm25"], "BM25 retrieval needs"),
    ("bm25", [], "BM25 retrieval needs"),
    ("hybrid", ["faiss"], "Hybrid retrieval needs"),
    ("hybrid", ["bm25"], "Hybrid retrieval needs"),
])
def test_constructor_raises_when_the_index_records_do_not_match_the_retrieval_type(make_retrieval_args, tmp_path, retrieval_type, given, message):
    """
    Input: a retrieval type and index records it cannot search with.
    Output: ValueError.
    What it does: checks each retrieval type gets exactly the records it searches.
    """
    faiss_index, vectordb_index, bm25_index = make_index_records(given)

    with pytest.raises(ValueError, match=message):
        Retrieval(make_retrieval_args(retrieval_types=[retrieval_type]), make_query_args(), faiss_index, vectordb_index, bm25_index, str(tmp_path))


def test_constructor_does_not_change_the_shared_bm25_index(make_retrieval_args, tmp_path):
    """
    Input: one BM25 index record, used by a BM25 setup with top_k 3.
    Output: the k of the shared BM25 retriever and of the setup's retriever.
    What it does: checks the setup searches with its own copy, so other setups using the record are not changed.
    """
    bm25_index = make_bm25_index()
    shared_k = bm25_index.retriever.k

    retrieval = Retrieval(make_retrieval_args(retrieval_types=["bm25"]), make_query_args(), None, None, bm25_index, str(tmp_path))

    assert bm25_index.retriever.k == shared_k
    assert retrieval.retriever.k == 3


def test_constructor_builds_the_reranker_only_with_reranking(make_retrieval_args, fake_cross_encoder, tmp_path):
    """
    Input: the same dense setup with reranking off and on.
    Output: the reranker of each.
    What it does: checks the cross-encoder reranker is built only with reranking, with rerank_top_k as its top_n.
    """
    plain_retrieval = Retrieval(make_retrieval_args(), make_query_args(), make_faiss_index(), None, None, str(tmp_path))
    reranked_retrieval = Retrieval(make_retrieval_args(reranking=True), make_query_args(), make_faiss_index(), None, None, str(tmp_path))

    assert plain_retrieval.reranker is None
    assert reranked_retrieval.reranker.top_n == 2
    assert isinstance(reranked_retrieval.reranker.model, FakeCrossEncoder)


# ---------- retrieve: retrieval type, index type and query type ----------


@pytest.mark.parametrize("retrieval_type, given, index_type", [
    ("dense", ["faiss"], "faiss_flatip"),
    ("dense", ["vectordb"], "qdrant"),
    ("bm25", ["bm25"], "bm25"),
    ("hybrid", ["faiss", "bm25"], "faiss_flatip+bm25"),
    ("hybrid", ["vectordb", "bm25"], "qdrant+bm25"),
])
def test_retrieve_searches_with_the_setup_and_reads_the_index_type_from_the_record(make_retrieval_args, tmp_path, retrieval_type, given, index_type):
    """
    Input: one setup of each retrieval type and index record, and one query searched with original.
    Output: its Retrieved.
    What it does: checks the setup, with the index type read from the record, and that at most top_k chunks are
    found, each once, with the doc id of its chunk id.
    """
    faiss_index, vectordb_index, bm25_index = make_index_records(given)
    retrieval = Retrieval(make_retrieval_args(retrieval_types=[retrieval_type]), make_query_args(), faiss_index, vectordb_index, bm25_index, str(tmp_path))

    retrieved = retrieval.retrieve(make_query())

    assert retrieved.id == "q1"
    assert retrieved.setup == Setup(chunk_type="fixed", index_type=index_type, query_type="original", retrieval_type=retrieval_type, reranking=False)
    assert 0 < len(retrieved.chunk_ids) <= 3
    assert len(set(retrieved.chunk_ids)) == len(retrieved.chunk_ids)
    for position in range(len(retrieved.chunk_ids)):
        assert retrieved.doc_ids[position] == retrieved.chunk_ids[position].rsplit("_", 1)[0]


@pytest.mark.parametrize("query_types, query_type, first_chunk_word", [
    ([], "original", "apple"),
    (["rewritten"], "rewritten", "cherry"),
])
def test_retrieve_searches_with_the_query_type_of_the_query_args(make_retrieval_args, tmp_path, query_types, query_type, first_chunk_word):
    """
    Input: a BM25 setup whose query settings hold no extra query type, or rewritten.
    Output: its Retrieved.
    What it does: checks no extra query type searches with original, and one searches with that type's text.
    """
    retrieval = Retrieval(make_retrieval_args(retrieval_types=["bm25"]), make_query_args(query_types=query_types), None, None, make_bm25_index(), str(tmp_path))

    retrieved = retrieval.retrieve(make_query())

    assert retrieved.setup.query_type == query_type
    assert first_chunk_word in CHUNK_TEXTS[retrieved.chunk_ids[0]]


def test_retrieve_joins_the_multi_phrasings_with_each_chunk_once(make_retrieval_args, tmp_path):
    """
    Input: a BM25 setup with top_k 4 searched with multi, whose phrasings "banana" and "date" each find two chunks.
    Output: its Retrieved.
    What it does: checks the chunks of both phrasings are kept, each once.
    """
    retrieval = Retrieval(make_retrieval_args(retrieval_types=["bm25"], top_k=4), make_query_args(query_types=["multi"]), None, None, make_bm25_index(), str(tmp_path))

    retrieved = retrieval.retrieve(make_query())

    assert retrieved.setup.query_type == "multi"
    assert len(set(retrieved.chunk_ids)) == len(retrieved.chunk_ids)
    assert set(retrieved.chunk_ids) == {"doc1_0", "doc1_1", "doc2_0", "doc2_1"}


def test_retrieve_does_not_search_a_hyde_i_dont_know(make_retrieval_args, tmp_path):
    """
    Input: a dense setup searched with hyde, and a query whose HyDE answer is I_DONT_KNOW.
    Output: its Retrieved.
    What it does: checks nothing is searched but a Retrieved with no chunks is still given, so it counts as a miss.
    """
    retrieval = Retrieval(make_retrieval_args(), make_query_args(query_types=["hyde"]), make_faiss_index(), None, None, str(tmp_path))

    retrieved = retrieval.retrieve(make_query(hyde_text=I_DONT_KNOW))

    assert retrieved.setup.query_type == "hyde"
    assert retrieved.chunk_ids == []


@pytest.mark.parametrize("query_type, missing_field", [
    ("rewritten", "rewritten_qus"),
    ("multi", "multi_qus"),
    ("hyde", "hyde_ans"),
])
def test_retrieve_raises_when_the_query_does_not_hold_the_query_type(make_retrieval_args, tmp_path, query_type, missing_field):
    """
    Input: a query whose query type of this setup is None.
    Output: ValueError.
    What it does: checks the input is rejected and names the missing query type.
    """
    query = make_query()
    setattr(query, missing_field, None)
    retrieval = Retrieval(make_retrieval_args(), make_query_args(query_types=[query_type]), make_faiss_index(), None, None, str(tmp_path))

    with pytest.raises(ValueError, match=query_type):
        retrieval.retrieve(query)


# ---------- retrieve: reranking ----------


def test_retrieve_reranks_against_the_original_question(make_retrieval_args, fake_cross_encoder, tmp_path):
    """
    Input: a reranked BM25 setup with top_k 2 searched with rewritten "cherry".
    Output: its Retrieved and the questions the cross-encoder scored.
    What it does: checks the chunks are sorted by the cross-encoder score, rerank_top_k are kept, the setup is marked
    reranked, and every chunk is scored against the original question, not the rewritten one.
    """
    retrieval_args = make_retrieval_args(retrieval_types=["bm25"], top_k=2, reranking=True)
    retrieval = Retrieval(retrieval_args, make_query_args(query_types=["rewritten"]), None, None, make_bm25_index(), str(tmp_path))

    retrieved = retrieval.retrieve(make_query())

    # "cherry" finds doc1_1 (score 0.9) and doc2_0 (score 0.5)
    assert retrieved.chunk_ids == ["doc1_1", "doc2_0"]
    assert retrieved.rerank_top_k == 2
    assert retrieved.setup.reranking is True
    assert retrieved.setup.query_type == "rewritten"
    assert set(fake_cross_encoder.scored_questions) == {"apple"}


# ---------- retrieve: cache ----------


def test_retrieve_saves_the_result_and_loads_it_without_searching(make_retrieval_args, tmp_path, capsys):
    """
    Input: the same setup retrieved twice, the second time by a new Retrieval with a query of the same texts and
    another id.
    Output: both results and what was printed.
    What it does: checks the first call searches and saves one file, and the second loads it and gives it the second
    query's id.
    """
    first_retrieved = Retrieval(make_retrieval_args(), make_query_args(), make_faiss_index(), None, None, str(tmp_path)).retrieve(make_query())
    assert ": made, " in capsys.readouterr().out
    assert len(os.listdir(str(tmp_path))) == 1

    second_retrieved = Retrieval(make_retrieval_args(), make_query_args(), make_faiss_index(), None, None, str(tmp_path)).retrieve(make_query(query_id="q2"))

    assert ": loaded from cache, " in capsys.readouterr().out
    assert second_retrieved.id == "q2"
    assert second_retrieved.chunk_ids == first_retrieved.chunk_ids
    assert second_retrieved.setup == first_retrieved.setup


def test_save_and_load_retrieved_entry_round_trip(tmp_path):
    """
    Input: one Retrieved, saved and loaded back.
    Output: the loaded Retrieved.
    What it does: checks every field but the id comes back, the id is empty, and no temporary file is left.
    """
    setup = Setup(chunk_type="fixed", index_type="faiss_flatip+bm25", query_type="multi", retrieval_type="hybrid", reranking=False)
    retrieved = make_retrieved("q1", setup, 3, 0, make_langchain_documents()[:2])
    retrieved_file = os.path.join(str(tmp_path), "retrieved", "entry.json")

    save_retrieved_entry(retrieved_file, "apple", retrieved)
    loaded = load_retrieved_entry(retrieved_file)

    assert loaded.id == ""
    assert loaded.setup == retrieved.setup
    assert loaded.top_k == 3
    assert loaded.rerank_top_k == 0
    assert loaded.chunk_ids == ["doc1_0", "doc1_1"]
    assert loaded.doc_ids == ["doc1", "doc1"]
    assert not os.path.exists(retrieved_file + ".tmp")


def test_load_retrieved_entry_gives_none_when_the_file_is_missing(tmp_path):
    """
    Input: a path with no file.
    Output: None.
    What it does: checks a missing entry is reported as None.
    """
    assert load_retrieved_entry(os.path.join(str(tmp_path), "missing.json")) is None


# ---------- get_retrieved_file_path ----------


def get_retrieved_file_name(retrieval_args, query_type="original", index_type="faiss_flatip", query_texts=None, original_qus="apple"):
    """
    Input: the arguments of get_retrieved_file_path, with defaults; query_texts defaults to ["apple"].
    Output: the file name of that content.
    What it does: builds the retrieved file path under "out" and keeps only its name.
    """
    if query_texts is None:
        query_texts = ["apple"]
    return os.path.basename(paths.get_retrieved_file_path("out", retrieval_args, query_type, index_type, query_texts, original_qus))


def test_get_retrieved_file_path_is_a_12_character_hash_in_the_given_folder(make_retrieval_args):
    """
    Input: the default content, twice.
    Output: the retrieved file path.
    What it does: checks the file is {hash}.json directly in the given folder, with a 12 hex character hash, the same
    for the same content.
    """
    retrieved_file = paths.get_retrieved_file_path("out", make_retrieval_args(), "original", "faiss_flatip", ["apple"], "apple")
    same_retrieved_file = paths.get_retrieved_file_path("out", make_retrieval_args(), "original", "faiss_flatip", ["apple"], "apple")

    file_name = os.path.basename(retrieved_file)
    assert os.path.dirname(retrieved_file) == "out"
    assert len(file_name) == 12 + len(".json")
    int(file_name[:12], 16)
    assert retrieved_file == same_retrieved_file


def test_get_retrieved_file_path_changes_with_each_hashed_content(make_retrieval_args, make_index_args, make_chunking_args):
    """
    Input: the default content, and the same with one part changed: the searched texts, the query type, the retrieval
    type, top_k, reranking, the index type, the input hash, a chunk setting or an index setting.
    Output: the file names.
    What it does: checks each changed part gives a different file.
    """
    default_name = get_retrieved_file_name(make_retrieval_args())

    changed_names = [
        get_retrieved_file_name(make_retrieval_args(), query_texts=["cherry"]),
        get_retrieved_file_name(make_retrieval_args(), query_type="rewritten"),
        get_retrieved_file_name(make_retrieval_args(retrieval_types=["bm25"])),
        get_retrieved_file_name(make_retrieval_args(top_k=5)),
        get_retrieved_file_name(make_retrieval_args(reranking=True)),
        get_retrieved_file_name(make_retrieval_args(), index_type="qdrant"),
        get_retrieved_file_name(make_retrieval_args(index_args=make_index_args(index_type="", input_hash="otherinputhash"))),
        get_retrieved_file_name(make_retrieval_args(index_args=make_index_args(index_type="", chunking_args=make_chunking_args(max_chunk_size=8)))),
        get_retrieved_file_name(make_retrieval_args(index_args=make_index_args(index_type="", hnsw_m=8))),
    ]

    for changed_name in changed_names:
        assert changed_name != default_name


def test_get_retrieved_file_path_uses_the_original_question_and_rerank_top_k_only_with_reranking(make_retrieval_args):
    """
    Input: another original question and rerank_top_k, with reranking off and on.
    Output: the file names.
    What it does: checks they change the file only of a reranked result, since only the cross-encoder reads them.
    """
    assert get_retrieved_file_name(make_retrieval_args(), original_qus="banana") == get_retrieved_file_name(make_retrieval_args())
    assert get_retrieved_file_name(make_retrieval_args(rerank_top_k=1)) == get_retrieved_file_name(make_retrieval_args())
    assert get_retrieved_file_name(make_retrieval_args(reranking=True), original_qus="banana") != get_retrieved_file_name(make_retrieval_args(reranking=True))
    assert get_retrieved_file_name(make_retrieval_args(reranking=True, rerank_top_k=1)) != get_retrieved_file_name(make_retrieval_args(reranking=True))
