"""
test_setup_ranking.py

Tests for evaluation.setup_ranking on small hand-built evals: grouping by setup, the means, the order and the top n.
"""

import os

import pytest

from common import paths
from evaluation.setup_ranking import load_best_setups
from evaluation.setup_ranking import rank_setups
from evaluation.setup_ranking import save_best_setups
from schema.manifest import BuildManifest
from schema.manifest import ChunkEntry
from schema.manifest import IndexEntry
from schema.retrieved import Retrieved
from schema.retrieved_eval import Eval
from schema.retrieved_eval import RetrievedEval
from schema.setup import Setup
from schema.setup import SetupRecord


def make_setup(index_type, reranking=False):
    """
    Input: index_type; reranking.
    Output: a dense original Setup of the fixed chunk type.
    What it does: builds one setup, told apart by its index type and reranking.
    """
    return Setup(chunk_type="fixed", index_type=index_type, query_type="original", retrieval_type="dense", reranking=reranking)


def make_retrieved_eval(setup, query_id, recall, mrr):
    """
    Input: setup; query_id; recall and mrr, the chunk level scores.
    Output: a RetrievedEval of that setup and query.
    What it does: builds one scored result. Only the chunk level scores are used by the ranking.
    """
    retrieved = Retrieved(id=query_id, setup=setup, top_k=3, rerank_top_k=0, chunk_ids=[], chunk_scores=[], doc_ids=[])
    chunk_eval = Eval(hit=1, precision=0.0, recall=recall, mrr=mrr, ndcg=0.5, first_relevant_rank=1)
    doc_eval = Eval(hit=0, precision=0.0, recall=0.0, mrr=0.0, ndcg=0.0, first_relevant_rank=0)
    return RetrievedEval(retrieved=retrieved, chunk_eval=chunk_eval, doc_eval=doc_eval)


def test_rank_setups_ranks_by_mean_recall_over_the_queries():
    """
    Input: three setups with two queries each.
    Output: the best setups.
    What it does: checks each setup is scored by the mean over its queries, not by one query, and the best comes first.
    """
    flatip = make_setup("faiss_flatip")
    hnsw = make_setup("faiss_hnsw")
    qdrant = make_setup("qdrant")
    retrieved_evals = [
        # flatip: one perfect query, one miss, mean recall 0.5
        make_retrieved_eval(flatip, "q1", 1.0, 1.0),
        make_retrieved_eval(flatip, "q2", 0.0, 0.0),
        # hnsw: mean recall 0.7
        make_retrieved_eval(hnsw, "q1", 0.6, 0.5),
        make_retrieved_eval(hnsw, "q2", 0.8, 0.5),
        # qdrant: mean recall 0.3
        make_retrieved_eval(qdrant, "q1", 0.3, 1.0),
        make_retrieved_eval(qdrant, "q2", 0.3, 1.0),
    ]

    assert rank_setups(retrieved_evals, 3) == [hnsw, flatip, qdrant]


def test_rank_setups_breaks_equal_recall_by_mrr():
    """
    Input: two setups with the same mean recall and different mean MRR.
    Output: the best setups.
    What it does: checks the higher MRR wins.
    """
    low_mrr = make_setup("faiss_flatip")
    high_mrr = make_setup("faiss_ivf")
    retrieved_evals = [
        make_retrieved_eval(low_mrr, "q1", 0.5, 0.2),
        make_retrieved_eval(high_mrr, "q1", 0.5, 0.9),
    ]

    assert rank_setups(retrieved_evals, 2) == [high_mrr, low_mrr]


def test_rank_setups_ranks_plain_and_reranked_together():
    """
    Input: a plain and a reranked setup of the same index, the reranked one scoring lower.
    Output: the best setup.
    What it does: checks reranked setups are not kept just for being reranked; only the better score is kept.
    """
    plain = make_setup("faiss_flatip")
    reranked = make_setup("faiss_flatip", reranking=True)
    retrieved_evals = [
        make_retrieved_eval(plain, "q1", 0.8, 0.5),
        make_retrieved_eval(reranked, "q1", 0.6, 0.9),
    ]

    assert rank_setups(retrieved_evals, 1) == [plain]


def test_rank_setups_keeps_all_when_there_are_fewer_than_n(capsys):
    """
    Input: two setups and n_top_setups 3.
    Output: the best setups and what was printed.
    What it does: checks both are kept and the ranking is printed.
    """
    flatip = make_setup("faiss_flatip")
    hnsw = make_setup("faiss_hnsw")
    retrieved_evals = [
        make_retrieved_eval(flatip, "q1", 0.9, 0.5),
        make_retrieved_eval(hnsw, "q1", 0.1, 0.5),
    ]

    best_setups = rank_setups(retrieved_evals, 3)

    assert best_setups == [flatip, hnsw]
    assert "Setups ranked : 2, kept : 2" in capsys.readouterr().out


def test_rank_setups_raises_when_n_is_below_one():
    """
    Input: n_top_setups 0.
    Output: ValueError.
    What it does: checks an n that keeps nothing is refused.
    """
    with pytest.raises(ValueError, match="n_top_setups"):
        rank_setups([], 0)


# ---------- save_best_setups and load_best_setups ----------


def make_manifest():
    """
    Input: none.
    Output: a BuildManifest with one sentence chunk entry and a FAISS HNSW, a BM25 and a Qdrant index entry.
    What it does: builds a small manifest, so the expected locations can be written out in the tests.
    """
    chunk_entry = ChunkEntry("sentence", 2, 6, 2, 0.5, "out/chunks/chunks_test", "out/embeddings/embeddings_test", 4)
    index_entries = [
        IndexEntry("sentence", "faiss_hnsw", {"hnsw_m": 4, "hnsw_ef_construction": 8}, "out/indexes/faiss_hnsw_test", 4),
        IndexEntry("sentence", "bm25", {"bm25_k1": 1.5, "bm25_b": 0.75, "bm25_tokenizer": "whitespace"}, "out/indexes/bm25_test", 4),
        IndexEntry("sentence", "qdrant", {}, "qdrant_test", 4),
    ]
    return BuildManifest(
        build_hash="testbuildhash", input_hash="testinputhash", dataset_name="dataset", dataset_size=2,
        documents_path="out/documents", embedding_model="fake-model", built_at="2026-11-20T00:00:00",
        chunk_entries=[chunk_entry], index_entries=index_entries,
    )


def make_any_setup(retrieval_type, index_type):
    """
    Input: retrieval_type; index_type.
    Output: a Setup of the sentence chunk type.
    What it does: builds one setup for the save tests.
    """
    return Setup(chunk_type="sentence", index_type=index_type, query_type="original", retrieval_type=retrieval_type, reranking=False)


def test_save_best_setups_finds_each_index_location(tmp_path):
    """
    Input: a FAISS, a BM25, a Qdrant and a hybrid setup.
    Output: their SetupRecords.
    What it does: checks each index location and the chunk folder come from the manifest, and that hybrid has the
    dense and then the BM25 location.
    """
    best_setups = [
        make_any_setup("dense", "faiss_hnsw"),
        make_any_setup("bm25", "bm25"),
        make_any_setup("dense", "qdrant"),
        make_any_setup("hybrid", "faiss_hnsw+bm25"),
    ]

    setup_records = save_best_setups(best_setups, make_manifest(), str(tmp_path), "testevalhash", "testbenchmarkhash")

    assert setup_records[0].index_paths == ("out/indexes/faiss_hnsw_test",)
    assert setup_records[1].index_paths == ("out/indexes/bm25_test",)
    assert setup_records[2].index_paths == ("qdrant_test",)
    assert setup_records[3].index_paths == ("out/indexes/faiss_hnsw_test", "out/indexes/bm25_test")
    for setup_record in setup_records:
        assert setup_record.chunk_path == "out/chunks/chunks_test"
    assert setup_records[3].setup == best_setups[3]


def test_save_best_setups_names_the_file_by_the_three_hashes(tmp_path):
    """
    Input: one best setup, the manifest and an eval dataset and benchmark hash.
    Output: the saved file.
    What it does: checks the file is best_setups/best_setups_{build_hash}_{eval_dataset_hash}_{benchmark_hash}.json.
    """
    save_best_setups([make_any_setup("dense", "qdrant")], make_manifest(), str(tmp_path), "testevalhash", "testbenchmarkhash")

    expected_file = os.path.join(str(tmp_path), "best_setups", "best_setups_testbuildhash_testevalhash_testbenchmarkhash.json")
    assert paths.get_best_setups_path(str(tmp_path), "testbuildhash", "testevalhash", "testbenchmarkhash") == expected_file
    assert os.path.exists(expected_file)


def test_save_then_load_best_setups_gives_the_same_records_in_order(tmp_path):
    """
    Input: two best setups saved.
    Output: the loaded records.
    What it does: checks the saved file loads back to the same SetupRecords, best first.
    """
    output_path = str(tmp_path)
    best_setups = [make_any_setup("hybrid", "faiss_hnsw+bm25"), make_any_setup("dense", "qdrant")]

    saved_records = save_best_setups(best_setups, make_manifest(), output_path, "testevalhash", "testbenchmarkhash")
    loaded_records = load_best_setups(output_path, "testbuildhash", "testevalhash", "testbenchmarkhash")

    assert loaded_records == saved_records
    assert isinstance(loaded_records[0], SetupRecord)
    assert isinstance(loaded_records[0].index_paths, tuple)


def test_load_best_setups_raises_when_nothing_is_saved(tmp_path):
    """
    Input: an output folder with no best setups file.
    Output: FileNotFoundError.
    What it does: checks serving is told to run the benchmark first instead of getting no setups.
    """
    with pytest.raises(FileNotFoundError, match="benchmark"):
        load_best_setups(str(tmp_path), "testbuildhash", "testevalhash", "testbenchmarkhash")
