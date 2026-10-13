"""
test_validate.py

Tests for l4_indexing.validate.validate_index_args.
"""

import pytest

from l4_indexing.validate import validate_index_args


@pytest.mark.parametrize("index_type", ["faiss_flatip", "faiss_ivf", "faiss_hnsw", "bm25", "qdrant"])
def test_valid_args_pass_for_each_index_type(make_index_args, index_type):
    """
    Input: valid IndexArgs for each supported index_type.
    Output: nothing raised.
    What it does: checks valid args pass every check.
    """
    validate_index_args(make_index_args(index_type=index_type))


def test_unknown_index_type_raises(make_index_args):
    """
    Input: index_type "annoy".
    Output: ValueError mentioning index_type.
    What it does: checks only the five index types are accepted.
    """
    with pytest.raises(ValueError, match="index_type"):
        validate_index_args(make_index_args(index_type="annoy"))


def test_ivf_nlist_zero_raises_for_ivf(make_index_args):
    """
    Input: faiss_ivf with ivf_nlist 0.
    Output: ValueError mentioning ivf_nlist.
    What it does: checks IVF needs at least one cluster.
    """
    with pytest.raises(ValueError, match="ivf_nlist"):
        validate_index_args(make_index_args(index_type="faiss_ivf", ivf_nlist=0))


def test_ivf_nlist_zero_passes_for_other_types(make_index_args):
    """
    Input: faiss_flatip with ivf_nlist 0.
    Output: nothing raised.
    What it does: checks ivf_nlist is only checked for faiss_ivf.
    """
    validate_index_args(make_index_args(index_type="faiss_flatip", ivf_nlist=0))


def test_hnsw_m_zero_raises_for_hnsw(make_index_args):
    """
    Input: faiss_hnsw with hnsw_m 0.
    Output: ValueError mentioning hnsw_m.
    What it does: checks HNSW needs at least one neighbour per node.
    """
    with pytest.raises(ValueError, match="hnsw_m"):
        validate_index_args(make_index_args(index_type="faiss_hnsw", hnsw_m=0))


def test_hnsw_ef_construction_below_m_raises(make_index_args):
    """
    Input: faiss_hnsw with hnsw_m 8 and hnsw_ef_construction 4.
    Output: ValueError mentioning hnsw_ef_construction.
    What it does: checks hnsw_ef_construction must be at least hnsw_m.
    """
    with pytest.raises(ValueError, match="hnsw_ef_construction"):
        validate_index_args(make_index_args(index_type="faiss_hnsw", hnsw_m=8, hnsw_ef_construction=4))


def test_negative_bm25_k1_raises(make_index_args):
    """
    Input: bm25_k1 -0.1.
    Output: ValueError mentioning bm25_k1.
    What it does: checks bm25_k1 must not be negative.
    """
    with pytest.raises(ValueError, match="bm25_k1"):
        validate_index_args(make_index_args(index_type="bm25", bm25_k1=-0.1))


@pytest.mark.parametrize("bm25_b", [-0.1, 1.1])
def test_bm25_b_outside_zero_to_one_raises(make_index_args, bm25_b):
    """
    Input: bm25_b below 0 or above 1.
    Output: ValueError mentioning bm25_b.
    What it does: checks bm25_b must be between 0 and 1.
    """
    with pytest.raises(ValueError, match="bm25_b"):
        validate_index_args(make_index_args(index_type="bm25", bm25_b=bm25_b))


def test_empty_bm25_tokenizer_raises(make_index_args):
    """
    Input: a whitespace-only bm25_tokenizer.
    Output: ValueError mentioning bm25_tokenizer.
    What it does: checks a tokenizer name must be given.
    """
    with pytest.raises(ValueError, match="bm25_tokenizer"):
        validate_index_args(make_index_args(index_type="bm25", bm25_tokenizer="  "))


def test_qdrant_batch_size_zero_raises(make_index_args):
    """
    Input: qdrant_batch_size 0.
    Output: ValueError mentioning qdrant_batch_size.
    What it does: checks the upload batch size must be positive.
    """
    with pytest.raises(ValueError, match="qdrant_batch_size"):
        validate_index_args(make_index_args(index_type="qdrant", qdrant_batch_size=0))
