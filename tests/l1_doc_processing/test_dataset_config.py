"""
test_dataset_config.py

Tests for common.dataset_config: the registered datasets and
get_dataset_config.
"""

import pytest

from common.dataset_config import get_dataset_config


def test_docx_corpus_is_registered_as_english_url_downloads():
    """
    Input: dataset_name "docx-corpus".
    Output: get_dataset_config's return value.
    """
    config = get_dataset_config("docx-corpus")

    assert config.identifier == "superdoc-dev/docx-corpus"
    assert config.doc_type == "docx"
    assert config.extension == "docx"
    assert config.content_field == "url"
    assert config.content_encoding == "url"
    assert config.split == "train"
    assert config.row_filter == {"language": "en"}


def test_existing_datasets_have_no_row_filter():
    """
    Input: dataset_names "cuad" and "pdfa-eng-wds".
    Output: each config's row_filter.
    """
    assert get_dataset_config("cuad").row_filter is None
    assert get_dataset_config("pdfa-eng-wds").row_filter is None


def test_get_dataset_config_raises_for_an_unregistered_name():
    """
    Input: a dataset_name that is not registered.
    Output: none; ValueError is expected.
    """
    with pytest.raises(ValueError):
        get_dataset_config("not-registered")
