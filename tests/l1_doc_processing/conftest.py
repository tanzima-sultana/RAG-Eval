"""
conftest.py

Shared pytest fixtures for the l1_doc_processing test suite.
"""

import pytest


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    """
    tmp_path: pytest's built-in temporary directory for this test.
    monkeypatch: pytest's built-in fixture for reverting changes after
        the test.

    Changes the current working directory to tmp_path for the
    duration of the test, so any code under test that writes to the
    relative "data/raw/" or "data/processed/" directories (see
    common.paths) writes into an isolated temporary directory instead
    of this repo's real data/ folder.

    Returns tmp_path, the isolated working directory.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path
