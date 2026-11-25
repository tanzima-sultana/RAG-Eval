"""
test_build_config.py

Tests for common.args.parse_args, the build config parser: nested per-index-type blocks become --{block}.{key}
flags, a list inside a block becomes one flag followed by its values, a FAISS type's settings are optional when its
block is not given, faiss_types and vectordb_types may be empty, and the BM25 settings are required.
"""

import sys

import pytest

from common.args import parse_args


BUILD_CONFIG_WITHOUT_INDEXING = """
ingestion:
  dataset_size: 2
  dataset_name: dataset
  input_path: data/
  output_path: out/
  document_type: pdf
embedder:
  embedding_provider: huggingface
  model_name: fake-model
  batch_size: 256
  normalize_embeddings: true
chunking:
  chunk_types: [fixed]
  min_chunk_size: 2
  max_chunk_size: 6
  fixed_chunk_overlap: [2, 4]
vectordb_indexing:
  vectordb_types: [qdrant]
  qdrant_url: http://localhost:6333
  qdrant_batch_size: 3
"""

INDEXING_WITH_EVERY_BLOCK = """
faiss_indexing:
  faiss_types: [faiss_flatip, faiss_ivf, faiss_hnsw]
  faiss_ivf:
    ivf_nlist_multiplier: [4, 8, 16]
  faiss_hnsw:
    hnsw_m: 32
    hnsw_ef_construction: 200
bm25_indexing:
  bm25_k1: 1.5
  bm25_b: 0.75
  bm25_tokenizer: whitespace
"""

INDEXING_FLATIP_ONLY = """
faiss_indexing:
  faiss_types: [faiss_flatip]
bm25_indexing:
  bm25_k1: 1.5
  bm25_b: 0.75
  bm25_tokenizer: whitespace
"""

INDEXING_WITHOUT_BM25 = """
faiss_indexing:
  faiss_types: [faiss_flatip]
"""

INDEXING_WITH_EMPTY_FAISS_TYPES = """
faiss_indexing:
  faiss_types: []
bm25_indexing:
  bm25_k1: 1.5
  bm25_b: 0.75
  bm25_tokenizer: whitespace
"""


def parse_build_config(tmp_path, monkeypatch, indexing_section, override_arguments):
    """
    Input: tmp_path; pytest's monkeypatch fixture; indexing_section, the indexing part of the config;
    override_arguments, flags given after --config.
    Output: the parsed build args.
    What it does: writes a build config with that indexing section and parses it as the command line would.
    """
    config_file = tmp_path / "build_config.yaml"
    config_file.write_text(BUILD_CONFIG_WITHOUT_INDEXING + indexing_section, encoding="utf-8")
    command_line = ["run_build.py", "--config", str(config_file)]
    for override_argument in override_arguments:
        command_line.append(override_argument)
    monkeypatch.setattr(sys, "argv", command_line)
    return parse_args()


def test_parse_args_reads_each_nested_index_block(tmp_path, monkeypatch):
    """
    Input: a faiss_indexing section with the faiss_ivf and faiss_hnsw blocks, and a bm25_indexing section.
    Output: the parsed args.
    What it does: checks each setting inside a block is read with its type.
    """
    parsed_args = parse_build_config(tmp_path, monkeypatch, INDEXING_WITH_EVERY_BLOCK, [])

    assert parsed_args.faiss_types == ["faiss_flatip", "faiss_ivf", "faiss_hnsw"]
    assert parsed_args.ivf_nlist_multiplier == [4.0, 8.0, 16.0]
    assert parsed_args.hnsw_m == 32
    assert parsed_args.hnsw_ef_construction == 200
    assert parsed_args.bm25_k1 == 1.5
    assert parsed_args.bm25_b == 0.75
    assert parsed_args.bm25_tokenizer == "whitespace"
    assert parsed_args.vectordb_types == ["qdrant"]
    assert parsed_args.qdrant_url == "http://localhost:6333"
    assert parsed_args.qdrant_batch_size == 3


def test_parse_args_leaves_settings_of_missing_faiss_blocks_none(tmp_path, monkeypatch):
    """
    Input: a faiss_indexing section with only faiss_types [faiss_flatip] and no blocks.
    Output: the parsed args.
    What it does: checks the config parses, and every FAISS IVF and HNSW setting is None.
    """
    parsed_args = parse_build_config(tmp_path, monkeypatch, INDEXING_FLATIP_ONLY, [])

    assert parsed_args.faiss_types == ["faiss_flatip"]
    assert parsed_args.ivf_nlist_multiplier is None
    assert parsed_args.hnsw_m is None
    assert parsed_args.hnsw_ef_construction is None


def test_parse_args_reads_empty_faiss_types_as_an_empty_list(tmp_path, monkeypatch):
    """
    Input: a faiss_indexing section with faiss_types [].
    Output: the parsed args.
    What it does: checks an empty list is read as an empty list, so no FAISS index is built.
    """
    parsed_args = parse_build_config(tmp_path, monkeypatch, INDEXING_WITH_EMPTY_FAISS_TYPES, [])

    assert parsed_args.faiss_types == []


def test_parse_args_requires_the_bm25_settings(tmp_path, monkeypatch):
    """
    Input: a config with no bm25_indexing section.
    Output: SystemExit from the parser.
    What it does: checks BM25 is always built, so its settings are required.
    """
    with pytest.raises(SystemExit):
        parse_build_config(tmp_path, monkeypatch, INDEXING_WITHOUT_BM25, [])


def test_parse_args_overrides_a_block_setting_with_its_block_flag(tmp_path, monkeypatch):
    """
    Input: the config with every block, and the override --faiss_ivf.ivf_nlist_multiplier 2.
    Output: the parsed args.
    What it does: checks the override replaces the file's list.
    """
    parsed_args = parse_build_config(tmp_path, monkeypatch, INDEXING_WITH_EVERY_BLOCK, ["--faiss_ivf.ivf_nlist_multiplier", "2"])

    assert parsed_args.ivf_nlist_multiplier == [2.0]


def test_parse_args_reads_the_embedder_section(tmp_path, monkeypatch):
    """
    Input: the config with the embedder section.
    Output: the parsed args.
    What it does: checks each embedder setting is read, with normalize_embeddings as "true" or "false".
    """
    parsed_args = parse_build_config(tmp_path, monkeypatch, INDEXING_FLATIP_ONLY, [])

    assert parsed_args.embedding_provider == "huggingface"
    assert parsed_args.model_name == "fake-model"
    assert parsed_args.batch_size == 256
    assert parsed_args.normalize_embeddings == "true"


def test_parse_args_reads_the_fixed_chunk_overlap_list(tmp_path, monkeypatch):
    """
    Input: the config with fixed_chunk_overlap [2, 4].
    Output: the parsed args.
    What it does: checks every overlap of the list is read as an int.
    """
    parsed_args = parse_build_config(tmp_path, monkeypatch, INDEXING_FLATIP_ONLY, [])

    assert parsed_args.fixed_chunk_overlap == [2, 4]
