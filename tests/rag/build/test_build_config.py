"""
test_build_config.py

Tests for common.args.parse_args, the build config parser: nested per-index-type blocks become --{block}.{key}
flags, and an index type's settings are optional when its block is not given.
"""

import sys

from common.args import parse_args


BUILD_CONFIG_WITHOUT_INDEXING = """
ingestion:
  dataset_size: 2
  dataset_name: dataset
  input_path: data/
  output_path: out/
  document_type: pdf
chunking:
  model_name: fake-model
  chunk_types: [fixed]
  min_chunk_size: 2
  max_chunk_size: 6
  fixed_chunk_overlap: 2
  semantic_threshold: 0.5
vectordb:
  vectordb_types: [qdrant]
  qdrant_url: http://localhost:6333
  qdrant_batch_size: 3
"""

INDEXING_WITH_EVERY_BLOCK = """
indexing:
  index_types: [faiss_flatip, faiss_ivf, faiss_hnsw, bm25]
  faiss_ivf:
    ivf_nlist: 256
  faiss_hnsw:
    hnsw_m: 32
    hnsw_ef_construction: 200
  bm25:
    bm25_k1: 1.5
    bm25_b: 0.75
    bm25_tokenizer: whitespace
"""

INDEXING_FLATIP_ONLY = """
indexing:
  index_types: [faiss_flatip]
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
    Input: an indexing section with the faiss_ivf, faiss_hnsw and bm25 blocks.
    Output: the parsed args.
    What it does: checks each setting inside a block is read with its type.
    """
    parsed_args = parse_build_config(tmp_path, monkeypatch, INDEXING_WITH_EVERY_BLOCK, [])

    assert parsed_args.index_types == ["faiss_flatip", "faiss_ivf", "faiss_hnsw", "bm25"]
    assert parsed_args.ivf_nlist == 256
    assert parsed_args.hnsw_m == 32
    assert parsed_args.hnsw_ef_construction == 200
    assert parsed_args.bm25_k1 == 1.5
    assert parsed_args.bm25_b == 0.75
    assert parsed_args.bm25_tokenizer == "whitespace"


def test_parse_args_leaves_settings_of_missing_blocks_none(tmp_path, monkeypatch):
    """
    Input: an indexing section with only index_types [faiss_flatip] and no blocks.
    Output: the parsed args.
    What it does: checks the config parses, and every per-index-type setting is None.
    """
    parsed_args = parse_build_config(tmp_path, monkeypatch, INDEXING_FLATIP_ONLY, [])

    assert parsed_args.index_types == ["faiss_flatip"]
    assert parsed_args.ivf_nlist is None
    assert parsed_args.hnsw_m is None
    assert parsed_args.hnsw_ef_construction is None
    assert parsed_args.bm25_k1 is None
    assert parsed_args.bm25_b is None
    assert parsed_args.bm25_tokenizer is None


def test_parse_args_overrides_a_block_setting_with_its_block_flag(tmp_path, monkeypatch):
    """
    Input: the config with every block, and the override --faiss_ivf.ivf_nlist 128.
    Output: the parsed args.
    What it does: checks the override replaces the file's value.
    """
    parsed_args = parse_build_config(tmp_path, monkeypatch, INDEXING_WITH_EVERY_BLOCK, ["--faiss_ivf.ivf_nlist", "128"])

    assert parsed_args.ivf_nlist == 128
