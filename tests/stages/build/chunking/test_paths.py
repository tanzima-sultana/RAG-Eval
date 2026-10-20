"""
test_paths.py

Tests for common.paths.get_chunk_path, the folder chunks are saved in.
"""

from common.paths import get_chunk_path


def test_get_chunk_path_builds_folder_from_every_chunking_setting(make_chunking_args):
    """
    Input: output_path "out" and ChunkingArgs with known values.
    Output: the chunk folder path.
    What it does: checks the path holds chunk_type, model_name and the
    min/max/overlap/threshold values, and ends with "/".
    """
    chunking_args = make_chunking_args(
        chunk_type="sentence",
        model_name="fake-model",
        min_chunk_size=2,
        max_chunk_size=6,
        fixed_chunk_overlap=1,
        semantic_threshold=0.5,
    )

    chunk_path = get_chunk_path("out", chunking_args)

    assert chunk_path == "out/chunks/sentence/fake-model/2_6_1_0.5/"


def test_get_chunk_path_adds_framework_folder_for_langchain(make_chunking_args):
    """
    Input: ChunkingArgs with framework "langchain".
    Output: the chunk folder path.
    What it does: checks LangChain chunks go under chunks/langchain/, apart
    from the built-in chunkers' folder.
    """
    chunking_args = make_chunking_args(framework="langchain", chunk_type="fixed", min_chunk_size=2, max_chunk_size=6, fixed_chunk_overlap=1, semantic_threshold=0.5)

    chunk_path = get_chunk_path("out", chunking_args)

    assert chunk_path == "out/chunks/langchain/fixed/fake-model/2_6_1_0.5/"


def test_get_chunk_path_differs_by_model_name(make_chunking_args):
    """
    Input: the same ChunkingArgs except for model_name.
    Output: two chunk folder paths.
    What it does: checks chunks from different models go to different folders.
    """
    first_path = get_chunk_path("out", make_chunking_args(model_name="model-a"))
    second_path = get_chunk_path("out", make_chunking_args(model_name="model-b"))

    assert first_path != second_path
