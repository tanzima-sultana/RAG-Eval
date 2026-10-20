"""
test_semantic.py

Tests for stages.build.chunking.chunkers.semantic.SemanticChunking. Embeddings come
from the conftest FakeSentenceTransformer: sentences with "cat" are
[1, 0] and all others [0, 1], so a switch between them is a topic change.
"""

import numpy as np
from nltk.tokenize import sent_tokenize

from stages.build.chunking.chunkers.semantic import SemanticChunking
from stages.build.chunking.schema import Chunk


def get_chunk_texts(chunks):
    """
    Input: chunks, a dict of chunk_id to Chunk.
    Output: the list of chunk texts in order.
    What it does: collects each chunk's text.
    """
    chunk_texts = []
    for chunk_id in chunks:
        chunk_texts.append(chunks[chunk_id].text)
    return chunk_texts


def make_semantic_chunker(make_chunker, chunking_args, document):
    """
    Input: the make_chunker fixture, chunking_args and one document.
    Output: a SemanticChunking ready to chunk that document.
    What it does: embeds the document's sentences the way Chunking does and
    passes the embeddings and offsets to the chunker.
    """
    probe_chunker = make_chunker(SemanticChunking, chunking_args)
    sentences = sent_tokenize(document.text)
    all_embeddings = probe_chunker.model.encode(sentences)
    embeddings_offset = {}
    embeddings_offset[document.id] = (0, len(sentences))
    return make_chunker(SemanticChunking, chunking_args, all_embeddings, embeddings_offset)


# ---------- _is_dissimilar ----------


def test_is_dissimilar_false_for_same_direction(make_chunker, make_chunking_args):
    """
    Input: two identical embeddings with threshold 0.5.
    Output: _is_dissimilar's result.
    What it does: checks cosine similarity 1 is not dissimilar.
    """
    chunker = make_chunker(SemanticChunking, make_chunking_args(chunk_type="semantic", semantic_threshold=0.5))
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0]])

    assert not chunker._is_dissimilar(embeddings, 0, 1)


def test_is_dissimilar_true_for_orthogonal(make_chunker, make_chunking_args):
    """
    Input: two orthogonal embeddings with threshold 0.5.
    Output: _is_dissimilar's result.
    What it does: checks cosine similarity 0 is dissimilar.
    """
    chunker = make_chunker(SemanticChunking, make_chunking_args(chunk_type="semantic", semantic_threshold=0.5))
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])

    assert chunker._is_dissimilar(embeddings, 0, 1)


def test_is_dissimilar_false_for_zero_vector(make_chunker, make_chunking_args):
    """
    Input: an all-zero embedding and a normal one.
    Output: _is_dissimilar's result.
    What it does: checks a zero vector does not divide by zero and is not dissimilar.
    """
    chunker = make_chunker(SemanticChunking, make_chunking_args(chunk_type="semantic", semantic_threshold=0.5))
    embeddings = np.array([[0.0, 0.0], [1.0, 0.0]])

    assert not chunker._is_dissimilar(embeddings, 0, 1)


# ---------- merge_chunks ----------


def test_merge_chunks_merges_small_chunk_into_previous(make_chunker, make_chunking_args):
    """
    Input: chunks of size 5, 1 and 5 with min 2 and max 10.
    Output: the merged chunks.
    What it does: checks the size-1 chunk joins the chunk before it,
    which keeps its own id.
    """
    chunker = make_chunker(SemanticChunking, make_chunking_args(chunk_type="semantic", min_chunk_size=2, max_chunk_size=10))
    chunks = {}
    chunks["doc_0"] = Chunk("doc_0", "first", 5, "semantic")
    chunks["doc_1"] = Chunk("doc_1", "tiny", 1, "semantic")
    chunks["doc_2"] = Chunk("doc_2", "third", 5, "semantic")

    merged_chunks = chunker.merge_chunks(chunks)

    assert list(merged_chunks.keys()) == ["doc_0", "doc_2"]
    assert merged_chunks["doc_0"].text == "first tiny"
    assert merged_chunks["doc_0"].size == 6


def test_merge_chunks_does_not_merge_past_max_size(make_chunker, make_chunking_args):
    """
    Input: chunks of size 9 and 1 with min 2 and max 9.
    Output: the merged chunks.
    What it does: checks the small chunk stays separate when merging
    would go over max_chunk_size.
    """
    chunker = make_chunker(SemanticChunking, make_chunking_args(chunk_type="semantic", min_chunk_size=2, max_chunk_size=9))
    chunks = {}
    chunks["doc_0"] = Chunk("doc_0", "big", 9, "semantic")
    chunks["doc_1"] = Chunk("doc_1", "tiny", 1, "semantic")

    merged_chunks = chunker.merge_chunks(chunks)

    assert list(merged_chunks.keys()) == ["doc_0", "doc_1"]


# ---------- chunk ----------


def test_semantic_starts_new_chunk_at_topic_change(make_chunker, make_chunking_args, make_document):
    """
    Input: two cat sentences then two dog sentences, all fitting in max 20.
    Output: the chunks.
    What it does: checks the chunk is split where the topic changes, even
    though the size limit was not reached.
    """
    chunking_args = make_chunking_args(chunk_type="semantic", min_chunk_size=1, max_chunk_size=20, semantic_threshold=0.5)
    document = make_document("doc", "The cat sat. The cat ran. Dogs bark loudly. Dogs run fast.")
    chunker = make_semantic_chunker(make_chunker, chunking_args, document)

    chunks = chunker.chunk(document)

    assert get_chunk_texts(chunks) == ["The cat sat. The cat ran.", "Dogs bark loudly. Dogs run fast."]
    assert chunks["doc_0"].type == "semantic"


def test_semantic_keeps_one_topic_together(make_chunker, make_chunking_args, make_document):
    """
    Input: three dog sentences, all fitting in max 20.
    Output: the chunks.
    What it does: checks similar sentences stay in one chunk.
    """
    chunking_args = make_chunking_args(chunk_type="semantic", min_chunk_size=1, max_chunk_size=20, semantic_threshold=0.5)
    document = make_document("doc", "Dogs bark. Dogs run. Dogs sleep.")
    chunker = make_semantic_chunker(make_chunker, chunking_args, document)

    chunks = chunker.chunk(document)

    assert get_chunk_texts(chunks) == ["Dogs bark. Dogs run. Dogs sleep."]


def test_semantic_splits_a_too_long_sentence_with_fixed_windows(make_chunker, make_chunking_args, make_document):
    """
    Input: a 10-word sentence with max_chunk_size 4 and overlap 1.
    Output: the chunks.
    What it does: checks the sentence is split into overlapping windows.
    """
    chunking_args = make_chunking_args(chunk_type="semantic", min_chunk_size=1, max_chunk_size=4, fixed_chunk_overlap=1, semantic_threshold=0.5)
    document = make_document("doc", "a b c d e f g h i j.")
    chunker = make_semantic_chunker(make_chunker, chunking_args, document)

    chunks = chunker.chunk(document)

    assert get_chunk_texts(chunks) == ["a b c d", "d e f g", "g h i j."]
