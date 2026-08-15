import os
import pickle
from dataclasses import dataclass

import nltk
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from constants import CHUNKED_DATA_PATH

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_embedding_model = None


@dataclass
class Chunk:
    doc_id: object
    chunk_id: str
    text: str


def _ensure_nltk_data():
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def fixed_chunking(text: str, max_chunk_size: int, fixed_chunk_overlap: int) -> list[str]:
    """Split text into fixed-size character windows with a fixed overlap between
    consecutive windows, ignoring sentence/word boundaries."""
    if fixed_chunk_overlap >= max_chunk_size:
        raise ValueError("fixed_chunk_overlap must be smaller than max_chunk_size")

    text = text.strip()
    if not text:
        return []

    step = max_chunk_size - fixed_chunk_overlap
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + max_chunk_size, text_length)
        chunks.append(text[start:end])
        if end == text_length:
            break
        start += step

    return chunks


def sentence_aware_chunking(text: str, max_chunk_size: int) -> list[str]:
    """Pack whole sentences into chunks up to max_chunk_size, never splitting a
    sentence across chunks unless a single sentence exceeds max_chunk_size on its
    own, in which case it falls back to a fixed-size split for that sentence."""
    _ensure_nltk_data()

    text = text.strip()
    if not text:
        return []

    sentences = nltk.sent_tokenize(text)

    chunks = []
    current_sentences: list[str] = []
    current_length = 0

    for sentence in sentences:
        if len(sentence) > max_chunk_size:
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences, current_length = [], 0
            chunks.extend(fixed_chunking(sentence, max_chunk_size, fixed_chunk_overlap=0))
            continue

        candidate_length = len(sentence) if not current_sentences else current_length + 1 + len(sentence)
        if candidate_length > max_chunk_size and current_sentences:
            chunks.append(" ".join(current_sentences))
            current_sentences, current_length = [sentence], len(sentence)
        else:
            current_sentences.append(sentence)
            current_length = candidate_length

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def semantic_chunking(text: str, max_chunk_size: int, semantic_threshold: float) -> list[str]:
    """Split text into sentences and group consecutive sentences into the same
    chunk as long as their embedding cosine similarity stays at or above
    semantic_threshold and the chunk stays within max_chunk_size; otherwise start
    a new chunk."""
    _ensure_nltk_data()

    text = text.strip()
    if not text:
        return []

    sentences = nltk.sent_tokenize(text)
    if len(sentences) == 1:
        return fixed_chunking(sentences[0], max_chunk_size, fixed_chunk_overlap=0)

    model = _get_embedding_model()
    embeddings = model.encode(sentences)

    chunks = []
    current_sentences = [sentences[0]]
    current_length = len(sentences[0])

    for i in range(1, len(sentences)):
        sentence = sentences[i]
        similarity = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
        candidate_length = current_length + 1 + len(sentence)

        if similarity < semantic_threshold or candidate_length > max_chunk_size:
            chunks.append(" ".join(current_sentences))
            current_sentences, current_length = [sentence], len(sentence)
        else:
            current_sentences.append(sentence)
            current_length = candidate_length

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def chunk_dataset(
    dataset_path: str,
    max_chunk_size: int,
    fixed_chunk_overlap: float,
    semantic_threshold: float,
) -> dict[str, str]:
    """Chunk every document in dataset_path with each of the three chunking
    strategies, save each strategy's output to its own file next to
    CHUNKED_DATA_PATH, and return a dict mapping strategy name -> output path."""
    df = pd.read_parquet(dataset_path)

    strategies = {
        "fixed": lambda text: fixed_chunking(text, max_chunk_size, fixed_chunk_overlap),
        "sentence": lambda text: sentence_aware_chunking(text, max_chunk_size),
        "semantic": lambda text: semantic_chunking(text, max_chunk_size, semantic_threshold),
    }

    output_dir = os.path.dirname(CHUNKED_DATA_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    output_paths = {}
    for strategy_name, chunk_fn in strategies.items():
        chunks: list[Chunk] = []
        for doc_id, text in zip(df["id"], df["text"]):
            for chunk_index, chunk_text in enumerate(chunk_fn(text)):
                chunk_id = str(doc_id) + "_" + str(chunk_index)
                chunks.append(Chunk(doc_id=doc_id, chunk_id=chunk_id, text=chunk_text))

        chunk_map = {chunk.chunk_id: chunk.text for chunk in chunks}

        output_path = f"{CHUNKED_DATA_PATH}_{strategy_name}.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(chunk_map, f)
        output_paths[strategy_name] = output_path

        avg_chunk_size = (sum(len(text) for text in chunk_map.values()) / len(chunk_map)) if chunk_map else 0.0
        file_size_bytes = os.path.getsize(output_path)
        print(
            f"[{strategy_name}] chunks: {len(chunk_map)}, "
            f"avg chunk size: {avg_chunk_size:.1f} chars, "
            f"file size: {file_size_bytes / 1024:.2f} KB"
        )

    return output_paths
