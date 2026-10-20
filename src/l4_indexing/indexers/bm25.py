"""
bm25.py

Builds the BM25 index of one indexing run: a BM25Okapi model over the
chunk texts. Saving and loading are in l4_indexing/cache.py.
"""

import os

from rank_bm25 import BM25Okapi

from common.args import IndexArgs
from stages.build.chunking.schema import Chunk
from l4_indexing import cache
from l4_indexing.schema import Bm25Index

def get_bm25_index(bm25_index_path: str, index_args: IndexArgs, chunks: dict[str, Chunk]) -> Bm25Index:
	"""
	Input: bm25_index_path, the BM25 index folder; index_args, the indexing
	parameters; chunks, a dict of chunk_id to Chunk from stages.build.chunking.
	Output: a Bm25Index holding the BM25Okapi model.
	What it does: loads the saved model from bm25_index_path if it holds
	exactly the same chunk_ids as chunks. Otherwise splits each chunk's
	text into words with bm25_tokenizer, builds a BM25Okapi model with
	bm25_k1 and bm25_b, saves it, and returns it.
	The "whitespace" tokenizer lowercases the text and splits it on spaces,
	tabs and new lines. Queries must be split the same way at search time.
	"""
	if len(chunks) == 0:
		raise ValueError("chunks is empty. There is nothing to index.")

	if index_args.bm25_tokenizer != "whitespace":
		raise ValueError(
			f"bm25_tokenizer ({index_args.bm25_tokenizer}) is not supported. Use \"whitespace\"."
		)

	# 1. Collect the chunk ids in order
	# BM25 position i will hold the words of chunk_ids[i]
	chunk_ids = []
	for chunk_id in chunks:
		chunk_ids.append(chunk_id)

	# 2. Load the saved model if it holds exactly the same chunks
	cached_bm25_index = cache._load_cached_bm25_index(bm25_index_path, chunk_ids)
	if cached_bm25_index is not None:
		print(f"BM25 index loaded from cache : {len(cached_bm25_index.chunk_ids)} chunks")
		return cached_bm25_index

	# 3. Split each chunk's text into lowercase words
	tokenized_chunks = []
	for chunk_id in chunk_ids:
		chunk_text = chunks[chunk_id].text
		chunk_words = chunk_text.lower().split()
		tokenized_chunks.append(chunk_words)

	# 4. Build the BM25 model
	new_model = BM25Okapi(tokenized_chunks, k1=index_args.bm25_k1, b=index_args.bm25_b)
	print(f"BM25 index built : {len(chunk_ids)} chunks")

	# 5. Save the model and its chunk ids
	cache._save_bm25_index(bm25_index_path, new_model, chunk_ids)

	return Bm25Index(
		bm25_model=new_model,
		chunk_ids=chunk_ids,
		index_file_path=os.path.join(bm25_index_path, "bm25.pkl")
	)
