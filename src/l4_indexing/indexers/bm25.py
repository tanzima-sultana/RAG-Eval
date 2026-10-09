"""
bm25.py

Builds, saves and loads the BM25 index of one indexing run. The index is
saved in the BM25 index folder (see common.paths.get_bm25_index_path) as
two files:
bm25.pkl, the BM25Okapi model, saved with pickle, and
chunk_ids.json, the chunk_id of each BM25 position, in order.

Each file is first written to a temporary file and then renamed into
place, so a crash mid-write never leaves a half-written file.
"""

import json
import os
import pickle

from rank_bm25 import BM25Okapi

from common.args import IndexArgs
from l2_chunking.schema import Chunk
from l4_indexing.schema import Bm25Index

def get_bm25_index(bm25_index_path: str, index_args: IndexArgs, chunks: dict[str, Chunk]) -> Bm25Index:
	"""
	Input: bm25_index_path, the BM25 index folder; index_args, the indexing
	parameters; chunks, a dict of chunk_id to Chunk from l2_chunking.
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

	model_file_path = os.path.join(bm25_index_path, "bm25.pkl")
	chunk_ids_file_path = os.path.join(bm25_index_path, "chunk_ids.json")

	# 1. Collect the chunk ids in order
	# BM25 position i will hold the words of chunk_ids[i]
	chunk_ids = []
	for chunk_id in chunks:
		chunk_ids.append(chunk_id)

	# 2. Load the saved model if it holds exactly the same chunks
	# The saved chunk_ids are used, not the new ones, because they match
	# the positions in the saved model
	if os.path.exists(model_file_path) and os.path.exists(chunk_ids_file_path):
		with open(chunk_ids_file_path, "r", encoding="utf-8") as f:
			saved_chunk_ids = json.load(f)

		if set(saved_chunk_ids) == set(chunk_ids):
			with open(model_file_path, "rb") as f:
				saved_model = pickle.load(f)
			print(f"BM25 index loaded from cache : {len(saved_chunk_ids)} chunks")
			return Bm25Index(
				bm25_model=saved_model,
				chunk_ids=saved_chunk_ids,
				index_file_path=model_file_path
			)

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
	os.makedirs(bm25_index_path, exist_ok=True)

	temporary_model_file = model_file_path + ".tmp"
	with open(temporary_model_file, "wb") as f:
		pickle.dump(new_model, f)
	os.replace(temporary_model_file, model_file_path)

	temporary_chunk_ids_file = chunk_ids_file_path + ".tmp"
	with open(temporary_chunk_ids_file, "w", encoding="utf-8") as f:
		json.dump(chunk_ids, f)
	os.replace(temporary_chunk_ids_file, chunk_ids_file_path)

	return Bm25Index(
		bm25_model=new_model,
		chunk_ids=chunk_ids,
		index_file_path=model_file_path
	)
