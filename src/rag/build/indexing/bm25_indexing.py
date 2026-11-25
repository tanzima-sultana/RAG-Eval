import os
import pickle

from langchain_community.retrievers import BM25Retriever

from common import paths
from common.args import Bm25IndexingArgs
from schema.chunk import Chunk
from schema.index import Bm25Index


def lowercase_whitespace_tokenize(text: str) -> list[str]:
	"""
	Input: text, one chunk or query text.
	Output: the list of lowercase words in text.
	What it does: lowercases text and splits it on whitespace. A named function, because pickle cannot save a lambda.
	"""
	return text.lower().split()


def save_bm25_index(bm25_index_path: str, retriever: BM25Retriever) -> None:
	"""
	Input: bm25_index_path, the BM25 index folder; retriever, the BM25Retriever.
	Output: nothing.
	What it does: saves the retriever with pickle to a temporary file, then renames it to bm25_retriever.pkl.
	"""
	os.makedirs(bm25_index_path, exist_ok=True)
	retriever_file_path = os.path.join(bm25_index_path, "bm25_retriever.pkl")
	temporary_retriever_file_path = retriever_file_path + ".tmp"
	with open(temporary_retriever_file_path, "wb") as f:
		pickle.dump(retriever, f)
	os.replace(temporary_retriever_file_path, retriever_file_path)


def load_bm25_index(bm25_index_path: str, chunk_ids: list[str]) -> BM25Retriever | None:
	"""
	Input: bm25_index_path, the BM25 index folder; chunk_ids, the chunk ids the retriever must hold.
	Output: the saved BM25Retriever, or None if there is none or it holds different chunk ids.
	What it does: loads bm25_retriever.pkl and compares its chunk ids with chunk_ids.
	"""
	retriever_file_path = os.path.join(bm25_index_path, "bm25_retriever.pkl")
	if not os.path.exists(retriever_file_path):
		return None

	# The file was written by this code, so reading it with pickle is safe here
	with open(retriever_file_path, "rb") as f:
		saved_retriever = pickle.load(f)

	saved_chunk_ids = []
	for document in saved_retriever.docs:
		saved_chunk_ids.append(document.metadata["chunk_id"])
	if set(saved_chunk_ids) != set(chunk_ids):
		return None

	return saved_retriever


def load_saved_bm25_index(chunks: dict[str, Chunk], index_folder: str) -> Bm25Index:
	"""
	Input: chunks, a dict of chunk_id to Chunk, the chunks the index must hold; index_folder, the index folder from the
	build manifest.
	Output: a Bm25Index.
	What it does: loads the BM25 index the build saved, and raises FileNotFoundError if it is not saved or holds
	different chunks. It never builds.
	"""
	chunk_ids = []
	for chunk_id in chunks:
		chunk_ids.append(chunk_id)

	retriever = load_bm25_index(index_folder, chunk_ids)
	if retriever is None:
		raise FileNotFoundError(
			f"No saved BM25 index holding these chunks in {index_folder}. Run scripts/build.sh first."
		)
	print(f"BM25 index loaded from cache : {len(chunk_ids)} chunks")
	return Bm25Index(retriever=retriever, index_path=os.path.join(index_folder, "bm25_retriever.pkl"))


def load_bm25_index_from_path(index_folder: str) -> Bm25Index:
	"""
	Input: index_folder, the index folder the build saved.
	Output: a Bm25Index.
	What it does: loads the saved BM25 index straight from its pickle, for serving. It does not check which chunks it
	holds, and raises FileNotFoundError if the file is not there.
	"""
	retriever_file_path = os.path.join(index_folder, "bm25_retriever.pkl")
	if not os.path.exists(retriever_file_path):
		raise FileNotFoundError(f"No saved BM25 index at {retriever_file_path}. Run scripts/build.sh first.")
	# The file was written by this code, so reading it with pickle is safe here
	with open(retriever_file_path, "rb") as f:
		retriever = pickle.load(f)
	print(f"BM25 index loaded : {index_folder}")
	return Bm25Index(retriever=retriever, index_path=retriever_file_path)


class Bm25Indexing:
	def __init__(self, bm25_args: Bm25IndexingArgs, chunking_hash: str):
		"""
		Input: bm25_args, the BM25 indexing settings; chunking_hash, the hash of the chunks the index is built from, from
		chunking.
		Output: nothing.
		What it does: stores the two and validates the settings.
		"""
		self.bm25_args = bm25_args
		self.chunking_hash = chunking_hash
		self.validate()

	def validate(self) -> None:
		"""
		Input: none.
		Output: nothing.
		What it does: raises ValueError if a BM25 setting is missing or not valid.
		"""
		bm25_args = self.bm25_args

		if bm25_args.bm25_k1 is None:
			raise ValueError("bm25_k1 is missing. Give it in the bm25_indexing config.")
		if bm25_args.bm25_b is None:
			raise ValueError("bm25_b is missing. Give it in the bm25_indexing config.")
		if bm25_args.bm25_tokenizer != "whitespace":
			raise ValueError(f"bm25_tokenizer ({bm25_args.bm25_tokenizer}) is not supported. Use \"whitespace\".")

	def bm25_indexing(self, chunks: dict[str, Chunk], output_path: str) -> tuple[Bm25Index, str, str]:
		"""
		Input: chunks, a dict of chunk_id to Chunk; output_path, the dataset's output folder.
		Output: (bm25_index, bm25_hash, bm25_index_path): a Bm25Index holding a LangChain BM25Retriever, its hash, and the
		folder it is saved in.
		What it does: loads the saved retriever if it holds the same chunks, otherwise builds and saves a new one.
		"""
		if len(chunks) == 0:
			raise ValueError("chunks is empty. There is nothing to index.")

		# 1. Collect the chunk ids, texts and metadata in the same order
		chunk_ids = []
		chunk_texts = []
		chunk_metadatas = []
		for chunk_id in chunks:
			chunk_ids.append(chunk_id)
			chunk_texts.append(chunks[chunk_id].chunk_texts)
			chunk_metadatas.append({"chunk_id": chunk_id})

		# 2. Load the saved retriever if it holds exactly the same chunks
		bm25_hash = paths.get_bm25_hash(self.bm25_args, self.chunking_hash)
		bm25_index_path = paths.get_bm25_index_path(output_path, bm25_hash)
		print(f"Index folder : {bm25_index_path}")
		retriever = load_bm25_index(bm25_index_path, chunk_ids)
		if retriever is not None:
			print(f"BM25 index loaded from cache : {len(chunk_ids)} chunks")
		else:
			# 3. Build the retriever and save it
			retriever = BM25Retriever.from_texts(
				texts=chunk_texts,
				metadatas=chunk_metadatas,
				ids=chunk_ids,
				bm25_params={"k1": self.bm25_args.bm25_k1, "b": self.bm25_args.bm25_b},
				preprocess_func=lowercase_whitespace_tokenize
			)
			save_bm25_index(bm25_index_path, retriever)
			print(f"BM25 index built : {len(chunk_ids)} chunks")

		bm25_index = Bm25Index(
			retriever=retriever,
			index_path=os.path.join(bm25_index_path, "bm25_retriever.pkl")
		)
		return bm25_index, bm25_hash, bm25_index_path
