import dataclasses
import functools
import json
import logging
import os

import torch
import sentence_transformers

from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import NLTKTextSplitter
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_text_splitters import Tokenizer
from langchain_text_splitters import split_text_on_tokens

from common import constants
from common import paths_temp
from common.args_temp import ChunkingArgs
from schema.document import Document
from schema.chunk import Chunk


class SentenceTransformerEmbeddings(Embeddings):
	"""
	A LangChain Embeddings object around an already loaded SentenceTransformer model, for SemanticChunker.
	"""

	def __init__(self, model):
		"""
		Input: model, a loaded SentenceTransformer model.
		Output: nothing.
		What it does: stores the model.
		"""
		self.model = model

	def embed_documents(self, texts: list[str]) -> list[list[float]]:
		"""
		Input: texts, a list of strings.
		Output: one embedding per text.
		What it does: encodes the texts with the model.
		"""
		embeddings = self.model.encode(texts, batch_size=256, show_progress_bar=False)
		embedding_lists = []
		for embedding in embeddings:
			embedding_lists.append(embedding.tolist())
		return embedding_lists

	def embed_query(self, text: str) -> list[float]:
		"""
		Input: text, one string.
		Output: its embedding.
		What it does: encodes the text with embed_documents.
		"""
		embedding_lists = self.embed_documents([text])
		return embedding_lists[0]


def save_chunks(chunk_path: str, doc_id: str, chunks: dict[str, Chunk]) -> None:
	"""
	Input: chunk_path; doc_id; chunks, the document's chunks.
	Output: nothing.
	What it does: saves the document's chunks as one JSON list to {doc_id}.json.
	"""
	os.makedirs(chunk_path, exist_ok=True)

	chunk_dicts = []
	for chunk_id in chunks:
		chunk_dicts.append(dataclasses.asdict(chunks[chunk_id]))

	chunk_file = os.path.join(chunk_path, f"{doc_id}.json")
	with open(chunk_file, "w", encoding="utf-8") as f:
		json.dump(chunk_dicts, f)


def load_chunks(chunk_path: str, doc_id: str) -> dict[str, Chunk] | None:
	"""
	Input: chunk_path; doc_id.
	Output: a dict of chunk_id to Chunk, or None if the document has no saved chunks.
	What it does: loads the document's chunks from {doc_id}.json.
	"""
	chunk_file = os.path.join(chunk_path, f"{doc_id}.json")
	if not os.path.exists(chunk_file):
		return None

	with open(chunk_file, encoding="utf-8") as f:
		chunk_dicts = json.load(f)

	chunks = {}
	for chunk_dict in chunk_dicts:
		chunk = Chunk(**chunk_dict)
		chunks[chunk.id] = chunk

	return chunks


class Chunking:
	def __init__(self, chunking_args: ChunkingArgs):
		"""
		Input: chunking_args, the chunking settings for one chunk type.
		Output: nothing.
		What it does: stores chunking_args. The tokenizer is set by the chunking method that runs.
		"""
		self.chunking_args = chunking_args
		self.tokenizer = None

	def validate(self) -> None:
		"""
		Input: none.
		Output: nothing.
		What it does: raises ValueError if a chunking setting is not valid.
		"""
		chunking_args = self.chunking_args

		if chunking_args.model_name == "":
			raise ValueError("model_name is empty.")

		if chunking_args.chunk_type not in [constants.FIXED_CHUNKING, constants.SENTENCE_CHUNKING, constants.SEMANTIC_CHUNKING]:
			raise ValueError(
				f"chunk_type ({chunking_args.chunk_type}) is not supported. Use "
				f"\"{constants.FIXED_CHUNKING}\", \"{constants.SENTENCE_CHUNKING}\" or \"{constants.SEMANTIC_CHUNKING}\"."
			)

		if chunking_args.min_chunk_size <= 0:
			raise ValueError(f"min_chunk_size ({chunking_args.min_chunk_size}) must be greater than 0.")

		if chunking_args.max_chunk_size < chunking_args.min_chunk_size:
			raise ValueError(
				f"max_chunk_size ({chunking_args.max_chunk_size}) must not be smaller than "
				f"min_chunk_size ({chunking_args.min_chunk_size})."
			)

		if chunking_args.fixed_chunk_overlap < 0 or chunking_args.fixed_chunk_overlap >= chunking_args.max_chunk_size:
			raise ValueError(
				f"fixed_chunk_overlap ({chunking_args.fixed_chunk_overlap}) must be at least 0 and smaller than "
				f"max_chunk_size ({chunking_args.max_chunk_size})."
			)

	def compute_chunks(self, documents: dict[str, Document], output_path: str) -> dict[str, Chunk]:
		"""
		Input: documents, a dict of doc_id to Document; output_path, the dataset's output folder.
		Output: a dict of chunk_id to Chunk for every document.
		What it does: loads saved chunks, chunks the remaining documents with LangChain and saves them.
		"""
		# 1. Validate the settings
		self.validate()

		# 2. Load saved chunks, and collect the documents that still need chunking
		chunk_path = paths_temp.get_chunk_path(output_path, self.chunking_args)
		print(f"Chunk folder : {chunk_path}")
		chunks = {}
		cached_chunk_count = 0
		uncached_documents = {}
		for doc_id in documents:
			cached_chunks = load_chunks(chunk_path, doc_id)
			if cached_chunks is not None:
				chunks.update(cached_chunks)
				cached_chunk_count += len(cached_chunks)
			else:
				uncached_documents[doc_id] = documents[doc_id]

		# 3. Chunk the documents that still need chunking, with the method of the chunk type
		chunks_by_document = {}
		if len(uncached_documents) > 0:
			chunk_type = self.chunking_args.chunk_type
			if chunk_type == constants.FIXED_CHUNKING:
				chunks_by_document = self.fixed_chunking(uncached_documents)
			elif chunk_type == constants.SENTENCE_CHUNKING:
				chunks_by_document = self.sentence_chunking(uncached_documents)
			elif chunk_type == constants.SEMANTIC_CHUNKING:
				chunks_by_document = self.semantic_chunking(uncached_documents)

		# 4. Save the new chunks, one file per document
		new_chunk_count = 0
		for doc_id in chunks_by_document:
			new_chunks = chunks_by_document[doc_id]
			save_chunks(chunk_path, doc_id, new_chunks)
			chunks.update(new_chunks)
			new_chunk_count += len(new_chunks)

		cached_document_count = len(documents) - len(uncached_documents)
		print(f"Documents : {len(documents)} ({cached_document_count} from cache, {len(uncached_documents)} newly chunked)")
		print(f"Chunks : {len(chunks)} ({cached_chunk_count} from cache, {new_chunk_count} new)")

		return chunks

	def fixed_chunking(self, documents: dict[str, Document]) -> dict[str, dict[str, Chunk]]:
		"""
		Input: documents, a dict of doc_id to Document.
		Output: a dict of doc_id to that document's chunks.
		What it does: splits each document into overlapping token windows with SentenceTransformersTokenTextSplitter.
		"""
		# 1. Build the splitter. It loads the model itself
		device = "cuda" if torch.cuda.is_available() else "cpu"
		splitter = SentenceTransformersTokenTextSplitter(
			chunk_overlap=self.chunking_args.fixed_chunk_overlap,
			model_name=self.chunking_args.model_name,
			tokens_per_chunk=self.chunking_args.max_chunk_size,
			model_kwargs={"device": device},
		)
		self.tokenizer = splitter.tokenizer

		# 2. Split each document into token windows
		chunks_by_document = {}
		for doc_id in documents:
			chunk_texts = splitter.split_text(documents[doc_id].text)
			chunks_by_document[doc_id] = self.make_chunks(doc_id, chunk_texts)

		return chunks_by_document

	def sentence_chunking(self, documents: dict[str, Document]) -> dict[str, dict[str, Chunk]]:
		"""
		Input: documents, a dict of doc_id to Document.
		Output: a dict of doc_id to that document's chunks.
		What it does: packs whole sentences into chunks with NLTKTextSplitter, then cuts too-long pieces.
		"""
		# 1. Load the model for its tokenizer
		device = "cuda" if torch.cuda.is_available() else "cpu"
		model = sentence_transformers.SentenceTransformer(self.chunking_args.model_name, device=device)
		self.tokenizer = model.tokenizer

		# 2. Build the splitter. Its warning for each too-long sentence is turned off; those are cut in step 3
		logging.getLogger("langchain_text_splitters.base").setLevel(logging.ERROR)
		splitter = NLTKTextSplitter(
			separator=" ",
			chunk_size=self.chunking_args.max_chunk_size,
			chunk_overlap=0,
			length_function=self.count_tokens,
		)

		# 3. Split each document into sentence chunks, then cut the too-long ones
		chunks_by_document = {}
		for doc_id in documents:
			piece_texts = splitter.split_text(documents[doc_id].text)
			chunk_texts = self.cut_long_pieces(piece_texts)
			chunks_by_document[doc_id] = self.make_chunks(doc_id, chunk_texts)

		return chunks_by_document

	def semantic_chunking(self, documents: dict[str, Document]) -> dict[str, dict[str, Chunk]]:
		"""
		Input: documents, a dict of doc_id to Document.
		Output: a dict of doc_id to that document's chunks.
		What it does: splits each document at topic changes with SemanticChunker, then cuts too-long pieces.
		"""
		# 1. Load the model, used both to embed sentences and for its tokenizer
		device = "cuda" if torch.cuda.is_available() else "cpu"
		model = sentence_transformers.SentenceTransformer(self.chunking_args.model_name, device=device)
		self.tokenizer = model.tokenizer

		# 2. Build the splitter
		splitter = SemanticChunker(
			embeddings=SentenceTransformerEmbeddings(model),
			buffer_size=0,
			breakpoint_threshold_type="percentile",
		)

		# 3. Split each document at topic changes, then cut the too-long pieces
		chunks_by_document = {}
		for doc_id in documents:
			piece_texts = splitter.split_text(documents[doc_id].text)
			chunk_texts = self.cut_long_pieces(piece_texts)
			chunks_by_document[doc_id] = self.make_chunks(doc_id, chunk_texts)

		return chunks_by_document

	def make_chunks(self, doc_id: str, chunk_texts: list[str]) -> dict[str, Chunk]:
		"""
		Input: doc_id; chunk_texts, the document's chunk texts in order.
		Output: a dict of chunk_id to Chunk.
		What it does: wraps each text into a Chunk with id "{doc_id}_{n}".
		"""
		chunks = {}
		for chunk_index in range(len(chunk_texts)):
			chunk_text = chunk_texts[chunk_index]
			chunk_id = f"{doc_id}_{chunk_index}"
			chunk_size = self.count_tokens(chunk_text)
			chunks[chunk_id] = Chunk(chunk_id, chunk_text, chunk_size, self.chunking_args.chunk_type)
		return chunks

	def cut_long_pieces(self, piece_texts: list[str]) -> list[str]:
		"""
		Input: piece_texts, one document's pieces from a LangChain splitter.
		Output: the chunk texts, in order.
		What it does: drops empty pieces and cuts pieces longer than max_chunk_size into token windows.
		"""
		# The windows hold no start and end special tokens
		long_piece_tokenizer = Tokenizer(
			chunk_overlap=self.chunking_args.fixed_chunk_overlap,
			tokens_per_chunk=self.chunking_args.max_chunk_size,
			decode=self.tokenizer.decode,
			encode=functools.partial(self.tokenizer.encode, add_special_tokens=False),
		)

		chunk_texts = []
		for piece_text in piece_texts:
			if piece_text.strip() == "":
				continue
			if self.count_tokens(piece_text) <= self.chunking_args.max_chunk_size:
				chunk_texts.append(piece_text)
			else:
				window_texts = split_text_on_tokens(text=piece_text, tokenizer=long_piece_tokenizer)
				for window_text in window_texts:
					chunk_texts.append(window_text)
		return chunk_texts

	# One line, but kept as a method because NLTKTextSplitter needs a function as its length_function
	def count_tokens(self, text: str) -> int:
		"""
		Input: text.
		Output: the number of tokens in text.
		What it does: counts the tokens without the start and end special tokens.
		"""
		return len(self.tokenizer.encode(text, add_special_tokens=False))
