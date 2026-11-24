import functools
import logging

import torch
import sentence_transformers

from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import NLTKTextSplitter
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_text_splitters import Tokenizer
from langchain_text_splitters import split_text_on_tokens

from common.args import ChunkingArgs
from common.args import EmbedderArgs
from schema.file import File
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


class Chunker:
	def __init__(self, embedder_args: EmbedderArgs, chunking_args: ChunkingArgs):
		"""
		Input: embedder_args, the shared embedder settings, whose model_name the chunkers use; chunking_args, the
		chunking settings for one chunk type.
		Output: nothing.
		What it does: stores embedder_args and chunking_args. The tokenizer is set by the chunking method that runs.
		"""
		self.embedder_args = embedder_args
		self.chunking_args = chunking_args
		self.tokenizer = None

	def fixed_chunking(self, documents: dict[str, File]) -> dict[str, dict[str, Chunk]]:
		"""
		Input: documents, a dict of doc_id to File.
		Output: a dict of doc_id to that document's chunks.
		What it does: splits each document into overlapping token windows with SentenceTransformersTokenTextSplitter,
		then merges too-short windows.
		"""
		# 1. Build the splitter. It loads the model itself
		device = "cuda" if torch.cuda.is_available() else "cpu"
		splitter = SentenceTransformersTokenTextSplitter(
			chunk_overlap=self.chunking_args.fixed_chunk_overlap,
			model_name=self.embedder_args.model_name,
			tokens_per_chunk=self.chunking_args.max_chunk_size,
			model_kwargs={"device": device},
		)
		self.tokenizer = splitter.tokenizer

		# 2. Split each document into token windows, then merge the too-short ones
		chunks_by_document = {}
		for doc_id in documents:
			window_texts = splitter.split_text(documents[doc_id].text)
			chunk_texts = self.merge_short_pieces(window_texts)
			chunks_by_document[doc_id] = self.make_chunks(doc_id, chunk_texts)

		return chunks_by_document

	def sentence_chunking(self, documents: dict[str, File]) -> dict[str, dict[str, Chunk]]:
		"""
		Input: documents, a dict of doc_id to File.
		Output: a dict of doc_id to that document's chunks.
		What it does: packs whole sentences into chunks with NLTKTextSplitter, then cuts too-long pieces and merges
		too-short ones.
		"""
		# 1. Load the model for its tokenizer
		device = "cuda" if torch.cuda.is_available() else "cpu"
		model = sentence_transformers.SentenceTransformer(self.embedder_args.model_name, device=device)
		self.tokenizer = model.tokenizer

		# 2. Build the splitter. Its warning for each too-long sentence is turned off; those are cut in step 3
		logging.getLogger("langchain_text_splitters.base").setLevel(logging.ERROR)
		splitter = NLTKTextSplitter(
			separator=" ",
			chunk_size=self.chunking_args.max_chunk_size,
			chunk_overlap=0,
			length_function=self.count_tokens,
		)

		# 3. Split each document into sentence chunks, then cut the too-long ones and merge the too-short ones
		chunks_by_document = {}
		for doc_id in documents:
			piece_texts = splitter.split_text(documents[doc_id].text)
			cut_texts = self.cut_long_pieces(piece_texts)
			chunk_texts = self.merge_short_pieces(cut_texts)
			chunks_by_document[doc_id] = self.make_chunks(doc_id, chunk_texts)

		return chunks_by_document

	def semantic_chunking(self, documents: dict[str, File]) -> dict[str, dict[str, Chunk]]:
		"""
		Input: documents, a dict of doc_id to File.
		Output: a dict of doc_id to that document's chunks.
		What it does: splits each document at topic changes with SemanticChunker, then cuts too-long pieces and merges
		too-short ones.
		"""
		# 1. Load the model, used both to embed sentences and for its tokenizer
		device = "cuda" if torch.cuda.is_available() else "cpu"
		model = sentence_transformers.SentenceTransformer(self.embedder_args.model_name, device=device)
		self.tokenizer = model.tokenizer

		# 2. Build the splitter
		splitter = SemanticChunker(
			embeddings=SentenceTransformerEmbeddings(model),
			buffer_size=0,
			breakpoint_threshold_type="percentile",
		)

		# 3. Split each document at topic changes, then cut the too-long pieces and merge the too-short ones
		chunks_by_document = {}
		for doc_id in documents:
			piece_texts = splitter.split_text(documents[doc_id].text)
			cut_texts = self.cut_long_pieces(piece_texts)
			chunk_texts = self.merge_short_pieces(cut_texts)
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

	def merge_short_pieces(self, piece_texts: list[str]) -> list[str]:
		"""
		Input: piece_texts, one document's chunk texts in order.
		Output: the chunk texts after merging, in order.
		What it does: joins a piece shorter than min_chunk_size tokens with the next piece, with a space, as long as the
		joined text stays at most max_chunk_size tokens. If the last piece is still shorter than min_chunk_size, it is
		joined to the previous piece under the same limit. A short piece that cannot be joined without going over
		max_chunk_size is kept as it is.
		"""
		min_chunk_size = self.chunking_args.min_chunk_size
		max_chunk_size = self.chunking_args.max_chunk_size

		# 1. Walk the pieces in order, joining the current piece with the next while it is too short
		merged_texts = []
		current_text = None
		for piece_text in piece_texts:
			if current_text is None:
				current_text = piece_text
				continue

			if self.count_tokens(current_text) < min_chunk_size:
				joined_text = current_text + " " + piece_text
				if self.count_tokens(joined_text) <= max_chunk_size:
					current_text = joined_text
					continue

			merged_texts.append(current_text)
			current_text = piece_text

		if current_text is None:
			return merged_texts

		# 2. The last piece: join it to the previous one if it is still too short and the join fits
		last_text_is_short = self.count_tokens(current_text) < min_chunk_size
		if last_text_is_short and len(merged_texts) > 0:
			joined_text = merged_texts[-1] + " " + current_text
			if self.count_tokens(joined_text) <= max_chunk_size:
				merged_texts[-1] = joined_text
				return merged_texts

		merged_texts.append(current_text)
		return merged_texts

	# One line, but kept as a method because NLTKTextSplitter needs a function as its length_function
	def count_tokens(self, text: str) -> int:
		"""
		Input: text.
		Output: the number of tokens in text.
		What it does: counts the tokens without the start and end special tokens.
		"""
		return len(self.tokenizer.encode(text, add_special_tokens=False))
