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

from common import constants
from common import paths
from common.args import ChunkingArgs

from stages.build.ingestion.schema import Document
from stages.build.chunking.schema import Chunk
from stages.build.chunking import cache


class SentenceTransformerEmbeddings(Embeddings):
	"""
	A LangChain Embeddings object around an already-loaded
	SentenceTransformer model, so LangChain's SemanticChunker embeds
	sentences with the same model used for counting tokens, without
	loading a second copy.
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
		Output: one embedding (a list of floats) per text.
		What it does: encodes all texts with the model in batches of 256.
		"""
		embeddings = self.model.encode(texts, batch_size=256, show_progress_bar=False)
		embedding_lists = []
		for embedding in embeddings:
			embedding_lists.append(embedding.tolist())
		return embedding_lists

	def embed_query(self, text: str) -> list[float]:
		"""
		Input: text, one string.
		Output: its embedding as a list of floats.
		What it does: encodes the single text with embed_documents.
		"""
		embedding_lists = self.embed_documents([text])
		return embedding_lists[0]


class Langchain:
	def __init__(self, chunking_args: ChunkingArgs):
		"""
		Input: chunking_args, a ChunkingArgs with framework "langchain".
		Output: nothing.
		What it does: stores chunking_args. The tokenizer is set by
		fixed_chunking, sentence_chunking or semantic_chunking, whichever
		runs, and is used by the helper methods to count and encode tokens.
		"""
		self.chunking_args = chunking_args
		self.tokenizer = None

	def compute_chunks(self, documents: dict[str, Document], output_path:str)-> dict[str, Chunk]:
		"""
		Input: documents, a dict of doc_id to Document from stages.build.ingestion;
		output_path, the dataset's output folder the chunk folder is built under.
		Output: a dict of chunk_id to Chunk for every document.
		What it does: loads already saved chunks for each document from the
		LangChain chunk folder, chunks the remaining documents with
		fixed_chunking, sentence_chunking or semantic_chunking (by
		chunk_type), and saves their new chunks. Prints how many documents
		and chunks were loaded from cache and how many were newly computed.
		"""
		chunk_path = paths.get_chunk_path(output_path, self.chunking_args)
		chunks = {}
		cached_chunk_count = 0
		new_chunk_count = 0

		# 1. Load cached chunks, and collect the documents that still need chunking
		uncached_documents = {}
		for doc_id in documents:
			cached_chunks = cache._load_cached_chunks(chunk_path, doc_id)
			if cached_chunks is not None:
				chunks.update(cached_chunks)
				cached_chunk_count += len(cached_chunks)
			else:
				uncached_documents[doc_id] = documents[doc_id]

		# 2. Chunk the documents that still need chunking
		chunks_by_document = {}
		if len(uncached_documents) > 0:
			chunk_type = self.chunking_args.chunk_type
			if chunk_type == constants.FIXED_CHUNKING:
				chunks_by_document = self.fixed_chunking(uncached_documents)
			elif chunk_type == constants.SENTENCE_CHUNKING:
				chunks_by_document = self.sentence_chunking(uncached_documents)
			elif chunk_type == constants.SEMANTIC_CHUNKING:
				chunks_by_document = self.semantic_chunking(uncached_documents)
			else:
				raise NotImplementedError(f"LangChain chunking for chunk_type ({chunk_type}) is not implemented.")

		# 3. Save the new chunks, one file per document
		for doc_id in chunks_by_document:
			new_chunks = chunks_by_document[doc_id]
			cache._save_chunks(chunk_path, doc_id, new_chunks)
			chunks.update(new_chunks)
			new_chunk_count += len(new_chunks)

		cached_document_count = len(documents) - len(uncached_documents)
		print(f"Chunk folder : {chunk_path}")
		print(f"Documents : {len(documents)} ({cached_document_count} from cache, {len(uncached_documents)} newly chunked)")
		print(f"Chunks : {len(chunks)} ({cached_chunk_count} from cache, {new_chunk_count} new)")

		return chunks

	def fixed_chunking(self, documents: dict[str, Document]) -> dict[str, dict[str, Chunk]]:
		"""
		Input: documents, a dict of doc_id to Document to chunk.
		Output: a dict of doc_id to that document's chunks (chunk_id to Chunk).
		What it does: chunks each document with LangChain's
		SentenceTransformersTokenTextSplitter, which tokenizes the text with
		the model named by model_name and cuts it into windows of
		max_chunk_size tokens that overlap by fixed_chunk_overlap tokens,
		stopping once a window reaches the last token. Same rule as our own
		fixed chunking. LangChain raises ValueError if max_chunk_size is
		larger than the model's maximum sequence length.
		"""
		# 1. Build the splitter. It loads the model itself.
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
		Input: documents, a dict of doc_id to Document to chunk.
		Output: a dict of doc_id to that document's chunks (chunk_id to Chunk).
		What it does: chunks each document with LangChain's
		NLTKTextSplitter, which cuts the text into sentences with NLTK's
		sent_tokenize and packs whole sentences, joined by a space, into
		chunks of up to max_chunk_size tokens, without overlap. Same rule as
		our own sentence chunking. LangChain keeps a sentence longer than
		max_chunk_size whole, so those are cut into overlapping token
		windows afterwards, like our own sentence chunking does.
		"""
		# 1. Load the model for its tokenizer
		device = "cuda" if torch.cuda.is_available() else "cpu"
		model = sentence_transformers.SentenceTransformer(self.chunking_args.model_name, device=device)
		self.tokenizer = model.tokenizer

		# 2. Build the splitter. LangChain logs a warning for every sentence
		# longer than chunk_size; those are cut in step 3, so the warning is
		# turned off to keep the run output readable.
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
		Input: documents, a dict of doc_id to Document to chunk.
		Output: a dict of doc_id to that document's chunks (chunk_id to Chunk).
		What it does: chunks each document with LangChain's SemanticChunker
		using its own defaults except buffer_size: it splits the text into
		sentences after ".", "?" or "!", embeds each sentence on its own
		(buffer_size 0, like our own semantic chunking), and starts a new
		chunk wherever the cosine distance between neighbouring sentences is
		above the 95th percentile of that document's distances. It has no
		size limit, so pieces longer than max_chunk_size are cut into
		overlapping token windows afterwards. semantic_threshold and
		min_chunk_size are not used.
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

	# ---------- helpers ----------

	def make_chunks(self, doc_id: str, chunk_texts: list[str]) -> dict[str, Chunk]:
		"""
		Input: doc_id, the document the texts belong to; chunk_texts, its
		chunk texts in order.
		Output: a dict of chunk_id to Chunk.
		What it does: wraps each text into a Chunk with id "{doc_id}_{n}",
		its token count as size and chunk_type as type.
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
		What it does: drops empty pieces (SemanticChunker returns one empty
		piece for an empty text), keeps pieces of up to max_chunk_size
		tokens as they are, and cuts longer pieces with LangChain's
		split_text_on_tokens into windows of max_chunk_size tokens that
		overlap by fixed_chunk_overlap tokens.
		"""
		# encode is the tokenizer's encode with add_special_tokens=False
		# filled in, so the windows hold no start and end special tokens
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

	# One line, but kept as a method because NLTKTextSplitter needs a
	# function to call as its length_function.
	def count_tokens(self, text: str) -> int:
		"""
		Input: text.
		Output: the number of tokens in text.
		What it does: counts the tokens of text without the start and end
		special tokens, the same way our own chunkers count chunk size.
		"""
		return len(self.tokenizer.encode(text, add_special_tokens=False))
