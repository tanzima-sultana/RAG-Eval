import bisect

import numpy as np
import torch
import sentence_transformers

from nltk.tokenize.punkt import PunktTokenizer

from common import constants
from common.args import ChunkingArgs
from common.args import EmbedderArgs
from schema.chunk import Chunk
from schema.file import File


class Chunker:
	def __init__(self, embedder_args: EmbedderArgs, chunking_args: ChunkingArgs):
		"""
		Input: embedder_args, the shared embedder settings, whose model_name the chunkers use; chunking_args, the
		chunking settings for one chunk type.
		Output: nothing.
		What it does: stores embedder_args and chunking_args. The model and its tokenizer are loaded by the chunking
		method that runs.
		"""
		self.embedder_args = embedder_args
		self.chunking_args = chunking_args
		self.model = None
		self.tokenizer = None

	def load_model(self) -> None:
		"""
		Input: none.
		Output: nothing.
		What it does: loads the SentenceTransformer model of the embedder, on cuda if it is available, otherwise on cpu,
		and keeps its tokenizer. The tokenizer counts tokens and gives the position of every token in the text. The
		model embeds sentences for semantic chunking.
		"""
		device = "cuda" if torch.cuda.is_available() else "cpu"
		self.model = sentence_transformers.SentenceTransformer(self.embedder_args.model_name, device=device)
		self.tokenizer = self.model.tokenizer

	def fixed_chunking(self, files: list[File]) -> dict[str, Chunk]:
		"""
		Input: files, a list of File.
		Output: a dict of chunk_id to Chunk for every file.
		What it does: cuts each file's document_texts into token windows of max_chunk_size tokens, each overlapping the
		previous one by fixed_chunk_overlap tokens, then merges too-short windows.
		"""
		self.load_model()

		chunks = {}
		for file in files:
			token_starts, token_ends = self.tokenize_document(file.document_texts)
			spans = self.window_spans(token_starts, token_ends, 0, len(token_starts))
			spans = self.merge_short_spans(spans, token_starts)
			chunks.update(self.make_chunks(file, spans, token_starts))

		return chunks

	def sentence_chunking(self, files: list[File]) -> dict[str, Chunk]:
		"""
		Input: files, a list of File.
		Output: a dict of chunk_id to Chunk for every file.
		What it does: splits each file's document_texts into sentences with NLTK's Punkt splitter and packs whole
		sentences into a chunk until the next sentence would go over max_chunk_size tokens. A sentence longer than
		max_chunk_size is cut into overlapping token windows. Too-short chunks are merged at the end.
		"""
		self.load_model()
		max_chunk_size = self.chunking_args.max_chunk_size
		sentence_splitter = PunktTokenizer("english")

		chunks = {}
		for file in files:
			token_starts, token_ends = self.tokenize_document(file.document_texts)
			sentence_spans = list(sentence_splitter.span_tokenize(file.document_texts))

			spans = []
			current_start = None
			current_end = None
			for sentence_start, sentence_end in sentence_spans:
				sentence_tokens = self.count_tokens(token_starts, sentence_start, sentence_end)

				# 1. A sentence longer than max_chunk_size on its own: close the chunk being built, then cut the sentence
				if sentence_tokens > max_chunk_size:
					if current_start is not None:
						spans.append((current_start, current_end))
						current_start = None
						current_end = None

					first_token = bisect.bisect_left(token_starts, sentence_start)
					last_token = bisect.bisect_left(token_starts, sentence_end)
					long_sentence_windows = self.window_spans(token_starts, token_ends, first_token, last_token)
					for long_sentence_window in long_sentence_windows:
						spans.append(long_sentence_window)
					continue

				# 2. Adding this sentence would go over max_chunk_size: close the chunk being built
				if current_start is not None:
					tokens_with_sentence = self.count_tokens(token_starts, current_start, sentence_end)
					if tokens_with_sentence > max_chunk_size:
						spans.append((current_start, current_end))
						current_start = None
						current_end = None

				# 3. Add the sentence to the chunk being built
				if current_start is None:
					current_start = sentence_start
				current_end = sentence_end

			if current_start is not None:
				spans.append((current_start, current_end))

			spans = self.merge_short_spans(spans, token_starts)
			chunks.update(self.make_chunks(file, spans, token_starts))

		return chunks

	def semantic_chunking(self, files: list[File]) -> dict[str, Chunk]:
		"""
		Input: files, a list of File.
		Output: a dict of chunk_id to Chunk for every file.
		What it does: splits each file's document_texts into sentences with NLTK's Punkt splitter, embeds every
		sentence and starts a new chunk where the cosine distance between a sentence and the one before it is above the
		SEMANTIC_BREAKPOINT_PERCENTILE percentile of that file's distances. A chunk longer than max_chunk_size is cut
		into overlapping token windows. Too-short chunks are merged at the end.
		"""
		self.load_model()
		max_chunk_size = self.chunking_args.max_chunk_size
		sentence_splitter = PunktTokenizer("english")

		chunks = {}
		for file in files:
			token_starts, token_ends = self.tokenize_document(file.document_texts)
			sentence_spans = list(sentence_splitter.span_tokenize(file.document_texts))
			if len(sentence_spans) == 0:
				continue

			# 1. Embed every sentence, and find the sentences after which the meaning changes the most
			sentence_texts = []
			for sentence_start, sentence_end in sentence_spans:
				sentence_texts.append(file.document_texts[sentence_start:sentence_end])
			embeddings = self.model.encode(sentence_texts, batch_size=256, show_progress_bar=False)
			norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
			norms[norms == 0] = 1
			normalized_embeddings = embeddings / norms

			break_after_sentences = set()
			if len(sentence_spans) > 1:
				similarities = np.sum(normalized_embeddings[:-1] * normalized_embeddings[1:], axis=1)
				distances = 1 - similarities
				breakpoint_distance = np.percentile(distances, constants.SEMANTIC_BREAKPOINT_PERCENTILE)
				for sentence_index in range(len(distances)):
					if distances[sentence_index] > breakpoint_distance:
						break_after_sentences.add(sentence_index)

			# 2. The pieces between the breakpoints
			piece_spans = []
			piece_start = None
			for sentence_index in range(len(sentence_spans)):
				sentence_start, sentence_end = sentence_spans[sentence_index]
				if piece_start is None:
					piece_start = sentence_start
				is_last_sentence = sentence_index == len(sentence_spans) - 1
				if sentence_index in break_after_sentences or is_last_sentence:
					piece_spans.append((piece_start, sentence_end))
					piece_start = None

			# 3. Cut the pieces longer than max_chunk_size into overlapping token windows, then merge the short ones
			spans = []
			for piece_start, piece_end in piece_spans:
				piece_tokens = self.count_tokens(token_starts, piece_start, piece_end)
				if piece_tokens <= max_chunk_size:
					spans.append((piece_start, piece_end))
					continue

				first_token = bisect.bisect_left(token_starts, piece_start)
				last_token = bisect.bisect_left(token_starts, piece_end)
				piece_windows = self.window_spans(token_starts, token_ends, first_token, last_token)
				for piece_window in piece_windows:
					spans.append(piece_window)

			spans = self.merge_short_spans(spans, token_starts)
			chunks.update(self.make_chunks(file, spans, token_starts))

		return chunks

	def tokenize_document(self, document_texts: str) -> tuple[list[int], list[int]]:
		"""
		Input: document_texts, the text of one document.
		Output: (token_starts, token_ends): the start and the end position in document_texts of every token, in order.
		What it does: tokenizes the whole text once, without the start and end special tokens, and reads the position of
		each token from the fast tokenizer's offset mapping. The positions are exact, so a stretch of the text can be
		counted and cut by tokens with no decode.
		"""
		encoded = self.tokenizer(document_texts, add_special_tokens=False, return_offsets_mapping=True)

		token_starts = []
		token_ends = []
		for token_start, token_end in encoded["offset_mapping"]:
			token_starts.append(token_start)
			token_ends.append(token_end)
		return token_starts, token_ends

	# One line, but kept as a method because every chunker and the merge step count tokens of a span
	def count_tokens(self, token_starts: list[int], start: int, end: int) -> int:
		"""
		Input: token_starts, the start position of every token of the document; start and end, a span of the text.
		Output: the number of tokens that start inside the span.
		What it does: counts the tokens of the span with two binary searches.
		"""
		return bisect.bisect_left(token_starts, end) - bisect.bisect_left(token_starts, start)

	def window_spans(self, token_starts: list[int], token_ends: list[int], first_token: int, last_token: int) -> list[tuple[int, int]]:
		"""
		Input: token_starts and token_ends, the positions of every token of the document; first_token and last_token,
		the range of tokens to cut, last_token not included.
		Output: a list of (start, end) spans in the text, one per window.
		What it does: slides a window of max_chunk_size tokens over the range, advancing by
		max_chunk_size - fixed_chunk_overlap tokens each step. It stops as soon as a window reaches the last token, so no
		extra window holds only tokens already covered by the overlap.
		"""
		max_chunk_size = self.chunking_args.max_chunk_size
		step = max_chunk_size - self.chunking_args.fixed_chunk_overlap

		spans = []
		window_first_token = first_token
		while window_first_token < last_token:
			window_end_token = window_first_token + max_chunk_size
			if window_end_token > last_token:
				window_end_token = last_token

			spans.append((token_starts[window_first_token], token_ends[window_end_token - 1]))

			if window_end_token >= last_token:
				break
			window_first_token += step

		return spans

	def merge_short_spans(self, spans: list[tuple[int, int]], token_starts: list[int]) -> list[tuple[int, int]]:
		"""
		Input: spans, one document's chunk spans in order; token_starts, the start position of every token.
		Output: the spans after merging, in order.
		What it does: joins a span shorter than min_chunk_size tokens with the next span, as long as the joined span
		stays at most max_chunk_size tokens. If the last span is still shorter than min_chunk_size, it is joined to the
		previous span under the same limit. A short span that cannot be joined without going over max_chunk_size is kept
		as it is. A joined span runs from the start of the first to the end of the last, so it stays an exact slice.
		"""
		min_chunk_size = self.chunking_args.min_chunk_size
		max_chunk_size = self.chunking_args.max_chunk_size

		# 1. Walk the spans in order, joining the current span with the next while it is too short
		merged_spans = []
		current_span = None
		for span in spans:
			if current_span is None:
				current_span = span
				continue

			current_tokens = self.count_tokens(token_starts, current_span[0], current_span[1])
			if current_tokens < min_chunk_size:
				joined_tokens = self.count_tokens(token_starts, current_span[0], span[1])
				if joined_tokens <= max_chunk_size:
					current_span = (current_span[0], span[1])
					continue

			merged_spans.append(current_span)
			current_span = span

		if current_span is None:
			return merged_spans

		# 2. The last span: join it to the previous one if it is still too short and the join fits
		last_tokens = self.count_tokens(token_starts, current_span[0], current_span[1])
		if last_tokens < min_chunk_size and len(merged_spans) > 0:
			previous_span = merged_spans[-1]
			joined_tokens = self.count_tokens(token_starts, previous_span[0], current_span[1])
			if joined_tokens <= max_chunk_size:
				merged_spans[-1] = (previous_span[0], current_span[1])
				return merged_spans

		merged_spans.append(current_span)
		return merged_spans

	def make_chunks(self, file: File, spans: list[tuple[int, int]], token_starts: list[int]) -> dict[str, Chunk]:
		"""
		Input: file, the File the spans are from; spans, its chunk spans in order; token_starts, the start position of
		every token of its document_texts.
		Output: a dict of chunk_id to Chunk.
		What it does: turns each span into a Chunk with id {document_id}_{n}. Its text is the exact slice of
		document_texts, and its page numbers are the pages whose range the span overlaps, so a chunk across a page break
		has both. A blank slice is dropped.
		"""
		# 1. Where each page sits in document_texts: the pages' texts are found one after the other, in page order
		page_ranges = []
		search_start = 0
		for page_number in sorted(file.pages):
			page_text = file.pages[page_number].page_texts
			page_start = file.document_texts.find(page_text, search_start)
			page_end = page_start + len(page_text)
			page_ranges.append((page_number, page_start, page_end))
			search_start = page_end

		# 2. One Chunk per span
		chunks = {}
		chunk_index = 0
		for span_start, span_end in spans:
			chunk_texts = file.document_texts[span_start:span_end]
			if chunk_texts.strip() == "":
				continue

			page_numbers = []
			for page_number, page_start, page_end in page_ranges:
				if page_start < span_end and page_end > span_start:
					page_numbers.append(page_number)

			chunk_id = f"{file.document_id}_{chunk_index}"
			chunks[chunk_id] = Chunk(
				chunk_id=chunk_id,
				chunk_texts=chunk_texts,
				chunk_size=self.count_tokens(token_starts, span_start, span_end),
				chunk_type=self.chunking_args.chunk_type,
				source=file.source,
				page_numbers=page_numbers,
			)
			chunk_index += 1

		return chunks
