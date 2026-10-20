from nltk.tokenize import sent_tokenize

from common.constants import SENTENCE_CHUNKING
from stages.build.chunking.chunker import Chunker
from stages.build.chunking.schema import Chunk

class SentenceChunking(Chunker):
	chunk_type = SENTENCE_CHUNKING

	def chunk(self, document) -> dict[str, Chunk]:
		"""
		Input: document, the Document to split.
		Output: a dict of chunk_id to Chunk for this document.
		What it does: splits the text into sentences and packs whole
		sentences into a chunk until the next sentence would exceed
		max_chunk_size tokens, then starts a new chunk. A single sentence
		longer than max_chunk_size is split with the fixed-window fallback.
		"""

		chunks = {}

		max_chunk_size = self.max_chunk_size

		doc_id = document.id
		text = document.text

		sentences = sent_tokenize(text)

		current_chunk_texts = []
		current_chunk_tokens = []

		chunk_idx = 0

		for i in range(len(sentences)):
			per_sentence = sentences[i]
			tokens_per_sentence = self.tokenizer.encode(per_sentence, add_special_tokens=False)

			# 1. a sentence longer than max_chunk_size on its own: flush what we have,
			# then split just this sentence with the fixed-window fallback.
			if len(tokens_per_sentence) > max_chunk_size:
				if current_chunk_texts:
					chunk = self._finalize_chunk(doc_id, chunk_idx, ' '.join(current_chunk_texts), current_chunk_tokens)
					chunks[chunk.id] = chunk

					chunk_idx += 1
					current_chunk_texts = []
					current_chunk_tokens = []

				sub_chunks, chunk_idx = self._split_tokens_into_chunks(doc_id, tokens_per_sentence, chunk_idx)
				chunks.update(sub_chunks)

				# Move to the next sentence after splitting the long sentence into chunks
				continue

			# 2. adding this sentence would exceed the token limit
			over_limit = len(current_chunk_tokens) + len(tokens_per_sentence) > max_chunk_size
			if current_chunk_texts and over_limit:
				chunk = self._finalize_chunk(doc_id, chunk_idx, ' '.join(current_chunk_texts), current_chunk_tokens)
				chunks[chunk.id] = chunk

				chunk_idx += 1
				current_chunk_texts = []
				current_chunk_tokens = []

			# 3. proceed normally: append the sentence to the current chunk
			current_chunk_texts.append(per_sentence)
			current_chunk_tokens.extend(tokens_per_sentence)

		# Outside loop
		# Add the last chunk if it has content
		if current_chunk_texts:
			chunk = self._finalize_chunk(doc_id, chunk_idx, ' '.join(current_chunk_texts), current_chunk_tokens)
			chunks[chunk.id] = chunk

		return chunks
