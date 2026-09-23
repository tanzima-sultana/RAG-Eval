import numpy as np
from nltk.tokenize import sent_tokenize

from common.constants import SEMANTIC_CHUNKING
from l1_doc_processing.schema import Document
from l2_chunking.chunker import Chunker
from l2_chunking.schema import Chunk

class SemanticChunking(Chunker):
	chunk_type = SEMANTIC_CHUNKING

	def _is_dissimilar(self, sentence_embeddings, prev_idx, curr_idx):
		embedding_1 = sentence_embeddings[prev_idx]
		embedding_2 = sentence_embeddings[curr_idx]
		denom = np.linalg.norm(embedding_1) * np.linalg.norm(embedding_2)
		if denom == 0:
			return False
		cosine_similarity = np.dot(embedding_1, embedding_2) / denom
		return cosine_similarity < self.semantic_threshold

	def merge_chunks(self, chunks) -> dict[str, Chunk]:
		merged_chunks = {}

		prev_chunk = None
		for i, chunk_id in enumerate(chunks):
			if i == 0:
				prev_chunk = chunks[chunk_id]
				merged_chunks[prev_chunk.id] = prev_chunk
				continue 

			cur_chunk = chunks[chunk_id]

			if cur_chunk.size < self.min_chunk_size and prev_chunk.size + cur_chunk.size <= self.max_chunk_size:
				chunk = Chunk(prev_chunk.id, prev_chunk.text + " " + cur_chunk.text, prev_chunk.size + cur_chunk.size, self.chunk_type)
			else:
				chunk = cur_chunk

			merged_chunks[chunk.id] = chunk
			prev_chunk = chunk

		return merged_chunks

	def chunk(self, document: Document) -> dict[str, Chunk]:

		chunks = {}

		max_chunk_size = self.max_chunk_size

		doc_id = document.id
		text = document.text

		start, end = self.embeddings_offset[doc_id]
		doc_embeddings = self.all_embeddings[start:end]

		sentences = sent_tokenize(text)

		current_chunk_texts = []
		current_chunk_tokens = []

		chunk_idx = 0
		prev_idx = None

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
				prev_idx = None
				continue

			# 2.1 adding this sentence would exceed the token limit
			over_limit = len(current_chunk_tokens) + len(tokens_per_sentence) > max_chunk_size

			# 2.2 it's too dissimilar from the previous sentence: finalize the current chunk.
			if prev_idx is not None:
				extra_split = self._is_dissimilar(doc_embeddings, prev_idx, i)
			else:
				extra_split = False

			if current_chunk_texts and (over_limit or extra_split):
				chunk = self._finalize_chunk(doc_id, chunk_idx, ' '.join(current_chunk_texts), current_chunk_tokens)
				chunks[chunk.id] = chunk

				chunk_idx += 1
				current_chunk_texts = []
				current_chunk_tokens = []

			# 3. proceed normally: append the sentence to the current chunk
			current_chunk_texts.append(per_sentence) # add one sentence at a time
			current_chunk_tokens.extend(tokens_per_sentence) # merge tokens into already existing one
			prev_idx = i

		# Outside loop
		# Add the last chunk if it has content
		if current_chunk_texts:
			chunk = self._finalize_chunk(doc_id, chunk_idx, ' '.join(current_chunk_texts), current_chunk_tokens)
			chunks[chunk.id] = chunk

		return self.merge_chunks(chunks)
