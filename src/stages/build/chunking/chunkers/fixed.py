from common.constants import FIXED_CHUNKING
from stages.build.chunking.chunker import Chunker
from stages.build.chunking.schema import Chunk

class FixedChunking(Chunker):
	chunk_type = FIXED_CHUNKING

	def chunk(self, document) -> dict[str, Chunk]:
		"""
		Input: document, the Document to split.
		Output: a dict of chunk_id to Chunk for this document.
		What it does: tokenizes the whole document text and splits the
		tokens into max_chunk_size windows that overlap by
		fixed_chunk_overlap tokens.
		"""
		tokens = self.tokenizer.encode(document.text, add_special_tokens=False)
		chunks, _ = self._split_tokens_into_chunks(document.id, tokens)
		return chunks
