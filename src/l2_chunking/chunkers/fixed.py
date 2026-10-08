from common.constants import FIXED_CHUNKING
from l2_chunking.chunker import Chunker
from l2_chunking.schema import Chunk

class FixedChunking(Chunker):
	chunk_type = FIXED_CHUNKING

	def chunk(self, document) -> dict[str, Chunk]:
		tokens = self.tokenizer.encode(document.text, add_special_tokens=False)
		chunks, _ = self._split_tokens_into_chunks(document.id, tokens)
		return chunks
