import abc

from l1_doc_processing.schema import Document
from l2_chunking.schema import Chunk
from l2_chunking.chunking import Chunking

class Chunker(Chunking, abc.ABC):
	# The chunk_type (e.g. "fixed") this strategy is registered under
	# in get_chunker's chunker_registry. Every concrete subclass must
	# set this to the same string it is registered with, so the
	# registry key and the strategy's own identity never diverge.
	chunk_type: str

	def __init__(self, chunking: Chunking):
		"""
		chunking: the already-constructed Chunking orchestrator
			instance running this pipeline, with its config already
			set and its SentenceTransformer model already loaded.

		Copies chunking's config and loaded model/tokenizer onto this
		strategy instance, instead of calling Chunking.__init__ (which
		would reload the SentenceTransformer model a second time).
		This is what lets a Chunker subclass use Chunking's own
		methods (e.g. save_chunk, load_documents) via inheritance
		while still sharing the one already-loaded model.

		Returns nothing.
		"""
		self.min_chunk_size = chunking.chunk_config.min_chunk_size
		self.max_chunk_size = chunking.chunk_config.max_chunk_size
		self.fixed_chunk_overlap = chunking.chunk_config.fixed_chunk_overlap
		self.semantic_threshold = chunking.chunk_config.semantic_threshold
		self.model = chunking.model
		self.tokenizer = chunking.tokenizer
		
		self.all_embeddings = chunking.all_embeddings
		self.embeddings_offset = chunking.embeddings_offset

	def _finalize_chunk(self, doc_id: str, chunk_idx: int, chunk_text: str, chunk_tokens: list) -> Chunk:
		"""
		doc_id: the document this chunk belongs to, used to build its id.
		chunk_idx: this chunk's position among the chunks already
			produced for doc_id, used to build its id.
		chunk_text: the chunk's final text.
		chunk_tokens: the token ids the chunk was built from, used only
			to report chunk.size.

		Wraps chunk_text and the token count into a Chunk, stamped with
		this strategy's own chunk_type. Shared by every concrete
		Chunker subclass, so a chunk built by any strategy has the same
		id format and size accounting.

		Returns the Chunk.
		"""
		chunk_id = f"{doc_id}_{chunk_idx}"
		chunk_size = len(chunk_tokens)
		chunk = Chunk(chunk_id, chunk_text, chunk_size, self.chunk_type)
		return chunk

	def _split_tokens_into_chunks(self, doc_id: str, tokens: list, chunk_idx: int = 0) -> tuple[dict, int]:
		"""
		doc_id: the document these tokens belong to, used to build
			chunk ids.
		tokens: a flat token-id list to split -- a whole document's
			tokens for fixed chunking, or one oversized sentence's
			tokens for the sentence/semantic fallback.
		chunk_idx: chunk number to start numbering from, so a caller
			that already emitted earlier chunks for this document can
			continue the sequence instead of restarting at 0.

		Slides a max_chunk_size window across tokens, advancing by
		(max_chunk_size - fixed_chunk_overlap) tokens each step, and
		turns each window into a Chunk via _finalize_chunk. Shared by
		every concrete Chunker subclass that needs this fixed-window
		splitting behavior, whether as its whole strategy (fixed
		chunking) or as a fallback for a unit too large to keep whole
		(sentence/semantic chunking).

		Returns the dict of produced chunks, and the next chunk_idx to
		continue numbering from.
		"""
		chunks = {}
		start = 0

		while start < len(tokens):
			end = start + self.max_chunk_size

			window_tokens = tokens[start:end]
			window_text = self.tokenizer.decode(window_tokens, skip_special_tokens=True)

			chunk = self._finalize_chunk(doc_id, chunk_idx, window_text, window_tokens)
			chunks[chunk.id] = chunk

			start += self.max_chunk_size - self.fixed_chunk_overlap
			chunk_idx += 1

		return chunks, chunk_idx

	@abc.abstractmethod
	def chunk(self, document: Document) -> dict[str, Chunk]:
		"""
		document: the Document to split into chunks.

		Splits document into a sequence of Chunks according to this
		strategy's rule. self.chunk_type identifies which strategy
		this is, for stamping into each produced Chunk's type field.

		Returns the list of Chunks produced from document.
		"""
		raise NotImplementedError


def get_chunker(chunk_type: str) -> type[Chunker]:
	"""
	chunk_type: name of the chunking strategy to look up (e.g.
		"fixed", "sentence", "semantic" from common.constants).

	Looks up chunk_type in the registry of implemented chunking
	strategies, mirroring l1_doc_processing.parser.get_parser's
	doc_type -> parser lookup. Validates that the class registered
	under chunk_type declares that same chunk_type as its own,
	catching a registry entry that was mistakenly pointed at the
	wrong strategy class.

	Raises ValueError if chunk_type has no implemented strategy
	registered, or if the registered class's own chunk_type does not
	match the key it is registered under.

	Returns the Chunker subclass registered for chunk_type.
	"""
	from common.constants import FIXED_CHUNKING, SENTENCE_CHUNKING, SEMANTIC_CHUNKING
	from l2_chunking.chunkers.fixed import FixedChunking
	from l2_chunking.chunkers.sentence import SentenceChunking
	from l2_chunking.chunkers.semantic import SemanticChunking
	# imported here, not at module level, to avoid a circular import:
	# each strategy module will subclass Chunker, so importing them at
	# module level would make this module depend on them before Chunker
	# has finished being defined.
	chunker_registry: dict[str, type[Chunker]] = {
		FIXED_CHUNKING: FixedChunking,
		SENTENCE_CHUNKING: SentenceChunking,
		SEMANTIC_CHUNKING: SemanticChunking,
	}

	if chunk_type not in chunker_registry:
		raise ValueError(
			f"Unknown chunk_type '{chunk_type}'. Registered types: {list(chunker_registry.keys())}"
		)

	chunker_cls = chunker_registry[chunk_type]
	if getattr(chunker_cls, "chunk_type", None) != chunk_type:
		raise ValueError(
			f"Registry inconsistency: chunk_type '{chunk_type}' is "
			f"registered to {chunker_cls.__name__}, whose own "
			f"chunk_type is '{getattr(chunker_cls, 'chunk_type', None)}'."
		)
	return chunker_cls