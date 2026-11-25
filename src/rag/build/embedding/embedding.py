import numpy as np
import torch

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from common import constants
from common import paths
from common.args import EmbedderArgs
from common.args import EmbeddingArgs
from rag.build.embedding.cache import load_document_embeddings
from rag.build.embedding.cache import save_embeddings
from schema.chunk import Chunk
from schema.embed import Embedded


def make_huggingface_embedding_model(embedder_args: EmbedderArgs) -> HuggingFaceEmbeddings:
	"""
	Input: embedder_args, the shared embedder settings.
	Output: LangChain's HuggingFaceEmbeddings.
	What it does: loads the model on cuda if it is available, otherwise on cpu, with batch_size and
	normalize_embeddings from embedder_args.
	"""
	device = "cuda" if torch.cuda.is_available() else "cpu"
	embedding_model = HuggingFaceEmbeddings(
		model_name=embedder_args.model_name,
		model_kwargs={"device": device},
		encode_kwargs={"batch_size": embedder_args.batch_size, "normalize_embeddings": embedder_args.normalize_embeddings},
		show_progress=True
	)
	return embedding_model


def make_embedding_model(embedder_args: EmbedderArgs) -> Embeddings:
	"""
	Input: embedder_args, the shared embedder settings.
	Output: a LangChain Embeddings of the embedding provider.
	What it does: calls the method of embedder_args.embedding_provider, and raises ValueError if the provider is not
	supported.
	"""
	if embedder_args.embedding_provider == constants.HUGGINGFACE_EMBEDDING:
		return make_huggingface_embedding_model(embedder_args)

	raise ValueError(
		f"embedding_provider ({embedder_args.embedding_provider}) is not supported. Use "
		f"\"{constants.HUGGINGFACE_EMBEDDING}\"."
	)


class Embedding:
	def __init__(self, embedding_args: EmbeddingArgs, chunking_hash: str):
		"""
		Input: embedding_args, the embedder settings and the chunking settings of one chunk type; chunking_hash, the hash
		of the chunking run the chunks are from.
		Output: nothing.
		What it does: stores both. The embedding model is loaded only when some chunks need embedding.
		"""
		# The folder compute_embeddings saved to, for the build manifest
		self.embedding_path = None
		self.embedding_args = embedding_args
		self.chunking_hash = chunking_hash
		self.model = None

	def compute_embeddings(self, chunks: dict[str, Chunk], output_path: str) -> tuple[dict[str, Embedded], str, str]:
		"""
		Input: chunks, a dict of chunk_id to Chunk, from chunking; output_path, the dataset's output folder.
		Output: (embeddings, embedding_hash, embedding_path): a dict of chunk_id to Embedded for every chunk, in chunk order,
		the hash of this embedding run, and the folder the embeddings are saved in.
		What it does: loads the saved embeddings of every document whose chunks are the same and were embedded with the
		same normalize_embeddings, embeds only the chunks of the other documents with LangChain, and saves them. If one
		document changes, only its chunks are embedded.
		"""
		embedder_args = self.embedding_args.embedder_args

		# 1. Name the folder of these settings and this embedding run
		embedding_path = paths.get_embedding_path(output_path, embedder_args, self.embedding_args.chunking_args)
		embedding_hash = paths.get_embedding_hash(embedder_args, self.chunking_hash)
		print(f"Embedding folder : {embedding_path}")
		self.embedding_path = embedding_path

		# 2. Group the chunks by document. A chunk id is "{document_id}_{n}", so its document id is the part before the
		# first "_"
		chunks_by_document_id = {}
		for chunk_id in chunks:
			document_id = chunk_id.split("_", 1)[0]
			if document_id not in chunks_by_document_id:
				chunks_by_document_id[document_id] = []
			chunks_by_document_id[document_id].append(chunks[chunk_id])

		# 3. Load the saved embeddings of each document. They are used only if they are of exactly this document's
		# chunks and were normalized the way the config says
		cached_embeddings_by_document_id = {}
		chunks_to_embed = []
		for document_id in chunks_by_document_id:
			document_chunks = chunks_by_document_id[document_id]
			saved_embeddings = load_document_embeddings(embedding_path, document_id)

			is_saved_and_same = saved_embeddings is not None
			if is_saved_and_same:
				document_chunk_ids = []
				for chunk in document_chunks:
					document_chunk_ids.append(chunk.chunk_id)
				if list(saved_embeddings.keys()) != document_chunk_ids:
					is_saved_and_same = False
			if is_saved_and_same:
				for chunk_id in saved_embeddings:
					if saved_embeddings[chunk_id].normalized != embedder_args.normalize_embeddings:
						is_saved_and_same = False

			if is_saved_and_same:
				cached_embeddings_by_document_id[document_id] = saved_embeddings
			else:
				for chunk in document_chunks:
					chunks_to_embed.append(chunk)

		# 4. Embed the other documents' chunks in one call
		new_embeddings = {}
		if len(chunks_to_embed) > 0:
			if self.model is None:
				self.model = make_embedding_model(embedder_args)

			chunk_texts_to_embed = []
			for chunk in chunks_to_embed:
				chunk_texts_to_embed.append(chunk.chunk_texts)
			embedding_vectors = self.model.embed_documents(chunk_texts_to_embed)

			for index in range(len(chunks_to_embed)):
				chunk_id = chunks_to_embed[index].chunk_id
				embedding_vector = np.array(embedding_vectors[index], dtype=np.float32)
				new_embeddings[chunk_id] = Embedded(chunk_id, embedding_vector, embedder_args.normalize_embeddings)

		# 5. Put the saved and the new embeddings together in chunk order
		embeddings = {}
		for chunk_id in chunks:
			document_id = chunk_id.split("_", 1)[0]
			if document_id in cached_embeddings_by_document_id:
				embeddings[chunk_id] = cached_embeddings_by_document_id[document_id][chunk_id]
			else:
				embeddings[chunk_id] = new_embeddings[chunk_id]

		# 6. Save the new embeddings and the list of this run, so the next run loads them from the cache
		document_ids = list(chunks_by_document_id.keys())
		save_embeddings(embedding_path, new_embeddings, document_ids, embedding_hash)

		cached_chunk_count = len(embeddings) - len(new_embeddings)
		print(f"Documents : {len(document_ids)} ({len(cached_embeddings_by_document_id)} from cache)")
		print(f"Embeddings : {len(embeddings)} ({cached_chunk_count} from cache, {len(new_embeddings)} new)")

		return embeddings, embedding_hash, embedding_path
