import numpy as np
import torch

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from common import constants
from common import paths
from common.args import EmbedderArgs
from common.args import EmbeddingArgs
from rag.build.embedding.cache import load_embeddings
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
	def __init__(self, embedding_args: EmbeddingArgs):
		"""
		Input: embedding_args, the embedder settings and the chunking settings of one chunk type.
		Output: nothing.
		What it does: stores embedding_args and loads the embedding model of the embedder's provider.
		"""
		# The folder compute_embeddings saved to, for the build manifest
		self.embedding_path = None
		self.embedding_args = embedding_args

		self.model = make_embedding_model(self.embedding_args.embedder_args)

	def compute_embeddings(self, chunks: dict[str, Chunk], output_path: str) -> dict[str, Embedded]:
		"""
		Input: chunks, a dict of chunk_id to Chunk; output_path, the dataset's output folder.
		Output: a dict of chunk_id to Embedded.
		What it does: loads saved embeddings, embeds the remaining chunks with LangChain and saves them.
		"""
		# 1. Load saved embeddings, and collect the chunks that still need embeddings
		# Saved per document. A chunk id is "{doc_id}_{n}", so its doc_id is the part before the last "_"
		embedding_path = paths.get_embedding_path(output_path, self.embedding_args.embedder_args, self.embedding_args.chunking_args)
		print(f"Embedding folder : {embedding_path}")
		self.embedding_path = embedding_path
		cached_embeddings = {}
		uncached_chunk_ids = []
		uncached_chunk_texts = []
		for chunk_id in chunks:
			if chunk_id not in cached_embeddings:
				doc_id = chunk_id.rsplit("_", 1)[0]
				document_embeddings = load_embeddings(embedding_path, doc_id)
				if document_embeddings is not None:
					cached_embeddings.update(document_embeddings)

			if chunk_id not in cached_embeddings:
				uncached_chunk_ids.append(chunk_id)
				uncached_chunk_texts.append(chunks[chunk_id].text)

		# 2. Embed the remaining chunks in one call, grouped by document for saving
		new_embeddings_by_document = {}
		if len(uncached_chunk_texts) > 0:
			embedding_vectors = self.model.embed_documents(uncached_chunk_texts)
			for index in range(len(uncached_chunk_ids)):
				chunk_id = uncached_chunk_ids[index]
				doc_id = chunk_id.rsplit("_", 1)[0]
				embedding_vector = np.array(embedding_vectors[index], dtype=np.float32)
				if doc_id not in new_embeddings_by_document:
					new_embeddings_by_document[doc_id] = {}
				new_embeddings_by_document[doc_id][chunk_id] = Embedded(chunk_id, embedding_vector, self.embedding_args.embedder_args.normalize_embeddings)

		# 3. Save the new embeddings, one file per document
		embeddings = {}
		embeddings.update(cached_embeddings)
		for doc_id in new_embeddings_by_document:
			new_embeddings = new_embeddings_by_document[doc_id]
			save_embeddings(embedding_path, doc_id, new_embeddings)
			embeddings.update(new_embeddings)

		new_embedding_count = len(embeddings) - len(cached_embeddings)
		print(f"Embeddings : {len(embeddings)} ({len(cached_embeddings)} from cache, {new_embedding_count} new)")

		return embeddings
