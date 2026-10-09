import torch

from sentence_transformers import SentenceTransformer

from common.args import ChunkingArgs
from common import paths

from l2_chunking.schema import Chunk
from l3_embedding.schema import Embedded
from l3_embedding import cache

class Embedding:
	def __init__(self, chunking_args: ChunkingArgs):
		self.chunking_args = chunking_args

		device = "cuda" if torch.cuda.is_available() else "cpu"
		self.model = SentenceTransformer(self.chunking_args.model_name, device=device)

	def compute_embeddings(self, chunks: dict[str, Chunk], output_path:str) -> dict[str, Embedded]:
		"""
		Input: chunks, a dict of chunk_id to Chunk from l2_chunking;
		output_path, the dataset's output folder the embedding folder is built under.
		Output: a dict of chunk_id to Embedded.
		What it does: loads the cached embeddings of each chunk's document
		from the embedding folder, collects the chunks that have no cached
		embedding, embeds those in one batch with the model, normalized to
		length 1, and saves one file per document that got new embeddings.
		Prints how many embeddings came from cache and how many are new.
		"""

		# 1. Use langchain if set
		#if self.chunking_args.framework == constants.LANGCHAIN_FRAMEWORK:
		#	return Langchain(self.chunking_args).compute_embeddings(chunks, output_path)
		
		embeddings = {}
		embedding_path = paths.get_embedding_path(output_path, self.chunking_args)

		# 2. Load cached embeddings, and collect the chunks that still need embeddings
		# Embeddings are cached in one file per document. A chunk id is
		# "{doc_id}_{n}", so each chunk's document id is the part before the
		# last "_". A document's file is read when one of its chunks is not
		# in cached_embeddings yet; that adds all its chunks, so the file is
		# not read again. The ids and texts lists stay in step, so step 3
		# can embed all texts in one batch.
		cached_embeddings = {}
		uncached_chunk_ids = []
		uncached_chunk_texts = []
		for chunk_id in chunks:
			if chunk_id not in cached_embeddings:
				doc_id = chunk_id.rsplit("_", 1)[0]
				document_embeddings = cache._load_cached_embeddings(embedding_path, doc_id)
				if document_embeddings is not None:
					cached_embeddings.update(document_embeddings)

			if chunk_id not in cached_embeddings:
				uncached_chunk_ids.append(chunk_id)
				uncached_chunk_texts.append(chunks[chunk_id].text)

		# 3. Embed the chunks that still need embeddings
		# All texts go to the model in one call, in batches of 256. Vectors
		# are normalized to length 1, so a dot product is cosine similarity.
		# The new embeddings are grouped by document, because the cache is
		# one file per document.
		new_embeddings_by_document = {}
		if len(uncached_chunk_texts) > 0:
			embedding_vectors = self.model.encode(
				uncached_chunk_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True
			)
			for index in range(len(uncached_chunk_ids)):
				chunk_id = uncached_chunk_ids[index]
				doc_id = chunk_id.rsplit("_", 1)[0]
				if doc_id not in new_embeddings_by_document:
					new_embeddings_by_document[doc_id] = {}
				new_embeddings_by_document[doc_id][chunk_id] = Embedded(chunk_id, embedding_vectors[index], True)

		# 4. Merge cached_embeddings & new_embeddings into embeddings and save
		embeddings.update(cached_embeddings)
		for doc_id in new_embeddings_by_document:
			new_embeddings = new_embeddings_by_document[doc_id]
			cache._save_embeddings(embedding_path, doc_id, new_embeddings)
			embeddings.update(new_embeddings)

		print(f"Embedding folder : {embedding_path}")
		print(f"Embeddings : {len(embeddings)} ({len(cached_embeddings)} from cache, {len(embeddings) - len(cached_embeddings)} new)")

		return embeddings