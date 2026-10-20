"""
langchain.py

Retrieval on the LangChain indexes, used with --framework langchain. Searches them with the same query
embeddings and tokens as the built-in indexes, so only the chunking and indexing differ.
"""

from common.constants import HYDE_QUERY
from common.constants import I_DONT_KNOW
from common.constants import MULTI_QUERY
from common.constants import ORIGINAL_QUERY
from common.constants import REWRITTEN_QUERY
from stages.build.indexing.schema import LangchainFaissIndex
from stages.build.indexing.schema import LangchainBm25Index
from stages.build.indexing.schema import LangchainQdrantIndex
from stages.question.query.schema import Query
from l6_retrieval.schema import Retrieved
from l6_retrieval.retrievars.dense import make_dense_retrieved
from l6_retrieval.retrievars.bm25 import make_bm25_retrieved

class Langchain:
	def __init__(self, top_k: int, query_types: list[str]):
		"""
		Input: top_k, the number of chunks to keep per query; query_types, the query types to search with.
		Output: none.
		What it does: stores top_k and query_types.
		"""
		self.top_k = top_k
		self.query_types = query_types

	def get_dense_outputs(self, queries: list[Query], chunk_type: str, index_type: str, index: LangchainFaissIndex | LangchainQdrantIndex) -> list[Retrieved]:
		"""
		Input: the queries, the chunk type, one LangChain FAISS or Qdrant vector store with its index type.
		Output: one Retrieved per query and query type in query_types, not reranked.
		What it does: searches the vector store with each query type's embeddings and keeps the top_k chunks.
		"""
		dense_outputs = []

		for query in queries:
			# 1. Collect the embeddings of each query type
			multi_vectors = []
			for multi_qus in query.multi_qus:
				multi_vectors.append(multi_qus.embeddings)

			# A HyDE answer the LLM refused to write is not searched
			hyde_vectors = []
			if query.hyde_ans.ans != I_DONT_KNOW:
				hyde_vectors.append(query.hyde_ans.embeddings)

			query_types_and_vectors = []
			query_types_and_vectors.append((ORIGINAL_QUERY, [query.original_qus.embeddings]))
			query_types_and_vectors.append((REWRITTEN_QUERY, [query.rewritten_qus.embeddings]))
			query_types_and_vectors.append((MULTI_QUERY, multi_vectors))
			query_types_and_vectors.append((HYDE_QUERY, hyde_vectors))

			# 2. Search the vector store with each query type
			# A query type with no embeddings gets a Retrieved with no chunks, so it still counts as a miss
			for query_type, query_vectors in query_types_and_vectors:
				if query_type not in self.query_types:
					continue
				hits = []
				for query_vector in query_vectors:
					# The score is the cosine similarity, since the vectors are normalized
					documents_and_scores = index.vector_store.similarity_search_with_score_by_vector(query_vector.tolist(), k=self.top_k)
					for document, score in documents_and_scores:
						hits.append((document.metadata["chunk_id"], float(score)))
				retrieved = make_dense_retrieved(query.id, query_type, chunk_type, index_type, self.top_k, hits)
				dense_outputs.append(retrieved)

		return dense_outputs

	def get_bm25_outputs(self, queries: list[Query], chunk_type: str, index_type: str, index: LangchainBm25Index) -> list[Retrieved]:
		"""
		Input: the queries, the chunk type, the LangChain BM25 retriever with its index type.
		Output: one Retrieved per query and query type in query_types (original, rewritten, multi), not reranked.
		What it does: scores every chunk with the retriever's BM25 model and each query type's tokens, and keeps the
		top_k chunks. HyDE has no tokens, so it is not searched.
		"""
		bm25_outputs = []

		# BM25 position i holds the Document of chunk i
		chunk_ids = []
		for document in index.retriever.docs:
			chunk_ids.append(document.metadata["chunk_id"])

		for query in queries:
			# 1. Collect the tokens of each query type
			multi_tokens = []
			for multi_qus in query.multi_qus:
				multi_tokens.append(multi_qus.tokens)

			query_types_and_tokens = []
			query_types_and_tokens.append((ORIGINAL_QUERY, [query.original_qus.tokens]))
			query_types_and_tokens.append((REWRITTEN_QUERY, [query.rewritten_qus.tokens]))
			query_types_and_tokens.append((MULTI_QUERY, multi_tokens))

			# 2. Search with each query type
			# BM25Retriever.invoke returns no scores, so its rank_bm25 model is scored directly
			for query_type, query_tokens_list in query_types_and_tokens:
				if query_type not in self.query_types:
					continue
				hits = []
				for query_tokens in query_tokens_list:
					scores = index.retriever.vectorizer.get_scores(query_tokens)
					# Positions of the top_k scores, best first
					top_positions = scores.argsort()[::-1][:self.top_k]
					for position in top_positions:
						score = float(scores[position])
						# A chunk with score 0 shares no word with the query
						if score <= 0.0:
							continue
						hits.append((chunk_ids[int(position)], score))
				retrieved = make_bm25_retrieved(query.id, query_type, chunk_type, index_type, self.top_k, hits)
				bm25_outputs.append(retrieved)

		return bm25_outputs
