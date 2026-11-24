from langchain_classic.retrievers.ensemble import unique_by_key
from langchain_core.documents import Document as LangchainDocument
from langchain_core.retrievers import BaseRetriever

from schema.retrieved import Retrieved
from schema.setup import Setup


def invoke_retriever(retriever: BaseRetriever, query_texts: list[str], top_k: int) -> list[LangchainDocument]:
	"""
	Input: retriever, the LangChain retriever of one setup; query_texts, the texts to search with, one for original,
	rewritten and hyde, one per phrasing for multi, none for a HyDE I don't know; top_k, the chunks to keep.
	Output: the found LangChain Documents, best first, at most top_k.
	What it does: invokes the retriever with each text, joins the results in order and keeps each chunk the first time
	it appears, by its chunk id, with LangChain's unique_by_key (the one EnsembleRetriever uses), then keeps the first
	top_k. The chunk id is used, not the whole Document, since a FAISS and a BM25 copy of one chunk differ in their id.
	"""
	documents = []
	for query_text in query_texts:
		documents.extend(retriever.invoke(query_text))

	unique_documents = list(unique_by_key(documents, key=lambda document: document.metadata["chunk_id"]))
	return unique_documents[:top_k]


def make_retrieved(query_id: str, setup: Setup, top_k: int, rerank_top_k: int, documents: list[LangchainDocument]) -> Retrieved:
	"""
	Input: query_id, the id of the query; setup, the setup that found the Documents; top_k; rerank_top_k, 0 when not
	reranked; documents, the found LangChain Documents, best first.
	Output: the Retrieved of the query and setup.
	What it does: takes each Document's chunk id from its metadata, and the document id as the part of the chunk id
	before the last "_".
	"""
	chunk_ids = []
	doc_ids = []
	for document in documents:
		chunk_id = document.metadata["chunk_id"]
		chunk_ids.append(chunk_id)
		doc_ids.append(chunk_id.rsplit("_", 1)[0])

	return Retrieved(
		id=query_id,
		setup=setup,
		top_k=top_k,
		rerank_top_k=rerank_top_k,
		chunk_ids=chunk_ids,
		doc_ids=doc_ids
	)
