import os 
from common import paths
from common.chunk_config import ChunkConfig
from l4_indexing.schema import Index

class Indexing:
	def __init__(self, chunk_config: ChunkConfig, dataset_size:int):
		self.chunk_config = chunk_config
		self.dataset_size = dataset_size
	
	def load_document_ids(self, data_path: str, dataset_size:int) -> list[str]:
		
		if not os.path.exists(data_path):
			raise FileNotFoundError(
				f"'{data_path}' does not exist. Run l1_doc_processing "
				f"for this dataset before chunking and embedding it."
			)

		sorted_file_names = sorted(os.listdir(data_path))

		if len(sorted_file_names) == 0:
			raise ValueError(
				f"'{data_path}' exists but holds no processed "
				f"documents. Run l1_doc_processing for this dataset "
				f"before chunking and embedding it."
			)

		if dataset_size > len(sorted_file_names):
			raise ValueError(
				f"Requested dataset_size {dataset_size} exceeds the "
				f"{len(sorted_file_names)} document(s) already processed "
				f"under '{data_path}'. Run l1_doc_processing with a "
				f"dataset_size of at least {dataset_size} first."
			)

		selected_file_names = sorted_file_names[:dataset_size]

		doc_ids = []
		for file_name in selected_file_names:
			doc_id, _ = os.path.splitext(file_name)
			doc_ids.append(doc_id)
		return doc_ids
	
	def compute_index(self, index_type: str)-> Index:
		index = None

		# imported here, not at module level, to avoid importing the
		# whole l2_chunking package (and its heavier dependencies)
		# unless compute_embeddings is actually called.
		from l2_chunking.chunker import get_chunker

		# Raises ValueError for an unregistered chunk_type up front,
		# so a typo'd --chunk_type fails loudly instead of silently
		# matching no cached chunks and returning an empty result.
		get_chunker(self.chunk_config.chunk_type)

		# Load document_ids
		data_path = paths.get_processed_dataset_path(self.chunk_config.dataset_name)
		document_ids = self.load_document_ids(data_path, self.dataset_size)

		# Load embeddings and cached indexing
		embedding_path = paths.get_embeddings_path(self.chunk_config)
		output_path = paths.get_index_path(index_type, self.chunk_config)

		return index
