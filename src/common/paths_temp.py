import os


# ------ Dataset
def get_output_path(output_path: str, framework: str, dataset_name: str) -> str:
	"""
	Input: output_path, the root output folder; framework; dataset_name.
	Output: {output_path}/{framework}/{dataset_name}
	What it does: builds the dataset's output folder for one framework.
	"""
	return os.path.join(output_path, framework, dataset_name)


# ------ Ingestion
def get_documents_path(output_path: str, dataset_size: int) -> str:
	"""
	Input: output_path, the dataset's output folder; dataset_size.
	Output: {output_path}/documents/{dataset_size}
	What it does: builds the folder that holds the ingested documents.
	"""
	return os.path.join(output_path, "documents", str(dataset_size))
