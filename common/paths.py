
import os

RAW_DATA_DIR = "data/raw/"
PROCESSED_DATA_DIR = "data/processed/"


# ------ Dataset
def dataset_raw_path(doc_type: str, dataset_name: str) -> str:
    return RAW_DATA_DIR + f"{doc_type}_{dataset_name}/"

def dataset_processed_path(doc_type: str, dataset_name: str) -> str:
    return PROCESSED_DATA_DIR + f"{doc_type}_{dataset_name}/docs/"

def get_file_path(path: str, file_name: str, extension: str) -> str:
    return os.path.join(path, f"{file_name}.{extension}")