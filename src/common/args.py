"""
args.py
"""

import dataclasses

@dataclasses.dataclass
class IngestArgs:
	dataset_size: int = 0
	input_path: str = ""
	output_path: str = ""
	download_workers: int = 1
	max_batch_rows: int = 1
	hugging_face_split: str = "train"
	parse_workers: int = 16
	parse_timeout_seconds: int = 0
	dedup_threshold: float = 0.0
	pdf_parser: str = "pypdf"


