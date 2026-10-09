"""
args.py
"""

import dataclasses

@dataclasses.dataclass
class IngestArgs:
	dataset_size: int 
	input_path: str
	output_path: str 
	download_workers: int 
	max_batch_rows: int 
	hugging_face_split: str 
	parse_workers: int 
	parse_timeout_seconds: int 
	dedup_threshold: float 
	pdf_parser: str 

@dataclasses.dataclass
class ChunkingArgs:
	framework: str
	model_name: str
	chunk_type: str
	min_chunk_size: int
	max_chunk_size: int
	fixed_chunk_overlap: int
	semantic_threshold: float


