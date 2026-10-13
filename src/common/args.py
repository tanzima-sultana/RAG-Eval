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

@dataclasses.dataclass
class IndexArgs:
	faiss_index_type: str
	ivf_nlist: int
	hnsw_m: int
	hnsw_ef_construction: int
	bm25_k1: float
	bm25_b: float
	bm25_tokenizer: str
	qdrant_batch_size: int

@dataclasses.dataclass
class EvalQnAArgs:
	eval_qna_llm: str
	eval_qna_model: str
	n_sample_documents: int
	n_sample_evals: int

@dataclasses.dataclass
class QueryArgs:
	embedding_model: str
	query_llm: str
	query_llm_model: str
	n_multi_qus: int
