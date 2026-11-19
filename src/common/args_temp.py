import argparse
import dataclasses

import yaml


@dataclasses.dataclass
class IngestionArgs:
	dataset_size: int
	dataset_name: str
	input_path: str
	output_path: str
	document_type: str


def make_ingestion_args(parsed_args: argparse.Namespace) -> IngestionArgs:
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_args().
	Output: an IngestionArgs holding the ingestion settings.
	What it does: copies each ingestion setting into the matching IngestionArgs field.
	"""
	ingestion_args = IngestionArgs(
		dataset_size=parsed_args.dataset_size,
		dataset_name=parsed_args.dataset_name,
		input_path=parsed_args.input_path,
		output_path=parsed_args.output_path,
		document_type=parsed_args.document_type
	)
	return ingestion_args


@dataclasses.dataclass
class ChunkingArgs:
	model_name: str
	chunk_type: str
	min_chunk_size: int
	max_chunk_size: int
	fixed_chunk_overlap: int
	semantic_threshold: float


def make_chunking_args(parsed_args: argparse.Namespace, chunk_type: str) -> ChunkingArgs:
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_args(); chunk_type, one of chunk_types.
	Output: a ChunkingArgs holding the chunking settings for that chunk type.
	What it does: copies each chunking setting and chunk_type into the matching ChunkingArgs field.
	"""
	chunking_args = ChunkingArgs(
		model_name=parsed_args.model_name,
		chunk_type=chunk_type,
		min_chunk_size=parsed_args.min_chunk_size,
		max_chunk_size=parsed_args.max_chunk_size,
		fixed_chunk_overlap=parsed_args.fixed_chunk_overlap,
		semantic_threshold=parsed_args.semantic_threshold
	)
	return chunking_args


@dataclasses.dataclass
class EmbeddingArgs:
	chunking_args: ChunkingArgs


def make_embedding_args(chunking_args: ChunkingArgs) -> EmbeddingArgs:
	"""
	Input: chunking_args, the chunking settings of one chunk type.
	Output: an EmbeddingArgs holding them.
	What it does: wraps chunking_args into an EmbeddingArgs.
	"""
	embedding_args = EmbeddingArgs(
		chunking_args=chunking_args
	)
	return embedding_args


@dataclasses.dataclass
class IndexArgs:
	chunking_args: ChunkingArgs
	index_type: str
	ivf_nlist: int
	hnsw_m: int
	hnsw_ef_construction: int
	bm25_k1: float
	bm25_b: float
	bm25_tokenizer: str


def make_index_args(parsed_args: argparse.Namespace, chunking_args: ChunkingArgs, index_type: str) -> IndexArgs:
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_args(); chunking_args, the chunking settings of one
	chunk type; index_type, one of index_types.
	Output: an IndexArgs holding the indexing settings for that index type.
	What it does: copies chunking_args, index_type and each indexing setting into the matching IndexArgs field.
	"""
	index_args = IndexArgs(
		chunking_args=chunking_args,
		index_type=index_type,
		ivf_nlist=parsed_args.ivf_nlist,
		hnsw_m=parsed_args.hnsw_m,
		hnsw_ef_construction=parsed_args.hnsw_ef_construction,
		bm25_k1=parsed_args.bm25_k1,
		bm25_b=parsed_args.bm25_b,
		bm25_tokenizer=parsed_args.bm25_tokenizer
	)
	return index_args


@dataclasses.dataclass
class VectorDbArgs:
	chunking_args: ChunkingArgs
	vectordb_type: str
	qdrant_url: str
	qdrant_batch_size: int


def make_vectordb_args(parsed_args: argparse.Namespace, chunking_args: ChunkingArgs, vectordb_type: str) -> VectorDbArgs:
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_args(); chunking_args, the chunking settings of one
	chunk type; vectordb_type, one of vectordb_types.
	Output: a VectorDbArgs holding the vector DB settings for that vector DB type.
	What it does: copies chunking_args, vectordb_type and each vector DB setting into the matching VectorDbArgs field.
	"""
	vectordb_args = VectorDbArgs(
		chunking_args=chunking_args,
		vectordb_type=vectordb_type,
		qdrant_url=parsed_args.qdrant_url,
		qdrant_batch_size=parsed_args.qdrant_batch_size
	)
	return vectordb_args


def parse_vectordb_args(parser: argparse.ArgumentParser) -> None:
	"""
	Input: parser, the argument parser of main.py.
	Output: nothing.
	What it does: adds the vector DB arguments to parser.
	"""
	parser.add_argument(
		"--vectordb_types",
		type=str,
		nargs="+",
		required=True,
		help="Vector DB types to build, one after another: qdrant")
	parser.add_argument(
		"--qdrant_url",
		type=str,
		required=True,
		help="URL of the Qdrant server, e.g. http://localhost:6333")
	parser.add_argument(
		"--qdrant_batch_size",
		type=int,
		required=True,
		help="Number of points uploaded to Qdrant in one request")


@dataclasses.dataclass
class OllamaArgs:
	ollama_api_base: str
	ollama_context_length: int


def make_ollama_args(parsed_args: argparse.Namespace) -> OllamaArgs:
	"""
	Input: parsed_args, the argparse.Namespace returned by parse_args().
	Output: an OllamaArgs holding the Ollama settings.
	What it does: copies each Ollama setting into the matching OllamaArgs field.
	"""
	ollama_args = OllamaArgs(
		ollama_api_base=parsed_args.ollama_api_base,
		ollama_context_length=parsed_args.ollama_context_length
	)
	return ollama_args


def parse_ollama_args(parser: argparse.ArgumentParser) -> None:
	"""
	Input: parser, the argument parser of main.py.
	Output: nothing.
	What it does: adds the Ollama arguments to parser.
	"""
	parser.add_argument(
		"--ollama_api_base",
		type=str,
		required=True,
		help="URL of the local Ollama server, e.g. http://localhost:11434")
	parser.add_argument(
		"--ollama_context_length",
		type=int,
		required=True,
		help="Context window in tokens of an Ollama model; the prompt and the reply must fit in it together")


def parse_indexing_args(parser: argparse.ArgumentParser) -> None:
	"""
	Input: parser, the argument parser of main.py.
	Output: nothing.
	What it does: adds the indexing arguments to parser.
	"""
	parser.add_argument(
		"--index_types",
		type=str,
		nargs="+",
		required=True,
		help="Index types to build, one after another: faiss_flatip, faiss_ivf, faiss_hnsw and/or bm25")
	parser.add_argument(
		"--ivf_nlist",
		type=int,
		required=True,
		help="Number of clusters of the FAISS IVF index")
	parser.add_argument(
		"--hnsw_m",
		type=int,
		required=True,
		help="Number of neighbours per node of the FAISS HNSW index")
	parser.add_argument(
		"--hnsw_ef_construction",
		type=int,
		required=True,
		help="Candidate list size used while building the FAISS HNSW index")
	parser.add_argument(
		"--bm25_k1",
		type=float,
		required=True,
		help="BM25 term frequency saturation")
	parser.add_argument(
		"--bm25_b",
		type=float,
		required=True,
		help="BM25 document length normalization")
	parser.add_argument(
		"--bm25_tokenizer",
		type=str,
		required=True,
		help="Tokenizer used by BM25: whitespace")


def parse_chunking_args(parser: argparse.ArgumentParser) -> None:
	"""
	Input: parser, the argument parser of main.py.
	Output: nothing.
	What it does: adds the chunking arguments to parser.
	"""
	parser.add_argument(
		"--model_name",
		type=str,
		required=True,
		help="SentenceTransformer model used for token counts and semantic embeddings")
	parser.add_argument(
		"--chunk_types",
		type=str,
		nargs="+",
		required=True,
		help="Chunk types to build, one after another: fixed, sentence and/or semantic")
	parser.add_argument(
		"--min_chunk_size",
		type=int,
		required=True,
		help="Minimum chunk size")
	parser.add_argument(
		"--max_chunk_size",
		type=int,
		required=True,
		help="Maximum chunk size")
	parser.add_argument(
		"--fixed_chunk_overlap",
		type=int,
		required=True,
		help="Overlap between consecutive fixed-size chunks")
	parser.add_argument(
		"--semantic_threshold",
		type=float,
		required=True,
		help="Similarity threshold used by semantic chunking")


def parse_ingestion_args(parser: argparse.ArgumentParser) -> None:
	"""
	Input: parser, the argument parser of main.py.
	Output: nothing.
	What it does: adds the ingestion arguments to parser.
	"""
	parser.add_argument(
		"--dataset_size",
		type=int,
		required=True,
		help="Number of documents to ingest")
	parser.add_argument(
		"--dataset_name",
		type=str,
		required=True,
		help="Name of the dataset, used as its output folder name")
	parser.add_argument(
		"--input_path",
		type=str,
		required=True,
		help="Local file or folder to ingest")
	parser.add_argument(
		"--output_path",
		type=str,
		required=True,
		help="Root output folder")
	parser.add_argument(
		"--document_type",
		type=str,
		required=True,
		help="Type of documents to keep, e.g. pdf")


def parse_args() -> argparse.Namespace:
	"""
	Input: none. Reads --config and any override flags from the command line.
	Output: an argparse.Namespace with one attribute per setting.
	What it does: reads the config file, turns each setting into its flag and parses them. A top-level key is one
	setting; a section such as ingestion holds settings. Flags given after --config override the file.
	"""
	# 1. Get the config file path. The other command line flags are overrides
	config_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
	config_parser.add_argument("--config", type=str, required=True, help="YAML config file, e.g. config_temp.yaml")
	config_args, override_arguments = config_parser.parse_known_args()

	with open(config_args.config, encoding="utf-8") as config_file:
		config = yaml.safe_load(config_file)

	# 2. Turn each setting of the file into its flag
	config_arguments = []
	for key in config:
		value = config[key]
		if isinstance(value, dict):
			for setting_key in value:
				config_arguments.append("--" + setting_key)
				setting_value = value[setting_key]
				# A list is passed as one flag followed by its values, e.g. --chunk_types fixed sentence
				if isinstance(setting_value, list):
					for item in setting_value:
						config_arguments.append(str(item))
				else:
					config_arguments.append(str(setting_value))
		else:
			config_arguments.append("--" + key)
			config_arguments.append(str(value))

	# 3. Parse the file's flags and then the overrides, so an override replaces the file's value
	parser = argparse.ArgumentParser(description="RAG parameters", allow_abbrev=False)
	parser.add_argument(
		"--framework",
		type=str,
		required=True,
		help="rag or langchain")
	
	parse_ingestion_args(parser)
	parse_chunking_args(parser)
	parse_indexing_args(parser)
	parse_vectordb_args(parser)
	parse_ollama_args(parser)

	parsed_args = parser.parse_args(config_arguments + override_arguments)
	return parsed_args
