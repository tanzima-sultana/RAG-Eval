import datetime
import hashlib
import importlib.metadata
import json
import os

from common import paths
from common.args import OllamaArgs
from common.args import EvalDatasetArgs
from evaluation.eval_dataset.cache import is_eval_dataset_complete
from evaluation.eval_dataset.cache import load_eval_dataset
from evaluation.eval_dataset.cache import save_eval_dataset
from evaluation.eval_dataset.cache import save_eval_dataset_metadata
from evaluation.eval_dataset.eval_sample import generate_eval_dataset
from schema.file import File
from schema.eval_dataset_metadata import EvalDatasetMetadata
from schema.eval_sample import EvalSample


class EvalDataset:
	def __init__(self, eval_dataset_args: EvalDatasetArgs, ollama_args: OllamaArgs):
		"""
		Input: eval_dataset_args, the eval dataset settings; ollama_args, the Ollama settings, used only if an eval dataset model is an Ollama model.
		Output: nothing.
		What it does: stores both settings.
		"""
		self.eval_dataset_args = eval_dataset_args
		self.ollama_args = ollama_args

	def validate(self) -> None:
		"""
		Input: none.
		Output: nothing.
		What it does: raises ValueError if an eval dataset setting is not valid. The two LLMs are validated by LLMProvider.
		"""
		eval_dataset_args = self.eval_dataset_args

		if eval_dataset_args.eval_dataset_embedding_model.strip() == "":
			raise ValueError("eval_dataset_embedding_model is empty. Use a SentenceTransformer model name, e.g. all-MiniLM-L6-v2.")

		if eval_dataset_args.n_sample_documents <= 0:
			raise ValueError(f"n_sample_documents ({eval_dataset_args.n_sample_documents}) must be greater than 0.")

		if eval_dataset_args.n_evals_requested <= 0:
			raise ValueError(f"n_evals_requested ({eval_dataset_args.n_evals_requested}) must be greater than 0.")

	def compute_eval_dataset(self, output_path: str) -> list[EvalSample]:
		"""
		Input: output_path, the dataset's output folder.
		Output: a list of EvalSample.
		What it does: if the eval dataset of these args is complete and unchanged, loads it, with no documents and no
		LLM. Otherwise it loads the first dataset_size documents of the document list at documents_path, generates the
		eval dataset with RAGAS, and saves it and its metadata.
		"""
		# 1. Validate the settings
		self.validate()

		# 2. Load the saved eval dataset if it is complete. No documents and no LLM, so this needs no API key or ALLOW_PAID_API
		eval_dataset_path = paths.get_eval_dataset_path(output_path, self.eval_dataset_args, self.ollama_args)
		print(f"Eval dataset file : {eval_dataset_path}")
		if is_eval_dataset_complete(output_path, self.eval_dataset_args, self.ollama_args):
			eval_samples = load_eval_dataset(eval_dataset_path)
			print(f"Eval dataset : {len(eval_samples)} (all from cache)")
			return eval_samples

		# 3. Load the first dataset_size documents of the document list the build saved, only needed to generate.
		# Each document is saved as {doc_id}.json in the same folder as the list
		documents_path = self.eval_dataset_args.documents_path
		dataset_size = self.eval_dataset_args.dataset_size
		with open(documents_path, encoding="utf-8") as f:
			doc_ids = json.load(f)
		if len(doc_ids) < dataset_size:
			raise ValueError(
				f"The document list {documents_path} has {len(doc_ids)} documents, fewer than dataset_size ({dataset_size})."
			)

		documents_folder = os.path.dirname(documents_path)
		documents = {}
		for doc_id in doc_ids[:dataset_size]:
			document_file = os.path.join(documents_folder, f"{doc_id}.json")
			with open(document_file, encoding="utf-8") as f:
				document_dict = json.load(f)
			documents[doc_id] = File.model_validate(document_dict)
		print(f"Documents : {len(documents)} of {len(doc_ids)} in {documents_path}")

		# 4. Generate and save the eval dataset
		print(f"No saved eval dataset, generating {self.eval_dataset_args.n_evals_requested} items with RAGAS")
		eval_samples = generate_eval_dataset(documents, output_path, self.eval_dataset_args, self.ollama_args)
		save_eval_dataset(eval_dataset_path, eval_samples)

		# 5. Save its metadata next to it: what it is and how it was made
		eval_dataset_metadata = self.make_eval_dataset_metadata(documents, output_path, eval_samples)
		eval_dataset_metadata_path = paths.get_eval_dataset_metadata_path(output_path, self.eval_dataset_args, self.ollama_args)
		save_eval_dataset_metadata(eval_dataset_metadata_path, eval_dataset_metadata)
		print(f"Eval dataset metadata saved : {eval_dataset_metadata_path}")
		return eval_samples

	def make_eval_dataset_metadata(self, documents: dict[str, File], output_path: str, eval_samples: list[EvalSample]) -> EvalDatasetMetadata:
		"""
		Input: documents, the document set it was generated from; output_path, the dataset's output folder;
		eval_samples, the saved eval samples.
		Output: an EvalDatasetMetadata.
		What it does: hashes the saved file and the sorted document ids, and collects every setting that shaped the output.
		"""
		eval_dataset_args = self.eval_dataset_args
		eval_dataset_path = paths.get_eval_dataset_path(output_path, eval_dataset_args, self.ollama_args)

		# 1. Hash of the saved eval dataset file
		with open(eval_dataset_path, "rb") as f:
			content_hash = hashlib.sha256(f.read()).hexdigest()

		# 2. Hash of the sorted document ids, one per line
		doc_ids = []
		for doc_id in documents:
			doc_ids.append(doc_id)
		doc_ids.sort()
		documents_hash = hashlib.sha256("\n".join(doc_ids).encode("utf-8")).hexdigest()

		# 3. The Ollama context length shapes the output only when one of the LLMs is an Ollama model
		ollama_providers = ["ollama", "ollama_chat"]
		eval_dataset_llm_provider = eval_dataset_args.eval_dataset_llm_args.llm_model.split("/", 1)[0]
		eval_dataset_kg_llm_provider = eval_dataset_args.eval_dataset_kg_llm_args.llm_model.split("/", 1)[0]
		ollama_context_length = None
		if eval_dataset_llm_provider in ollama_providers or eval_dataset_kg_llm_provider in ollama_providers:
			ollama_context_length = self.ollama_args.ollama_context_length

		eval_dataset_metadata = EvalDatasetMetadata(
			eval_dataset_hash=paths.get_eval_dataset_hash(eval_dataset_args, self.ollama_args),
			eval_dataset_path=eval_dataset_path,
			knowledge_graph_path=paths.get_eval_dataset_kg_path(output_path, eval_dataset_args, self.ollama_args),
			content_hash=content_hash,
			n_evals_generated=len(eval_samples),
			created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
			dataset_name=eval_dataset_args.dataset_name,
			dataset_size=eval_dataset_args.dataset_size,
			documents_path=eval_dataset_args.documents_path,
			documents_hash=documents_hash,
			eval_dataset_llm_model=eval_dataset_args.eval_dataset_llm_args.llm_model,
			eval_dataset_llm_max_tokens=eval_dataset_args.eval_dataset_llm_args.max_tokens,
			eval_dataset_kg_llm_model=eval_dataset_args.eval_dataset_kg_llm_args.llm_model,
			eval_dataset_kg_llm_max_tokens=eval_dataset_args.eval_dataset_kg_llm_args.max_tokens,
			ollama_context_length=ollama_context_length,
			embedding_model=eval_dataset_args.eval_dataset_embedding_model,
			n_sample_documents=eval_dataset_args.n_sample_documents,
			n_evals_requested=eval_dataset_args.n_evals_requested,
			sample_documents_seed=eval_dataset_args.sample_documents_seed,
			min_document_tokens=eval_dataset_args.min_document_tokens,
			passage_match_chars=eval_dataset_args.passage_match_chars,
			ragas_version=importlib.metadata.version("ragas")
		)
		return eval_dataset_metadata

	def load_saved_eval_dataset(self, output_path: str) -> list[EvalSample]:
		"""
		Input: output_path, the dataset's output folder.
		Output: a list of EvalSample.
		What it does: loads the saved eval dataset if it is complete and unchanged, and raises ValueError if not. It
		never generates one.
		"""
		self.validate()

		eval_dataset_path = paths.get_eval_dataset_path(output_path, self.eval_dataset_args, self.ollama_args)
		print(f"Eval dataset file : {eval_dataset_path}")
		if not is_eval_dataset_complete(output_path, self.eval_dataset_args, self.ollama_args):
			raise ValueError(
				f"No complete saved eval dataset at {eval_dataset_path}. Run scripts/eval_dataset.sh first to generate it."
			)

		eval_samples = load_eval_dataset(eval_dataset_path)
		print(f"Eval dataset : {len(eval_samples)} (all from cache)")
		return eval_samples
