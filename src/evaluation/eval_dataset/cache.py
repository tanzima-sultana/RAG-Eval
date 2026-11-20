import dataclasses
import hashlib
import json
import os

from common import paths
from common.args import OllamaArgs
from common.args import EvalDatasetArgs
from schema.eval_dataset_metadata import EvalDatasetMetadata
from schema.eval_sample import EvalSample


def save_eval_dataset(eval_dataset_path: str, eval_samples: list[EvalSample]) -> None:
	"""
	Input: eval_dataset_path, the eval dataset file; eval_samples, the eval samples.
	Output: nothing.
	What it does: saves eval_samples as JSON to a temporary file, then renames it to eval_dataset_path.
	"""
	os.makedirs(os.path.dirname(eval_dataset_path), exist_ok=True)

	eval_sample_dicts = []
	for eval_sample in eval_samples:
		eval_sample_dicts.append(dataclasses.asdict(eval_sample))

	temporary_file = eval_dataset_path + ".tmp"
	with open(temporary_file, "w", encoding="utf-8") as f:
		json.dump(eval_sample_dicts, f, indent=2)
	os.replace(temporary_file, eval_dataset_path)


def load_eval_dataset(eval_dataset_path: str) -> list[EvalSample] | None:
	"""
	Input: eval_dataset_path, the eval dataset file.
	Output: the saved list of EvalSample, or None if there is no file.
	What it does: loads the eval dataset from eval_dataset_path.
	"""
	if not os.path.exists(eval_dataset_path):
		return None

	with open(eval_dataset_path, encoding="utf-8") as f:
		eval_sample_dicts = json.load(f)

	eval_samples = []
	for eval_sample_dict in eval_sample_dicts:
		eval_samples.append(EvalSample(**eval_sample_dict))
	return eval_samples


def save_eval_dataset_metadata(eval_dataset_metadata_path: str, eval_dataset_metadata: EvalDatasetMetadata) -> None:
	"""
	Input: eval_dataset_metadata_path, the metadata file; eval_dataset_metadata, the metadata of one eval dataset.
	Output: nothing.
	What it does: saves the metadata as JSON to a temporary file, then renames it to eval_dataset_metadata_path.
	"""
	os.makedirs(os.path.dirname(eval_dataset_metadata_path), exist_ok=True)

	temporary_file = eval_dataset_metadata_path + ".tmp"
	with open(temporary_file, "w", encoding="utf-8") as f:
		json.dump(dataclasses.asdict(eval_dataset_metadata), f, indent=2)
	os.replace(temporary_file, eval_dataset_metadata_path)


def is_eval_dataset_complete(output_path: str, eval_dataset_args: EvalDatasetArgs, ollama_args: OllamaArgs) -> bool:
	"""
	Input: output_path, the dataset's output folder; eval_dataset_args, the eval dataset settings; ollama_args, the
	Ollama settings.
	Output: True if the eval dataset of these args was generated and is unchanged, else False.
	What it does: looks for eval_dataset_{hash}_metadata.json, checks the eval dataset and knowledge graph files
	exist, and checks the eval dataset file's SHA-256 matches the content_hash in the metadata. Prints the first
	thing missing or changed.
	"""
	# 1. The metadata of an eval dataset with these args
	eval_dataset_metadata_path = paths.get_eval_dataset_metadata_path(output_path, eval_dataset_args, ollama_args)
	if not os.path.exists(eval_dataset_metadata_path):
		print(f"No eval dataset metadata at {eval_dataset_metadata_path}")
		return False
	with open(eval_dataset_metadata_path, encoding="utf-8") as f:
		eval_dataset_metadata_dict = json.load(f)

	# 2. The eval dataset and knowledge graph files
	eval_dataset_path = paths.get_eval_dataset_path(output_path, eval_dataset_args, ollama_args)
	if not os.path.exists(eval_dataset_path):
		print(f"Missing eval dataset : {eval_dataset_path}")
		return False
	kg_path = paths.get_eval_dataset_kg_path(output_path, eval_dataset_args, ollama_args)
	if not os.path.exists(kg_path):
		print(f"Missing knowledge graph : {kg_path}")
		return False

	# 3. The eval dataset file is unchanged since it was saved
	with open(eval_dataset_path, "rb") as f:
		content_hash = hashlib.sha256(f.read()).hexdigest()
	if content_hash != eval_dataset_metadata_dict["content_hash"]:
		print(
			f"Eval dataset changed : {eval_dataset_path} has SHA-256 {content_hash}, the metadata has "
			f"{eval_dataset_metadata_dict['content_hash']}"
		)
		return False

	return True
