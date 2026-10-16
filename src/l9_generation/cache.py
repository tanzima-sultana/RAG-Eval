"""
cache.py

Saves and loads the answers of one generation model as one JSON file (see common.paths.get_generated_path):

generated.json: {prompt_hash: {"qus": ..., "answer": ...}}, where prompt_hash is the sha256 of the system prompt
                and user prompt. The same prompt gives the same answer whatever setup it came from, and a changed
                prompt template or context gives a new hash, so an old answer is never reused for it.

Prompts with an empty context are not stored, since they get I_DONT_KNOW without an LLM call.

The file is first written to a temporary file and then renamed into place, so a crash mid-write never leaves a
half-written file.
"""

import json
import os

def load_generated_cache(generated_file: str) -> dict:
	"""
	Input: generated_file, the generation cache file of one model.
	Output: the cached {prompt_hash: {"qus", "answer"}} dict, or an empty dict if the file does not exist.
	What it does: reads the file.
	"""
	if not os.path.exists(generated_file):
		return {}

	with open(generated_file, encoding="utf-8") as f:
		answer_cache = json.load(f)

	return answer_cache

def save_generated_cache(generated_file: str, answer_cache: dict) -> None:
	"""
	Input: generated_file, the generation cache file of one model; answer_cache, its {prompt_hash: {"qus", "answer"}} dict.
	Output: nothing.
	What it does: writes the dict as JSON to generated_file, creating its folder first if needed.
	"""
	os.makedirs(os.path.dirname(generated_file), exist_ok=True)

	temporary_generated_file = generated_file + ".tmp"
	with open(temporary_generated_file, "w", encoding="utf-8") as f:
		json.dump(answer_cache, f, indent=2)
	os.replace(temporary_generated_file, generated_file)
