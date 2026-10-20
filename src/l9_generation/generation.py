import os

from common import constants
from common import paths
from common.args import GenerationArgs
from stages.evaluation.prompt.schema import Prompt
from l9_generation import cache
from l9_generation.schema import Generated
from l9_generation.validate import validate_generation_args
from llm.llm import get_llm
from llm.validate import validate_llm

def generate_answers(prompt_list: list[Prompt], generation_args: GenerationArgs, generated_file: str, batch_id_file: str, answer_cache: dict) -> list[Generated]:
	"""
	Input: prompt_list, the prompts; generation_args, the generation LLM and batch settings; generated_file and
	batch_id_file, the cache and batch id files; answer_cache, the cached {prompt_hash: {"qus", "answer"}} dict.
	Output: one Generated per Prompt, in the same order, so generated_list[i] belongs to prompt_list[i].
	What it does: sends every prompt not in the cache to the LLM as one batch, saves the answers to the cache,
	then answers every prompt from the cache. A prompt with an empty user prompt gets I_DONT_KNOW without an LLM call.
	"""
	# 1. Collect the prompts not in the cache, by prompt hash, so a prompt that comes more than once (two setups that
	# built the same context) is sent only once
	requests = {}
	# The question of each prompt hash, stored next to its answer so the cache file is readable
	qus_by_prompt_hash = {}
	for prompt in prompt_list:
		if prompt.user_prompt == "":
			continue
		if prompt.prompt_hash in answer_cache:
			continue
		requests[prompt.prompt_hash] = {"system_prompt": prompt.system_prompt, "user_prompt": prompt.user_prompt}
		qus_by_prompt_hash[prompt.prompt_hash] = prompt.qus
	print(f"Prompts to send to the LLM : {len(requests)}")

	# 2. Send them as one batch. The LLM is checked and built only when there is something to send, so a fully cached
	# run needs no API keys, ALLOW_PAID_API or Ollama server
	if len(requests) > 0:
		validate_llm(generation_args.llm_args)
		validate_generation_args(generation_args)
		generation_llm = get_llm(generation_args.llm_args)
		replies = generation_llm.generate_batch(requests, batch_id_file, generation_args.batch_poll_seconds, generation_args.batch_timeout_seconds)
		generation_llm.compute_cost()

		# 3. Save every reply to the cache, then remove the batch id, since its replies are now in the cache.
		# A resumed batch from an earlier run can hold prompts this run does not have; their answers are kept too,
		# with an empty question
		for prompt_hash in replies:
			if prompt_hash in qus_by_prompt_hash:
				qus = qus_by_prompt_hash[prompt_hash]
			else:
				qus = ""

			# The system prompt asks for exactly I_DONT_KNOW, but the LLM may add a full stop or a reason after it.
			# Such a reply is stored as exactly I_DONT_KNOW, so l10 scores it as a failure
			answer = replies[prompt_hash].text.strip()
			answer_lower = answer.lower().replace("’", "'")
			if answer_lower.startswith("i don't know") or answer_lower.startswith("i do not know"):
				answer = constants.I_DONT_KNOW

			answer_cache[prompt_hash] = {"qus": qus, "answer": answer}
		cache.save_generated_cache(generated_file, answer_cache)
		if os.path.exists(batch_id_file):
			os.remove(batch_id_file)

		# 4. A request that failed in the batch has no answer. Stop, so a run again sends only those
		failed_count = 0
		for prompt_hash in requests:
			if prompt_hash not in replies:
				failed_count = failed_count + 1
		if failed_count > 0:
			raise ValueError(f"{failed_count} of {len(requests)} generation requests failed. The other answers are cached; run again to send only the failed ones.")

	# 5. Answer every prompt in order: I_DONT_KNOW for an empty context, otherwise from the cache.
	# The empty ones still go into generated_list, so l10 scores them as failures
	generated_list = []
	empty_context_count = 0
	llm_dont_know_count = 0
	for prompt in prompt_list:
		if prompt.user_prompt == "":
			answer = constants.I_DONT_KNOW
			empty_context_count = empty_context_count + 1
		else:
			answer = answer_cache[prompt.prompt_hash]["answer"]
			if answer == constants.I_DONT_KNOW:
				llm_dont_know_count = llm_dont_know_count + 1

		generated = Generated(
			id=prompt.id,
			qus=prompt.qus,
			answer=answer,
			llm=generation_args.llm_args.llm,
			llm_model=generation_args.llm_args.llm_model
		)
		generated_list.append(generated)

	# 6. Print the counts
	print(f"Empty context, no LLM call : {empty_context_count} of {len(prompt_list)}")
	print(f"LLM answered I don't know : {llm_dont_know_count} of {len(prompt_list)}")

	return generated_list

def get_generated_list(output_path: str, prompt_list: list[Prompt], generation_args: GenerationArgs) -> list[Generated]:
	"""
	Input: output_path, the dataset's output folder; prompt_list, the prompts; generation_args, the generation LLM
	and batch settings.
	Output: one Generated per Prompt, in the same order.
	What it does: loads the model's answer cache, then answers every prompt from it or from one LLM batch.
	"""
	# 1. Load the cached answers of this model
	generated_file = paths.get_generated_path(output_path, generation_args.llm_args.llm_model)
	batch_id_file = paths.get_generated_batch_id_path(output_path, generation_args.llm_args.llm_model)
	answer_cache = cache.load_generated_cache(generated_file)
	print(f"Cached answers : {len(answer_cache)} in {generated_file}")

	# 2. Answer every prompt, from the cache or the LLM
	generated_list = generate_answers(prompt_list, generation_args, generated_file, batch_id_file, answer_cache)
	return generated_list
