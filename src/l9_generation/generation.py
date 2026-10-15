from common import constants
from l8_prompt.schema import Prompt
from l9_generation.schema import Generated
from llm.schema import LLM
from llm.validate import validate_llm

# TODO: from l9_generation import cache, from llm.llm import get_llm

def generate_answers(prompt_list: list[Prompt], llm_args: LLM) -> list[Generated]:
	"""
	Input: prompt_list, the l8 prompts; llm_args, the generation LLM settings.
	Output: one Generated per Prompt, in the same order, so generated_list[i] belongs to prompt_list[i].
	What it does: sends each prompt to the LLM. A prompt with an empty user prompt gets I_DONT_KNOW without an LLM call.
	"""
	generated_list = []

	# 1. Build the generation LLM once
	# TODO: generation_llm = get_llm(llm_args)

	for prompt in prompt_list:
		# 2. An empty user prompt means an empty context, so skip the LLM call and answer I_DONT_KNOW
		# TODO: if prompt.user_prompt == "": answer = constants.I_DONT_KNOW

		# 3. Otherwise send the system prompt and user prompt to the LLM
		# TODO: needs a system_prompt parameter in llm/ generate()

		# 4. Keep the Generated in the same place as its Prompt, with qus=prompt.qus
		# TODO
		pass

	# 5. Print the LLM cost
	# TODO

	return generated_list

def get_generated_list(output_path: str, prompt_list: list[Prompt], llm_args: LLM) -> list[Generated]:
	"""
	Input: output_path, the dataset's output folder; prompt_list, the l8 prompts; llm_args, the generation LLM settings.
	Output: one Generated per Prompt, in the same order.
	What it does: validates llm_args, then returns the cached answers if they exist,
	otherwise generates them, saves them and returns them.
	"""
	# 1. Validate the generation LLM
	validate_llm(llm_args)

	# 2. Load cached answers, or generate and save them
	# TODO: generation_path = paths.get_generation_path(...), cache.load_cached_generated(generation_path)

	generated_list = generate_answers(prompt_list, llm_args)

	# TODO: cache.save_generated(generation_path, generated_list)
	return generated_list
