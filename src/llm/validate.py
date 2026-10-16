from common.constants import CLAUDE_LLM
from common.constants import CHATGPT_LLM
from common.constants import GEMINI_LLM
from common.constants import OLLAMA_LLM
from llm.chatgpt import validate_chatgpt
from llm.claude import validate_claude
from llm.gemini import validate_gemini
from llm.ollama import validate_ollama
from common.args import LLMArgs

def validate_llm(llm_args: LLMArgs) -> bool:
	"""
	Input: llm_args, the LLM settings of one layer.
	Output: True. Raises ValueError for the first check that fails.
	What it does: checks the settings every provider shares, then runs the checks of llm_args.llm's provider,
	which live in that provider's file.
	"""
	# 1. Check llm_model
	if llm_args.llm_model.strip() == "":
		raise ValueError(f"llm_model is empty for llm ({llm_args.llm}). Provide a model name.")

	# 2. Check the prices
	if llm_args.input_price < 0 or llm_args.output_price < 0 or llm_args.cache_write_price < 0 or llm_args.cache_read_price < 0:
		raise ValueError(
			f"Prices of {llm_args.llm_model} must be 0 or more, got input {llm_args.input_price}, output {llm_args.output_price}, "
			f"cache write {llm_args.cache_write_price} and cache read {llm_args.cache_read_price}"
		)

	# 3. Check max_tokens
	if llm_args.max_tokens <= 0:
		raise ValueError(f"max_tokens ({llm_args.max_tokens}) must be larger than 0")

	# 4. Check timeout_seconds
	if llm_args.timeout_seconds <= 0:
		raise ValueError(f"timeout_seconds ({llm_args.timeout_seconds}) must be larger than 0")

	# 5. Check cache_ttl. 5m and 1h are the two lifetimes Claude's prompt cache takes
	if llm_args.cache_ttl not in ("5m", "1h"):
		raise ValueError(f"cache_ttl ({llm_args.cache_ttl}) must be 5m or 1h")

	# 6. Check the provider
	if llm_args.llm == CLAUDE_LLM:
		return validate_claude(llm_args)
	if llm_args.llm == CHATGPT_LLM:
		return validate_chatgpt(llm_args)
	if llm_args.llm == GEMINI_LLM:
		return validate_gemini(llm_args)
	if llm_args.llm == OLLAMA_LLM:
		return validate_ollama(llm_args)

	raise ValueError(
		f"llm ({llm_args.llm}) is not supported. Use {CLAUDE_LLM}, {CHATGPT_LLM}, {GEMINI_LLM} or {OLLAMA_LLM}."
	)
