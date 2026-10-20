from common.args import GenerationArgs

def validate_generation_args(generation_args: GenerationArgs):
	"""
	Input: generation_args, the generation LLM and batch settings.
	Output: nothing. Raises ValueError if a batch setting is invalid. The LLM settings are checked by
	llm.validate.validate_llm.
	"""

	# 1. Check batch_poll_seconds
	if generation_args.batch_poll_seconds < 1:
		raise ValueError(
			f"batch_poll_seconds ({generation_args.batch_poll_seconds}) must be at least 1, "
			"so the batch status is not checked without a pause"
		)

	# 2. Check batch_timeout_seconds is not less than batch_poll_seconds
	if generation_args.batch_timeout_seconds < generation_args.batch_poll_seconds:
		raise ValueError(
			f"batch_timeout_seconds ({generation_args.batch_timeout_seconds}) must not be less than "
			f"batch_poll_seconds ({generation_args.batch_poll_seconds}), so the batch is checked at least once"
		)
