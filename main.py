import time

from common.constants import RAG_FRAMEWORK
from common.constants import LANGCHAIN_FRAMEWORK
from common.args_temp import parse_args
from run_rag import run_rag
from run_langchain_rag import run_langchain_rag

if __name__ == "__main__":
	print("\n ------------- RAG -------- \n")
	start = time.time()

	"""
	Input: --config and any override flags from the command line.
	Output: nothing.
	What it does: parses the args and runs the built-in RAG for "rag" or the LangChain RAG for "langchain".
	"""
	args = parse_args()
	framework = args.framework

	if framework == RAG_FRAMEWORK:
		run_rag(args)
	elif framework == LANGCHAIN_FRAMEWORK:
		run_langchain_rag(args)
	else:
		raise ValueError(
			f"framework ({framework}) is empty or not supported. "
			f"Use \"{RAG_FRAMEWORK}\" or \"{LANGCHAIN_FRAMEWORK}\"."
		)

	elapsed_time = time.time() - start
	print("\n\nElapsed time : ", elapsed_time)
