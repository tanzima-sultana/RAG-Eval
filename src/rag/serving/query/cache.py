import dataclasses
import json
import os
import sqlite3

from langchain_community.cache import SQLiteCache
from langchain_core.load import loads
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration

from schema.query import Query


def save_query(query_file_path: str, query: Query) -> None:
	"""
	Input: query_file_path, the file the query is saved to; query, one Query.
	Output: nothing.
	What it does: saves the query as JSON to a temporary file, then renames it to query_file_path, so a crash never
	leaves a half-written file. A file with the same name is overwritten.
	"""
	os.makedirs(os.path.dirname(query_file_path), exist_ok=True)

	temporary_file = query_file_path + ".tmp"
	with open(temporary_file, "w", encoding="utf-8") as f:
		json.dump(dataclasses.asdict(query), f, indent=2)
	os.replace(temporary_file, query_file_path)


def load_query(query_file_path: str) -> Query | None:
	"""
	Input: query_file_path, the file a query was saved to.
	Output: the saved Query, or None if there is no file.
	What it does: reads the JSON file and builds the Query from it.
	"""
	if not os.path.exists(query_file_path):
		return None

	with open(query_file_path, encoding="utf-8") as f:
		query_dict = json.load(f)
	query = Query(**query_dict)
	return query


class QueryLLMCache(SQLiteCache):
	def __init__(self, database_path: str, settings_key: str):
		"""
		Input: database_path, the SQLite file the query LLM's replies are saved in; settings_key, the query LLM settings
		LangChain leaves out of its cache key (max_tokens, the Ollama context length and the Ollama URL), as one string.
		Output: nothing.
		What it does: opens the LangChain SQLite LLM cache on the file, creating its table if needed, and stores the
		file path and settings_key. Each reply is saved under (prompt, LangChain's llm string + settings_key).
		"""
		super().__init__(database_path=database_path)
		self.database_path = database_path
		self.settings_key = settings_key

	def lookup(self, prompt: str, llm_string: str) -> list[ChatGeneration] | None:
		"""
		Input: prompt, the prompt LangChain sends to the query LLM; llm_string, LangChain's string of the LLM settings.
		Output: the saved replies of this prompt and settings, or None if none is saved.
		What it does: reads the saved rows from the SQLite file and turns each back into a ChatGeneration with
		LangChain's loads, naming the allowed classes explicitly, so no warning is raised and future LangChain versions
		still read them.
		"""
		connection = sqlite3.connect(self.database_path)
		try:
			rows = connection.execute(
				"SELECT response FROM full_llm_cache WHERE prompt = ? AND llm = ? ORDER BY idx",
				(prompt, llm_string + self.settings_key)
			).fetchall()
		finally:
			connection.close()

		if len(rows) == 0:
			return None

		generations = []
		for row in rows:
			generations.append(loads(row[0], allowed_objects=[ChatGeneration, AIMessage]))
		return generations

	def update(self, prompt: str, llm_string: str, return_val: list[ChatGeneration]) -> None:
		"""
		Input: prompt, the prompt LangChain sent to the query LLM; llm_string, LangChain's string of the LLM settings;
		return_val, the query LLM's replies.
		Output: nothing.
		What it does: turns LiteLLM's Usage object in each reply's token_usage into a plain dict, on a copy of the
		reply, since LangChain cannot save it, and saves the replies under (prompt, llm_string + settings_key).
		"""
		cleaned_generations = []
		for generation in return_val:
			message = generation.message
			response_metadata = dict(message.response_metadata)
			token_usage = response_metadata.get("token_usage")
			if token_usage is not None and hasattr(token_usage, "model_dump"):
				response_metadata["token_usage"] = token_usage.model_dump()
			cleaned_message = message.model_copy(update={"response_metadata": response_metadata})
			cleaned_generations.append(ChatGeneration(message=cleaned_message, generation_info=generation.generation_info))

		super().update(prompt, llm_string + self.settings_key, cleaned_generations)
