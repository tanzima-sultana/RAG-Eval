import dataclasses
import json
import os
import random
import re

import torch

from langchain_core.documents import Document as LangchainDocument
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.graph import Node
from ragas.testset.graph import NodeType
from ragas.testset.transforms import apply_transforms
from ragas.testset.transforms import default_transforms
from ragas.testset.transforms.base import LLMBasedExtractor
from ragas.testset.transforms.base import LLMBasedNodeFilter
from ragas.testset.transforms.engine import Parallel
from ragas.utils import num_tokens_from_string

from common import paths_temp
from common.args_temp import OllamaArgs
from common.args_temp import QnaArgs
from common.constants import EVAL_MIN_DOCUMENT_TOKENS
from common.constants import EVAL_PASSAGE_MATCH_CHARS
from common.constants import SAMPLE_DOCUMENTS_SEED
from llm_provider.llm_provider import LLMProvider
from schema.document import Document
from schema.qna import Qna


def save_qna(qna_path: str, qna_items: list[Qna]) -> None:
	"""
	Input: qna_path, the QnA file; qna_items, the QnA list.
	Output: nothing.
	What it does: saves qna_items as JSON to a temporary file, then renames it to qna_path.
	"""
	os.makedirs(os.path.dirname(qna_path), exist_ok=True)

	qna_dicts = []
	for qna_item in qna_items:
		qna_dicts.append(dataclasses.asdict(qna_item))

	temporary_file = qna_path + ".tmp"
	with open(temporary_file, "w", encoding="utf-8") as f:
		json.dump(qna_dicts, f, indent=2)
	os.replace(temporary_file, qna_path)


def load_qna(qna_path: str) -> list[Qna] | None:
	"""
	Input: qna_path, the QnA file.
	Output: the saved list of Qna, or None if there is no file.
	What it does: loads the QnA from qna_path.
	"""
	if not os.path.exists(qna_path):
		return None

	with open(qna_path, encoding="utf-8") as f:
		qna_dicts = json.load(f)

	qna_items = []
	for qna_dict in qna_dicts:
		qna_items.append(Qna(**qna_dict))
	return qna_items


class ReplyDidNotFinishError(Exception):
	"""
	Raised when an LLM reply stopped for a reason other than finishing normally, for example max_tokens or refusal.
	The message holds the stop reason and the reply's output tokens.
	"""


def check_reply_finished(response) -> bool:
	"""
	Input: response, the LangChain LLMResult of one RAGAS LLM call.
	Output: True if every reply finished normally. Raises ReplyDidNotFinishError otherwise.
	What it does: reads each reply's finish_reason or stop_reason, and raises with the stop reason and output tokens
	instead of RAGAS's message that always blames max_tokens.
	"""
	finished_reasons = ["stop", "STOP", "MAX_TOKENS", "eos_token", "end_turn"]
	for generation_list in response.generations:
		for generation in generation_list:
			stop_reason = None
			output_tokens = None
			if generation.generation_info is not None:
				stop_reason = generation.generation_info.get("finish_reason")
			message = getattr(generation, "message", None)
			if stop_reason is None and message is not None:
				stop_reason = message.response_metadata.get("finish_reason")
				if stop_reason is None:
					stop_reason = message.response_metadata.get("stop_reason")
			if message is not None and message.usage_metadata is not None:
				output_tokens = message.usage_metadata.get("output_tokens")
			if stop_reason is not None and stop_reason not in finished_reasons:
				raise ReplyDidNotFinishError(f"stop reason {stop_reason}, output tokens {output_tokens}")
	return True


def make_failure_safe_extract(extractor: LLMBasedExtractor, failures: list[tuple]):
	"""
	Input: extractor, one RAGAS LLM extractor; failures, the list that collects (node, step name, error).
	Output: an async extract function to use in place of the extractor's own.
	What it does: calls the extractor's own extract for a node. If the reply did not finish, it records the node and
	returns an empty value for it, so the step goes on; the node is dropped after the step.
	"""
	original_extract = extractor.extract

	async def failure_safe_extract(node):
		try:
			return await original_extract(node)
		except ReplyDidNotFinishError as error:
			failures.append((node, type(extractor).__name__, error))
			# Not None: RAGAS adds a property only while it is None, so a second copy of the node would add it again
			return extractor.property_name, ""

	return failure_safe_extract


def make_failure_safe_custom_filter(node_filter: LLMBasedNodeFilter, failures: list[tuple]):
	"""
	Input: node_filter, one RAGAS LLM node filter; failures, the list that collects (node, step name, error).
	Output: an async custom_filter function to use in place of the filter's own.
	What it does: calls the filter's own custom_filter for a node. If the reply did not finish, it records the node and
	keeps it for now; the node is dropped after the step.
	"""
	original_custom_filter = node_filter.custom_filter

	async def failure_safe_custom_filter(node, knowledge_graph):
		try:
			return await original_custom_filter(node, knowledge_graph)
		except ReplyDidNotFinishError as error:
			failures.append((node, type(node_filter).__name__, error))
			return False

	return failure_safe_custom_filter


def apply_transforms_dropping_failed_nodes(knowledge_graph: KnowledgeGraph, transforms: list, kg_path: str) -> KnowledgeGraph:
	"""
	Input: knowledge_graph, the graph of sampled document nodes; transforms, the RAGAS default transforms; kg_path,
	the knowledge graph file.
	Output: the transformed knowledge graph.
	What it does: applies the transforms one step at a time. A node whose LLM reply did not finish is dropped after the
	step, with a document's chunks. The graph is saved after every step to {kg_path}.partial, with the steps done in
	{kg_path}.partial_steps.json, so a stopped run goes on from the last saved step.
	"""
	partial_kg_path = kg_path + ".partial"
	partial_steps_path = kg_path + ".partial_steps.json"

	# 1. Go on from the last saved step, if a run stopped before
	steps_done = 0
	if os.path.exists(partial_kg_path) and os.path.exists(partial_steps_path):
		knowledge_graph = KnowledgeGraph.load(partial_kg_path)
		with open(partial_steps_path, encoding="utf-8") as f:
			steps_done = json.load(f)
		print(f"Knowledge graph : going on from step {steps_done + 1} of {len(transforms)}, saved in {partial_kg_path}")

	dropped_node_count = 0
	for step_index in range(steps_done, len(transforms)):
		step = transforms[step_index]

		# 2. Make each LLM call of the step record a reply that did not finish, instead of stopping the run
		if isinstance(step, Parallel):
			step_transforms = step.transformations
		else:
			step_transforms = [step]

		failures = []
		for transform in step_transforms:
			if isinstance(transform, LLMBasedExtractor):
				transform.extract = make_failure_safe_extract(transform, failures)
			elif isinstance(transform, LLMBasedNodeFilter):
				transform.custom_filter = make_failure_safe_custom_filter(transform, failures)

		apply_transforms(knowledge_graph, step)

		# 3. Drop the failed nodes, and a failed document's chunks
		for node, step_name, error in failures:
			nodes_to_drop = [node]
			if node.type == NodeType.DOCUMENT:
				for relationship in knowledge_graph.relationships:
					if relationship.source == node and relationship.type == "child":
						nodes_to_drop.append(relationship.target)
			# RAGAS's HeadlineSplitter adds a document it cannot split to the graph a second time, so every copy is removed
			for node_to_drop in nodes_to_drop:
				was_dropped = False
				while node_to_drop in knowledge_graph.nodes:
					knowledge_graph.remove_node(node_to_drop)
					was_dropped = True
				if was_dropped:
					dropped_node_count += 1
			print(f"Dropped {node.type.value} node {node.id} after {step_name}: {error}")

		# 4. Save the graph and the steps done
		knowledge_graph.save(partial_kg_path + ".tmp")
		os.replace(partial_kg_path + ".tmp", partial_kg_path)
		with open(partial_steps_path + ".tmp", "w", encoding="utf-8") as f:
			json.dump(step_index + 1, f)
		os.replace(partial_steps_path + ".tmp", partial_steps_path)

	print(f"Knowledge graph nodes dropped because a reply did not finish : {dropped_node_count}")

	return knowledge_graph


class QnaSet:
	def __init__(self, qna_args: QnaArgs, ollama_args: OllamaArgs):
		"""
		Input: qna_args, the QnA settings; ollama_args, the Ollama settings, used only if a QnA model is an Ollama model.
		Output: nothing.
		What it does: stores both settings.
		"""
		self.qna_args = qna_args
		self.ollama_args = ollama_args

	def validate(self) -> None:
		"""
		Input: none.
		Output: nothing.
		What it does: raises ValueError if a QnA setting is not valid. The two LLMs are validated by LLMProvider.
		"""
		qna_args = self.qna_args

		if qna_args.qna_embedding_model.strip() == "":
			raise ValueError("qna_embedding_model is empty. Use a SentenceTransformer model name, e.g. all-MiniLM-L6-v2.")

		if qna_args.n_sample_documents <= 0:
			raise ValueError(f"n_sample_documents ({qna_args.n_sample_documents}) must be greater than 0.")

		if qna_args.n_sample_evals <= 0:
			raise ValueError(f"n_sample_evals ({qna_args.n_sample_evals}) must be greater than 0.")

	def compute_qna(self, documents: dict[str, Document], output_path: str) -> list[Qna]:
		"""
		Input: documents, a dict of doc_id to Document; output_path, the dataset's output folder.
		Output: a list of Qna.
		What it does: loads the saved QnA. Only if there is none, it generates the QnA with RAGAS and saves it.
		"""
		# 1. Validate the settings
		self.validate()

		# 2. Load the saved QnA. No LLM is built, so a cached run needs no API key or ALLOW_PAID_API
		qna_path = paths_temp.get_qna_path(output_path, self.qna_args)
		print(f"QnA file : {qna_path}")
		qna_items = load_qna(qna_path)
		if qna_items is not None:
			print(f"QnA : {len(qna_items)} (all from cache)")
			return qna_items

		# 3. Generate and save the QnA
		print(f"No saved QnA, generating {self.qna_args.n_sample_evals} items with RAGAS")
		qna_items = self.generate_qna(documents, output_path)
		save_qna(qna_path, qna_items)
		return qna_items

	def generate_qna(self, documents: dict[str, Document], output_path: str) -> list[Qna]:
		"""
		Input: documents, a dict of doc_id to Document; output_path, the dataset's output folder.
		Output: a list of Qna. Raises ValueError if RAGAS generates none.
		What it does: samples documents long enough for RAGAS, builds or loads the RAGAS knowledge graph, generates
		QnA from it, prints the LLM cost, and links each QnA to its source documents.
		"""
		qna_args = self.qna_args

		# 1. Keep documents long enough for RAGAS
		eligible_doc_ids = []
		for doc_id in documents:
			document_tokens = num_tokens_from_string(documents[doc_id].text)
			if document_tokens > EVAL_MIN_DOCUMENT_TOKENS:
				eligible_doc_ids.append(doc_id)
		eligible_doc_ids.sort()
		print(f"Documents with more than {EVAL_MIN_DOCUMENT_TOKENS} tokens : {len(eligible_doc_ids)} of {len(documents)}")

		if len(eligible_doc_ids) < qna_args.n_sample_documents:
			raise ValueError(
				f"only {len(eligible_doc_ids)} of {len(documents)} documents have more than "
				f"{EVAL_MIN_DOCUMENT_TOKENS} tokens; n_sample_documents ({qna_args.n_sample_documents}) is too large"
			)

		# 2. Sample documents
		random_generator = random.Random(SAMPLE_DOCUMENTS_SEED)
		sampled_doc_ids = random_generator.sample(eligible_doc_ids, qna_args.n_sample_documents)

		sampled_documents = {}
		for doc_id in sampled_doc_ids:
			sampled_documents[doc_id] = documents[doc_id]
		print(f"Documents sampled for RAGAS : {len(sampled_documents)} of {len(eligible_doc_ids)}")

		# 3. Make LangChain documents from the sampled documents
		langchain_documents = []
		for doc_id in sampled_documents:
			document = sampled_documents[doc_id]
			langchain_document = LangchainDocument(
				page_content=document.text,
				metadata={"doc_id": doc_id, "source": document.metadata.source}
			)
			langchain_documents.append(langchain_document)

		# 4. Get the LLMs. One model builds the knowledge graph, another writes the questions.
		# LLMProvider validates each, so a paid model needs ALLOW_PAID_API=1.
		# RAGAS sets temperature 0.01 by default, which the Claude 5.5 models reject, so the model's own temperature is
		# kept. check_reply_finished raises with the stop reason when a reply does not finish normally
		kg_llm_provider = LLMProvider(qna_args.qna_kg_llm_args, self.ollama_args)
		question_llm_provider = LLMProvider(qna_args.qna_llm_args, self.ollama_args)
		ragas_kg_llm = LangchainLLMWrapper(kg_llm_provider.chat_model, bypass_temperature=True, is_finished_parser=check_reply_finished)
		ragas_question_llm = LangchainLLMWrapper(question_llm_provider.chat_model, bypass_temperature=True, is_finished_parser=check_reply_finished)

		device = "cuda" if torch.cuda.is_available() else "cpu"
		huggingface_embeddings = HuggingFaceEmbeddings(
			model_name=qna_args.qna_embedding_model,
			model_kwargs={"device": device},
			encode_kwargs={"normalize_embeddings": True}
		)
		ragas_embeddings = LangchainEmbeddingsWrapper(huggingface_embeddings)

		# 5. Build or load the knowledge graph. Saved after building, so a failed or repeated run does not pay again
		kg_path = paths_temp.get_qna_kg_path(output_path, qna_args)
		if os.path.exists(kg_path):
			knowledge_graph = KnowledgeGraph.load(kg_path)
			print(f"Knowledge graph loaded from cache : {kg_path}")
		else:
			knowledge_graph = KnowledgeGraph()
			for langchain_document in langchain_documents:
				document_node = Node(
					type=NodeType.DOCUMENT,
					properties={"page_content": langchain_document.page_content, "document_metadata": langchain_document.metadata}
				)
				knowledge_graph.nodes.append(document_node)
			transforms = default_transforms(
				documents=langchain_documents,
				llm=ragas_kg_llm,
				embedding_model=ragas_embeddings
			)
			os.makedirs(os.path.dirname(kg_path), exist_ok=True)
			knowledge_graph = apply_transforms_dropping_failed_nodes(knowledge_graph, transforms, kg_path)
			knowledge_graph.save(kg_path + ".tmp")
			os.replace(kg_path + ".tmp", kg_path)
			# The step files are only needed while the graph is not complete
			if os.path.exists(kg_path + ".partial"):
				os.remove(kg_path + ".partial")
			if os.path.exists(kg_path + ".partial_steps.json"):
				os.remove(kg_path + ".partial_steps.json")
			print(f"Knowledge graph saved : {kg_path}")

		# 6. Generate the test set
		testset_generator = TestsetGenerator(
			llm=ragas_question_llm,
			embedding_model=ragas_embeddings,
			knowledge_graph=knowledge_graph
		)
		testset = testset_generator.generate(testset_size=qna_args.n_sample_evals)
		kg_cost = kg_llm_provider.compute_cost()
		question_cost = question_llm_provider.compute_cost()
		print(f"QnA total cost : ${kg_cost.total_cost + question_cost.total_cost:.4f}")

		if len(testset.samples) == 0:
			raise ValueError(
				f"RAGAS generated 0 QnA from {len(sampled_documents)} sampled documents, so nothing is saved. "
				"The sampled documents may be too short for RAGAS; try a larger n_sample_documents."
			)

		# 7. Convert the test set to Qna, linking each item to its source documents.
		# RAGAS may rejoin a long chunk's words with single spaces, so whitespace is collapsed on both sides before matching
		collapsed_document_texts = {}
		for doc_id in sampled_documents:
			collapsed_document_texts[doc_id] = " ".join(sampled_documents[doc_id].text.split())

		# The model string is "{provider}/{model}", e.g. anthropic/claude-sonnet-5-5
		qna_llm, qna_llm_model = qna_args.qna_llm_args.llm_model.split("/", 1)

		qna_items = []
		sample_number = 0
		for testset_sample in testset.samples:
			ragas_sample = testset_sample.eval_sample

			context = []
			doc_ids = []
			doc_paths = []
			if ragas_sample.reference_contexts is not None:
				for reference_context in ragas_sample.reference_contexts:
					context.append(reference_context)

					# Multi-hop contexts start with a "<1-hop>\n\n" marker added by RAGAS, which is not in the document text
					passage_text = reference_context
					hop_marker_match = re.match(r"<\d+-hop>\n\n", reference_context)
					if hop_marker_match is not None:
						passage_text = reference_context[hop_marker_match.end():]
					collapsed_passage_text = " ".join(passage_text.split())
					# RAGAS may stitch non-adjacent sections together, so only the start of the passage is matched
					passage_start = collapsed_passage_text[:EVAL_PASSAGE_MATCH_CHARS]

					for doc_id in sampled_documents:
						if doc_id in doc_ids:
							continue
						if passage_start in collapsed_document_texts[doc_id]:
							doc_ids.append(doc_id)
							doc_paths.append(sampled_documents[doc_id].metadata.source)

			qna_item = Qna(
				id=f"eval_qna_{sample_number}",
				qus=ragas_sample.user_input,
				ans=ragas_sample.reference,
				context=context,
				qus_type=testset_sample.synthesizer_name,
				doc_ids=doc_ids,
				doc_paths=doc_paths,
				llm=qna_llm,
				llm_model=qna_llm_model
			)
			qna_items.append(qna_item)
			sample_number = sample_number + 1

		if len(qna_items) < qna_args.n_sample_evals:
			print(f"Warning : Generated {len(qna_items)} of {qna_args.n_sample_evals} QnA. Saving the ones generated.")

		return qna_items
