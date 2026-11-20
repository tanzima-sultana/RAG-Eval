import dataclasses
import datetime
import hashlib
import importlib.metadata
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

from common import paths
from common.args import OllamaArgs
from common.args import EvalDatasetArgs
from llm_provider.llm_provider import LLMProvider
from schema.document import Document
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

	def is_eval_dataset_complete(self, output_path: str) -> bool:
		"""
		Input: output_path, the dataset's output folder.
		Output: True if the eval dataset of these args was generated and is unchanged, else False.
		What it does: looks for eval_dataset_{hash}_metadata.json, checks the eval dataset and knowledge graph files
		exist, and checks the eval dataset file's SHA-256 matches the content_hash in the metadata. Prints the first
		thing missing or changed.
		"""
		# 1. The metadata of an eval dataset with these args
		eval_dataset_metadata_path = paths.get_eval_dataset_metadata_path(output_path, self.eval_dataset_args, self.ollama_args)
		if not os.path.exists(eval_dataset_metadata_path):
			print(f"No eval dataset metadata at {eval_dataset_metadata_path}")
			return False
		with open(eval_dataset_metadata_path, encoding="utf-8") as f:
			eval_dataset_metadata_dict = json.load(f)

		# 2. The eval dataset and knowledge graph files
		eval_dataset_path = paths.get_eval_dataset_path(output_path, self.eval_dataset_args, self.ollama_args)
		if not os.path.exists(eval_dataset_path):
			print(f"Missing eval dataset : {eval_dataset_path}")
			return False
		kg_path = paths.get_eval_dataset_kg_path(output_path, self.eval_dataset_args, self.ollama_args)
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
		if self.is_eval_dataset_complete(output_path):
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
			documents[doc_id] = Document.model_validate(document_dict)
		print(f"Documents : {len(documents)} of {len(doc_ids)} in {documents_path}")

		# 4. Generate and save the eval dataset
		print(f"No saved eval dataset, generating {self.eval_dataset_args.n_evals_requested} items with RAGAS")
		eval_samples = self.generate_eval_dataset(documents, output_path)
		save_eval_dataset(eval_dataset_path, eval_samples)

		# 5. Save its metadata next to it: what it is and how it was made
		eval_dataset_metadata = self.make_eval_dataset_metadata(documents, output_path, eval_samples)
		eval_dataset_metadata_path = paths.get_eval_dataset_metadata_path(output_path, self.eval_dataset_args, self.ollama_args)
		save_eval_dataset_metadata(eval_dataset_metadata_path, eval_dataset_metadata)
		print(f"Eval dataset metadata saved : {eval_dataset_metadata_path}")
		return eval_samples

	def make_eval_dataset_metadata(self, documents: dict[str, Document], output_path: str, eval_samples: list[EvalSample]) -> EvalDatasetMetadata:
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
		if not self.is_eval_dataset_complete(output_path):
			raise ValueError(
				f"No complete saved eval dataset at {eval_dataset_path}. Run scripts/eval_dataset.sh first to generate it."
			)

		eval_samples = load_eval_dataset(eval_dataset_path)
		print(f"Eval dataset : {len(eval_samples)} (all from cache)")
		return eval_samples

	def generate_eval_dataset(self, documents: dict[str, Document], output_path: str) -> list[EvalSample]:
		"""
		Input: documents, a dict of doc_id to Document; output_path, the dataset's output folder.
		Output: a list of EvalSample. Raises ValueError if RAGAS generates none.
		What it does: samples documents long enough for RAGAS, builds or loads the RAGAS knowledge graph, generates
		eval dataset from it, prints the LLM cost, and links each eval sample to its source documents.
		"""
		eval_dataset_args = self.eval_dataset_args

		# 1. Keep documents long enough for RAGAS
		eligible_doc_ids = []
		for doc_id in documents:
			document_tokens = num_tokens_from_string(documents[doc_id].text)
			if document_tokens > eval_dataset_args.min_document_tokens:
				eligible_doc_ids.append(doc_id)
		eligible_doc_ids.sort()
		print(f"Documents with more than {eval_dataset_args.min_document_tokens} tokens : {len(eligible_doc_ids)} of {len(documents)}")

		if len(eligible_doc_ids) < eval_dataset_args.n_sample_documents:
			raise ValueError(
				f"only {len(eligible_doc_ids)} of {len(documents)} documents have more than "
				f"{eval_dataset_args.min_document_tokens} tokens; n_sample_documents ({eval_dataset_args.n_sample_documents}) is too large"
			)

		# 2. Sample documents
		random_generator = random.Random(eval_dataset_args.sample_documents_seed)
		sampled_doc_ids = random_generator.sample(eligible_doc_ids, eval_dataset_args.n_sample_documents)

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
		kg_llm_provider = LLMProvider(eval_dataset_args.eval_dataset_kg_llm_args, self.ollama_args)
		question_llm_provider = LLMProvider(eval_dataset_args.eval_dataset_llm_args, self.ollama_args)
		ragas_kg_llm = LangchainLLMWrapper(kg_llm_provider.chat_model, bypass_temperature=True, is_finished_parser=check_reply_finished)
		ragas_question_llm = LangchainLLMWrapper(question_llm_provider.chat_model, bypass_temperature=True, is_finished_parser=check_reply_finished)

		device = "cuda" if torch.cuda.is_available() else "cpu"
		huggingface_embeddings = HuggingFaceEmbeddings(
			model_name=eval_dataset_args.eval_dataset_embedding_model,
			model_kwargs={"device": device},
			encode_kwargs={"normalize_embeddings": True}
		)
		ragas_embeddings = LangchainEmbeddingsWrapper(huggingface_embeddings)

		# 5. Build or load the knowledge graph. Saved after building, so a failed or repeated run does not pay again
		kg_path = paths.get_eval_dataset_kg_path(output_path, eval_dataset_args, self.ollama_args)
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
		testset = testset_generator.generate(testset_size=eval_dataset_args.n_evals_requested)
		kg_cost = kg_llm_provider.compute_cost()
		question_cost = question_llm_provider.compute_cost()
		print(f"Eval dataset total cost : ${kg_cost.total_cost + question_cost.total_cost:.4f}")

		if len(testset.samples) == 0:
			raise ValueError(
				f"RAGAS generated 0 eval samples from {len(sampled_documents)} sampled documents, so nothing is saved. "
				"The sampled documents may be too short for RAGAS; try a larger n_sample_documents."
			)

		# 7. Convert the test set to EvalSample, linking each item to its source documents.
		# RAGAS may rejoin a long chunk's words with single spaces, so whitespace is collapsed on both sides before matching
		collapsed_document_texts = {}
		for doc_id in sampled_documents:
			collapsed_document_texts[doc_id] = " ".join(sampled_documents[doc_id].text.split())

		# The model string is "{provider}/{model}", e.g. anthropic/claude-sonnet-5-5
		eval_dataset_llm, eval_dataset_llm_model = eval_dataset_args.eval_dataset_llm_args.llm_model.split("/", 1)

		eval_samples = []
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
					passage_start = collapsed_passage_text[:eval_dataset_args.passage_match_chars]

					for doc_id in sampled_documents:
						if doc_id in doc_ids:
							continue
						if passage_start in collapsed_document_texts[doc_id]:
							doc_ids.append(doc_id)
							doc_paths.append(sampled_documents[doc_id].metadata.source)

			eval_sample = EvalSample(
				id=f"eval_qna_{sample_number}",
				qus=ragas_sample.user_input,
				ans=ragas_sample.reference,
				context=context,
				qus_type=testset_sample.synthesizer_name,
				doc_ids=doc_ids,
				doc_paths=doc_paths,
				llm=eval_dataset_llm,
				llm_model=eval_dataset_llm_model
			)
			eval_samples.append(eval_sample)
			sample_number = sample_number + 1

		if len(eval_samples) < eval_dataset_args.n_evals_requested:
			print(f"Warning : Generated {len(eval_samples)} of {eval_dataset_args.n_evals_requested} eval samples. Saving the ones generated.")

		return eval_samples
