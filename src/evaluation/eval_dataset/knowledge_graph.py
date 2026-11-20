import json
import os

from ragas.testset.graph import KnowledgeGraph
from ragas.testset.graph import NodeType
from ragas.testset.transforms import apply_transforms
from ragas.testset.transforms.base import LLMBasedExtractor
from ragas.testset.transforms.base import LLMBasedNodeFilter
from ragas.testset.transforms.engine import Parallel


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
