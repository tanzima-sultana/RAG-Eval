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
from ragas.testset.transforms import default_transforms
from ragas.utils import num_tokens_from_string

from common import paths
from common.args import OllamaArgs
from common.args import EvalDatasetArgs
from evaluation.eval_dataset.knowledge_graph import apply_transforms_dropping_failed_nodes
from evaluation.eval_dataset.knowledge_graph import check_reply_finished
from llm_provider.llm_provider import LLMProvider
from schema.document import File
from schema.eval_sample import EvalSample


def generate_eval_dataset(
	documents: dict[str, File],
	output_path: str,
	eval_dataset_args: EvalDatasetArgs,
	ollama_args: OllamaArgs
) -> list[EvalSample]:
	"""
	Input: documents, a dict of doc_id to File; output_path, the dataset's output folder; eval_dataset_args, the
	eval dataset settings; ollama_args, the Ollama settings.
	Output: a list of EvalSample. Raises ValueError if RAGAS generates none.
	What it does: samples documents long enough for RAGAS, builds or loads the RAGAS knowledge graph, generates
	eval dataset from it, prints the LLM cost, and links each eval sample to its source documents.
	"""
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
	kg_llm_provider = LLMProvider(eval_dataset_args.eval_dataset_kg_llm_args, ollama_args)
	question_llm_provider = LLMProvider(eval_dataset_args.eval_dataset_llm_args, ollama_args)
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
	kg_path = paths.get_eval_dataset_kg_path(output_path, eval_dataset_args, ollama_args)
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
