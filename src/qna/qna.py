
import dataclasses
import json
import os
import random
import re

import torch
from langchain_core.documents import Document as LangchainDocument
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.graph import Node
from ragas.testset.graph import NodeType
from ragas.testset.transforms import apply_transforms
from ragas.testset.transforms import default_transforms
from ragas.utils import num_tokens_from_string

from common import paths
from common.args import EvalQnAArgs
from common.constants import SAMPLE_DOCUMENTS_SEED
from common.constants import EVAL_MIN_DOCUMENT_TOKENS
from common.constants import EVAL_PASSAGE_MATCH_CHARS
from l1_ingestion.schema import Document
from llm.llm import get_llm_generator
from qna.schema import EvalQnA
from qna import validate

def load_cached_eval_qna(output_path: str, eval_qna_args: EvalQnAArgs) -> list[EvalQnA] | None:
	"""
	Input: output_path, the dataset's output folder; eval_qna_args, the eval QnA settings.
	Output: the cached list of EvalQnA, or None if there is no cache file.
	What it does: reads the eval QnA cache file from paths.get_eval_qna_path.
	"""
	cache_file = paths.get_eval_qna_path(output_path, eval_qna_args)
	if not os.path.exists(cache_file):
		return None

	with open(cache_file, encoding="utf-8") as f:
		eval_qna_dicts = json.load(f)

	eval_qna = []
	for eval_qna_dict in eval_qna_dicts:
		eval_qna.append(EvalQnA(**eval_qna_dict))

	return eval_qna

def save_eval_qna(output_path: str, eval_qna_args: EvalQnAArgs, eval_qna: list[EvalQnA]) -> None:
	"""
	Input: output_path, the dataset's output folder; eval_qna_args, the eval QnA settings; eval_qna, the EvalQnA list.
	Output: nothing.
	What it does: writes eval_qna as JSON to the cache file from paths.get_eval_qna_path.
	"""
	os.makedirs(output_path, exist_ok=True)

	eval_qna_dicts = []
	for qna_item in eval_qna:
		eval_qna_dicts.append(dataclasses.asdict(qna_item))

	cache_file = paths.get_eval_qna_path(output_path, eval_qna_args)
	temporary_file = cache_file + ".tmp"
	with open(temporary_file, "w", encoding="utf-8") as f:
		json.dump(eval_qna_dicts, f, indent=2)
	os.replace(temporary_file, cache_file)

def generate_eval_qna(output_path: str, documents: dict[str, Document], eval_qna_args: EvalQnAArgs) -> list[EvalQnA]:
	"""
	Input: output_path, the dataset's output folder; documents, a dict of doc_id to l1 Document; eval_qna_args, the eval QnA settings.
	Output: a list of EvalQnA. Raises ValueError if RAGAS generates none.
	What it does: samples documents long enough for RAGAS, builds or loads the RAGAS knowledge graph,
	generates QnA from it with the LLM generator, prints the LLM cost, and links each QnA to its source documents.
	"""
	# 1. Keep documents long enough for RAGAS
	eligible_doc_ids = []
	for doc_id in documents:
		document_tokens = num_tokens_from_string(documents[doc_id].text)
		if document_tokens > EVAL_MIN_DOCUMENT_TOKENS:
			eligible_doc_ids.append(doc_id)
	eligible_doc_ids.sort()
	print(f"Documents with more than {EVAL_MIN_DOCUMENT_TOKENS} tokens : {len(eligible_doc_ids)} of {len(documents)}")

	if len(eligible_doc_ids) < eval_qna_args.n_sample_documents:
		raise ValueError(
			f"only {len(eligible_doc_ids)} of {len(documents)} documents have more than "
			f"{EVAL_MIN_DOCUMENT_TOKENS} tokens; n_sample_documents ({eval_qna_args.n_sample_documents}) is too large"
		)

	# 2. Sample documents
	random_generator = random.Random(SAMPLE_DOCUMENTS_SEED)
	sampled_doc_ids = random_generator.sample(eligible_doc_ids, eval_qna_args.n_sample_documents)

	sampled_documents = {}
	for doc_id in sampled_doc_ids:
		sampled_documents[doc_id] = documents[doc_id]
	print(f"Documents sampled for RAGAS : {len(sampled_documents)} of {len(eligible_doc_ids)}")

	# 3. Make langchain documents from sampled documents
	langchain_documents = []
	for doc_id in sampled_documents:
		document = sampled_documents[doc_id]
		langchain_document = LangchainDocument(
			page_content=document.text,
			metadata={"doc_id": doc_id, "source": document.metadata.source}
		)
		langchain_documents.append(langchain_document)

	# 4. Get LLM generator
	llm_generator = get_llm_generator(eval_qna_args.eval_qna_llm)

	device = "cuda" if torch.cuda.is_available() else "cpu"
	huggingface_embeddings = HuggingFaceEmbeddings(
		model_name=eval_qna_args.eval_qna_model,
		model_kwargs={"device": device},
		encode_kwargs={"normalize_embeddings": True}
	)
	eval_embeddings = LangchainEmbeddingsWrapper(huggingface_embeddings)

	# 5. Build or load the knowledge graph
	# Saved after building, so a failed or repeated run does not pay to build it again
	kg_path = paths.get_eval_kg_path(output_path, eval_qna_args)
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
			llm=llm_generator.transforms_llm,
			embedding_model=eval_embeddings
		)
		apply_transforms(knowledge_graph, transforms)
		os.makedirs(output_path, exist_ok=True)
		knowledge_graph.save(kg_path + ".tmp")
		os.replace(kg_path + ".tmp", kg_path)
		print(f"Knowledge graph saved : {kg_path}")

	# 6. Generate test sets
	testset_generator = TestsetGenerator(
		llm=llm_generator.question_llm,
		embedding_model=eval_embeddings,
		knowledge_graph=knowledge_graph
	)
	testset = testset_generator.generate(testset_size=eval_qna_args.n_sample_evals)
	llm_generator.compute_cost()

	if len(testset.samples) == 0:
		raise ValueError(
			f"RAGAS generated 0 eval QnA from {len(sampled_documents)} sampled documents, so nothing is saved. "
			"The sampled documents may be too short for RAGAS; try a larger n_sample_documents."
		)

	# 7. Convert test set to schema EvalQnA
	# RAGAS may rejoin a long chunk's words with single spaces, so whitespace is collapsed on both sides before matching
	collapsed_document_texts = {}
	for doc_id in sampled_documents:
		collapsed_document_texts[doc_id] = " ".join(sampled_documents[doc_id].text.split())

	eval_qna = []
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

		qna_item = EvalQnA(
			id=f"eval_qna_{sample_number}",
			qus=ragas_sample.user_input,
			ans=ragas_sample.reference,
			context=context,
			qus_type=testset_sample.synthesizer_name,
			doc_ids=doc_ids,
			doc_paths=doc_paths
		)
		eval_qna.append(qna_item)
		sample_number = sample_number + 1

	if len(eval_qna) < eval_qna_args.n_sample_evals:
		print(f"Warning : Generated {len(eval_qna)} of {eval_qna_args.n_sample_evals} eval QnA. Saving the ones generated.")

	return eval_qna

def get_eval_qna(output_path: str, documents: dict[str, Document], eval_qna_args: EvalQnAArgs) -> list[EvalQnA]:
	"""
	Input: output_path, the dataset's output folder; documents, a dict of doc_id to l1 Document; eval_qna_args, the eval QnA settings.
	Output: a list of EvalQnA.
	What it does: validates eval_qna_args, then returns the cached eval QnA if it exists,
	otherwise generates it, saves it and returns it.
	"""
	# 1. Validate eval QnA args
	validate.validate_eval_qna_args(eval_qna_args)

	# 2. Load cached eval QnA, or generate and save it
	eval_qna = load_cached_eval_qna(output_path, eval_qna_args)
	if eval_qna is not None:
		print(f"Eval QnA cache found : {len(eval_qna)} items")
		return eval_qna

	print(f"No eval QnA cache, generating {eval_qna_args.n_sample_evals} items with RAGAS")
	eval_qna = generate_eval_qna(output_path, documents, eval_qna_args)
	save_eval_qna(output_path, eval_qna_args, eval_qna)
	return eval_qna