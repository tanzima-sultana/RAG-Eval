"""
report.py

Writes the summary tables of one run as Markdown and joins every section into the run's report.
"""

from analysis.schema import AnalysisRun
from analysis.schema import AnswerSummary
from analysis.schema import RetrievalSummary

def get_setup_cells(setup_key: tuple) -> str:
	"""
	Input: setup_key, a setup tuple (chunk type, retrieval type, index type, query type, reranking, k).
	Output: the setup as the first cells of a Markdown table row, e.g. "| sentence | hybrid | faiss_flatip+bm25 | original | no | 20 |".
	What it does: writes each setup field as one cell, with reranking as yes or no.
	"""
	chunk_type, retrieval_type, index_type, query_type, reranking, k = setup_key
	if reranking:
		rerank_text = "yes"
	else:
		rerank_text = "no"
	setup_cells = f"| {chunk_type} | {retrieval_type} | {index_type} | {query_type} | {rerank_text} | {k} |"
	return setup_cells

def get_retrieval_summary_table(retrieval_summaries: list[RetrievalSummary], title: str) -> str:
	"""
	Input: retrieval_summaries, one RetrievalSummary per setup; title, the table's heading.
	Output: the retrieval summary as a Markdown table under the heading.
	What it does: writes one row per setup with its chunk-level and doc-level means.
	"""
	lines = []
	lines.append(f"## {title}")
	lines.append("")
	lines.append("Means over the eval questions. c_ is chunk level: a chunk counts if it overlaps the passage the question was written from. d_ is doc level: a chunk counts if it comes from the question's document.")
	lines.append("")
	lines.append("| Chunk | Retrieval | Index | Query | Rerank | k | n | c_hit | c_prec | c_recall | c_mrr | c_ndcg | d_hit | d_prec | d_recall | d_mrr | d_ndcg |")
	lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
	for retrieval_summary in retrieval_summaries:
		setup = retrieval_summary.setup
		setup_key = (setup.chunk_type, setup.retrieval_type, setup.index_type, setup.query_type, setup.reranking, setup.k)
		row = (
			get_setup_cells(setup_key) + f" {retrieval_summary.n} |"
			f" {retrieval_summary.chunk_hit:.3f} | {retrieval_summary.chunk_precision:.3f} | {retrieval_summary.chunk_recall:.3f} |"
			f" {retrieval_summary.chunk_mrr:.3f} | {retrieval_summary.chunk_ndcg:.3f} |"
			f" {retrieval_summary.doc_hit:.3f} | {retrieval_summary.doc_precision:.3f} | {retrieval_summary.doc_recall:.3f} |"
			f" {retrieval_summary.doc_mrr:.3f} | {retrieval_summary.doc_ndcg:.3f} |"
		)
		lines.append(row)

	retrieval_summary_table = "\n".join(lines)
	return retrieval_summary_table

def get_answer_summary_table(answer_summaries: list[AnswerSummary], judge: str) -> str:
	"""
	Input: answer_summaries, one AnswerSummary per kept setup; judge, the framework and judge model.
	Output: the answer summary as a Markdown table, titled with the judge.
	What it does: writes one row per setup with its I_DONT_KNOW share and mean scores.
	"""
	lines = []
	lines.append(f"## Answer evaluation, judged by {judge}")
	lines.append("")
	lines.append("Means over the eval questions. idk is the share of I don't know answers, which count as 0 on the three scores.")
	lines.append("")
	lines.append("| Chunk | Retrieval | Index | Query | Rerank | k | n | idk | correct | faithful | relevant |")
	lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
	for answer_summary in answer_summaries:
		setup = answer_summary.setup
		setup_key = (setup.chunk_type, setup.retrieval_type, setup.index_type, setup.query_type, setup.reranking, setup.k)
		row = (
			get_setup_cells(setup_key) + f" {answer_summary.n} |"
			f" {answer_summary.i_dont_know:.3f} | {answer_summary.correctness:.3f} |"
			f" {answer_summary.faithfulness:.3f} | {answer_summary.relevance:.3f} |"
		)
		lines.append(row)

	answer_summary_table = "\n".join(lines)
	return answer_summary_table

def get_report(analysis_run: AnalysisRun, sections: list[str]) -> str:
	"""
	Input: analysis_run, the current run; sections, the report sections as Markdown text, in report order.
	Output: the full report as Markdown text.
	What it does: writes a title naming the run, its time, judge and settings hash, then each section.
	"""
	lines = []
	lines.append(f"# Analysis report: run {analysis_run.run_id}")
	lines.append("")
	lines.append(f"- **Created at:** {analysis_run.created_at}")
	lines.append(f"- **Judge:** {analysis_run.judge}")
	lines.append(f"- **Settings hash:** {analysis_run.settings_hash}")
	for section in sections:
		lines.append("")
		lines.append(section)

	report = "\n".join(lines)
	return report
