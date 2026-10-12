---
name: paid-api-run-approval
description: Requires the user's explicit approval before every single run that can call a paid API (Claude/Anthropic, OpenAI, Gemini). Use before running src/eval.py, scripts/eval.sh, anything that calls qna.get_eval_qna or qna.generate_eval_qna, anything that imports from src/llm/, RAGAS TestsetGenerator or evaluation, or any script, test or one-off Python that may send a request to a paid LLM API. Also use when retrying or re-running such a command after a failure.
---

# Paid API run approval

Every run that can call a paid API costs the user money. Never start one
without the user's explicit approval for that exact run. An approval covers
one run only. It never carries over to a retry, a re-run, a changed
parameter, or a later run in the same conversation.

## What counts as a paid run

A run is paid if it can send even one request to a paid LLM API:

- `src/eval.py` and `scripts/eval.sh`. Eval QnA generation calls Claude
  through RAGAS when its cache file does not exist.
- Any code that calls `qna.qna.get_eval_qna` or `qna.qna.generate_eval_qna`.
- Any code that builds or calls an LLM from `src/llm/` (`claude.py`,
  `chatgpt.py`, `gemini.py`, `llm.py`).
- Any RAGAS test generation or evaluation that uses an LLM.
- Any curl, SDK call or one-off script that reaches `api.anthropic.com`,
  `api.openai.com` or a Gemini endpoint.

Not paid, so no approval is needed: `src/build.py` and `scripts/run.sh`
(local embedding models and a local Qdrant), Hugging Face dataset or model
downloads, imports and `py_compile` checks, and tests where the LLM is
replaced by a stub that raises before any request is made.

If unsure whether a run is paid, treat it as paid.

## Before a paid run

1. Do not run it yet. Do not start it in the background either.
2. Check whether it will really call the API. For eval QnA, build the cache
   path with `common.paths.get_eval_qna_path` and check whether the file
   exists. If it exists, the run loads it and makes no API call. Still tell
   the user that, and still ask, because an approval is needed for any run
   of a paid command.
3. Ask the user with AskUserQuestion. Show:
   - the exact command or code that will run
   - which paid API and model it calls (for example `claude-sonnet-5-5`)
   - the settings that drive the cost (for eval QnA: dataset size,
     `n_sample_documents`, `n_sample_evals`)
   - whether a cache file exists, so no call will be made
   - a cost estimate if one is known from an earlier run's printed
     `Claude total cost` in the logs; otherwise say the cost is not known yet
4. Run it only if the user picks the approve option. Any other answer, or no
   answer, means do not run.

## During and after the run

- Run exactly what was approved. Do not change the command, the model or any
  setting without asking again.
- If the run fails, stop and report the error. Do not retry or re-run it,
  and do not "fix and re-run". Each new attempt needs a new approval.
- After the run, report the cost it printed (`Claude total cost`), so the
  next estimate is based on real numbers.
