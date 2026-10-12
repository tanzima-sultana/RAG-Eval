#!/usr/bin/env bash
# PreToolUse hook for Bash.
# Input: the hook JSON on stdin, holding the Bash command Claude wants to run.
# Output: if the command can call a paid LLM API, a JSON permission decision of "ask",
# so the user must approve this exact run in the permission prompt. Otherwise nothing.
# What it does: a command is treated as paid when it runs something (python, bash, sh,
# curl, wget, source or a ./ path) AND mentions a paid entry point
# (eval.py, eval.sh, the qna or llm modules, RAGAS, ChatAnthropic, or an LLM API host).
# See .claude/skills/paid-api-run-approval/SKILL.md.

hook_input="$(cat)"

runner_pattern='(python3?|bash|[^a-z]sh |curl|wget|source |\./)'
paid_pattern='(eval\.py|eval\.sh|get_eval_qna|generate_eval_qna|qna\.qna|from qna|import qna|llm\.claude|llm\.chatgpt|llm\.gemini|llm\.llm|from llm|import llm|ragas|ChatAnthropic|anthropic\.com|openai\.com|googleapis\.com)'

if echo "$hook_input" | grep -Eq "$runner_pattern"; then
	if echo "$hook_input" | grep -Eq "$paid_pattern"; then
		cat <<'EOF'
{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "ask",
    "permissionDecisionReason": "PAID API GUARD: this command may call a paid LLM API (Claude via RAGAS / src/llm). Approve only if you authorized this exact run."
  }
}
EOF
	fi
fi
exit 0
