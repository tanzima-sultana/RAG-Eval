#!/usr/bin/env bash
# PreToolUse hook for Bash.
# Input: the hook JSON on stdin, holding the Bash command Claude wants to run.
# Output: if the command can call a paid LLM API, a JSON permission decision of "ask",
# so the user must approve this exact run in the permission prompt. Otherwise nothing.
# What it does: src/llm/llm.py get_llm() refuses to build a Claude LLM unless ALLOW_PAID_API=1 is set,
# so a run through the project code can only be paid if its command sets ALLOW_PAID_API=1.
# The hook asks for those commands, and for commands that reach a paid API without get_llm()
# (an API host, the anthropic SDK or ChatAnthropic). Local Ollama and cached runs pass without a prompt.
# See .claude/skills/paid-api-run-approval/SKILL.md.

hook_input="$(cat)"

paid_pattern='(ALLOW_PAID_API=1|anthropic\.com|openai\.com|googleapis\.com|import anthropic|from anthropic|langchain_anthropic|ChatAnthropic)'

if echo "$hook_input" | grep -Eq "$paid_pattern"; then
	cat <<'JSON'
{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "ask",
    "permissionDecisionReason": "PAID API GUARD: this command may call a paid LLM API (it sets ALLOW_PAID_API=1 or reaches a paid API directly). Approve only if you authorized this exact run."
  }
}
JSON
fi
exit 0
