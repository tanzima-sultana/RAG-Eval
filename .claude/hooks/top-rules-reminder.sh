#!/usr/bin/env bash
# UserPromptSubmit hook.
# Input: the hook JSON on stdin (not used).
# Output: the "## Top rules" section of CLAUDE.md on stdout, which Claude Code adds to the context of every prompt.
# What it does: prints CLAUDE.md from the "## Top rules" heading up to, not including, the "## Code style" heading,
# so the top rules are in front of Claude with every message, not only at the start of the session.

cat > /dev/null

claude_md="$CLAUDE_PROJECT_DIR/CLAUDE.md"
if [ ! -f "$claude_md" ]; then
	exit 0
fi

echo "Before answering or acting on this message, go through these top rules from CLAUDE.md:"
sed -n '/^## Top rules/,/^## Code style/p' "$claude_md" | sed '$d'
