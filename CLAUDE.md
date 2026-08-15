# CLAUDE.md

## Environment setup

This project's Python dependencies (pytest, pandas, pyarrow, etc.) live in a
virtualenv at `~/pyenv`, not in the system Python. Before running any Python
command in this repo (scripts, tests, pip installs), activate it first:

```bash
source ~/pyenv/bin/activate
```

## Committing

Whenever the user asks to commit, run the pre-commit tests first (with
`~/pyenv` activated) before creating the commit:

```bash
source ~/pyenv/bin/activate
python3 -m pytest tests/
```

There is also a git hook (`githooks/pre-commit`, activated via
`core.hooksPath`) that runs `tests/test_dataset.py` automatically on
`git commit` if `src/dataset.py` changed — but it only catches that one
file, and only works if `~/pyenv` is active in the shell running the commit.
Run the full test suite explicitly first so failures surface before the
commit attempt, not as a hook rejection.
