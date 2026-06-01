# GRACE Toolkit

## Installation

The splitter and source unifier use only the Python standard library. The scorer uses `scikit-learn` when it is installed; a small fallback is included so the tiny examples can run in minimal environments.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Repository Layout

```text
src/data-splitter/       Split JSON datasets into train/eval/test subsets.
src/source-unifier/      Convert CASIMEDICOS and merge it with GRACE data.
src/ensemble-creation/   Generate task weights, ensemble predictions, assemble runs.
example-run/             Tiny runnable examples with local COMMANDS.md files.
METHODS.md               Concise method notes for paper/report writing.
```

## Quickstart

Each example folder contains the exact commands used to generate its outputs. Run them directly from the folder-specific `COMMANDS.md` files:

- `example-run/splitted-data/COMMANDS.md`
- `example-run/source-unification/COMMANDS.md`
- `example-run/ensemble/COMMANDS.md`
- `example-run/submission-ready-run/COMMANDS.md`

## Input Format

The tools expect GRACE-style JSON arrays. A case usually contains:

- `id`
- `raw_text`
- `metadata.context_sentences`
- `metadata.choices`
- `annotations` for gold data, or `predictions` for model output

Final submissions only need the original case structure plus a populated `predictions` block.

## Notes

- CASIMEDICOS rationale/explanation text is included by default during conversion because some source relations point to those explanation sentences. Use `--exclude-casimedicos-rationale` if you want converted records truncated at `CORRECT ANSWER:`.
- The ensemble scripts keep task-specific outputs separate and then merge S1, S2, and S3 with `run_assembler.py`.
