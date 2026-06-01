# Source Unifier

Convert CASIMEDICOS argumentation records into GRACE format and merge them with an existing GRACE JSON file.

## Inputs

- GRACE JSON array.
- CASIMEDICOS main file, either JSON or JSONL, with tokenized `text` and BIO `labels`.
- CASIMEDICOS relations file, either JSON or JSONL, keyed by record ID.

## Usage

```bash
python3 source_unifier.py \
  --grace grace.json \
  --casimedicos-main casimedicos_ordered.jsonl \
  --casimedicos-relations casimedicos_relations.jsonl \
  --output unified.json
```

By default, converted CASIMEDICOS records keep the post-answer rationale/explanation sentences. This improves relation coverage because CASIMEDICOS relations can point to rationale text.

To truncate CASIMEDICOS records at the `CORRECT ANSWER:` line:

```bash
python3 source_unifier.py \
  --grace grace.json \
  --casimedicos-main casimedicos_ordered.jsonl \
  --casimedicos-relations casimedicos_relations.jsonl \
  --output unified_no_rationale.json \
  --exclude-casimedicos-rationale
```

## Output

The output is one GRACE-style JSON array. Original GRACE records receive `origin: "GRACE"` and converted CASIMEDICOS records receive `origin: "CASIMEDICOS"`.

See `../../example-run/source-unification/COMMANDS.md` for a complete tiny run.
