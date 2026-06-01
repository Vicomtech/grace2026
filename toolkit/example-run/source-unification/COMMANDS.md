# Source Unification Example Commands

Run from `grace-toolkit/example-run/source-unification`.

```bash
mkdir -p output

python3 ../../src/source-unifier/source_unifier.py \
  --grace input/grace_sample.json \
  --casimedicos-main input/casimedicos_sample_ordered.jsonl \
  --casimedicos-relations input/casimedicos_sample_relations.jsonl \
  --output output/unified_grace_casimedicos.json
```
