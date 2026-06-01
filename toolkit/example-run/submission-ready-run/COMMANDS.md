# Submission Assembly Example Commands

Run from `grace-toolkit/example-run/submission-ready-run`.

This example uses the task-specific ensemble outputs generated in `../ensemble`.

```bash
mkdir -p output

python3 ../../src/ensemble-creation/run_assembler.py \
  --s1 ../ensemble/output/ensembles/ensemble_s1.json \
  --s2 ../ensemble/output/ensembles/ensemble_s2.json \
  --s3 ../ensemble/output/ensembles/ensemble_s3.json \
  --output output/final_submission.json \
  --save-report output/assembly_report.json

python3 ../../src/ensemble-creation/track2_scoring_program.py \
  --predictions output/final_submission.json \
  --gold ../ensemble/input/gold.json \
  --task all \
  --output output/final_results.json
```
