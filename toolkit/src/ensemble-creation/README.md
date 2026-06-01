# Ensemble Creation and Run Assembly

Build task-specific GRACE ensembles and assemble a final submission-ready JSON file.

## Scripts

- `task_weight_generator.py`: evaluates prediction files with the included scorer and writes task-specific weight files.
- `prediction_aggregator.py`: aggregates multiple GRACE prediction files for S1, S2, S3, or all subtasks.
- `run_assembler.py`: combines selected S1, S2, and S3 outputs into one final run.
- `track2_scoring_program.py`: modified official scorer with a `--task` option.

## Typical Workflow

```bash
python3 task_weight_generator.py \
  --pred-dir predictions \
  --gold gold.json \
  --evaluator track2_scoring_program.py \
  --output-dir weights

python3 prediction_aggregator.py \
  --inputs predictions/model_a-s1.json predictions/model_b-s1.json \
  --weights weights/weights_s1.json \
  --task-evaluated 1 \
  --s1-strategy majority \
  --output ensembles/ensemble_s1.json

python3 prediction_aggregator.py \
  --inputs predictions/model_a-s2.json predictions/model_b-s2.json \
  --weights weights/weights_s2.json \
  --task-evaluated 2 \
  --s2-strategy cluster_vote \
  --output ensembles/ensemble_s2.json

python3 prediction_aggregator.py \
  --inputs predictions/model_a-s3.json predictions/model_b-s3.json \
  --weights weights/weights_s3.json \
  --task-evaluated 3 \
  --s3-strategy entity_aligned_vote \
  --output ensembles/ensemble_s3.json

python3 run_assembler.py \
  --s1 ensembles/ensemble_s1.json \
  --s2 ensembles/ensemble_s2.json \
  --s3 ensembles/ensemble_s3.json \
  --output final_submission.json
```

## Notes

- File stems are model aliases. Weight files must use the same stems as the input prediction filenames.
- For S2 and S3, `--min-votes`, `--entity-iou`, and `--relation-iou` control how strict voting/clustering is.
- The final assembled run uses S2 as the authoritative entity block and validates/remaps S3 relation arguments against those entities.

See `../../example-run/ensemble/COMMANDS.md` and `../../example-run/submission-ready-run/COMMANDS.md` for tiny runnable examples.
