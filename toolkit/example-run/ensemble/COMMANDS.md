# Ensemble Example Commands

Run from `grace-toolkit/example-run/ensemble`.

```bash
mkdir -p output/weights output/ensembles output/evaluation

python3 ../../src/ensemble-creation/task_weight_generator.py \
  --pred-dir input/predictions \
  --gold input/gold.json \
  --evaluator ../../src/ensemble-creation/track2_scoring_program.py \
  --output-dir output/weights \
  --iou 0.5

python3 ../../src/ensemble-creation/prediction_aggregator.py \
  --inputs input/predictions/model_a-s1.json input/predictions/model_b-s1.json input/predictions/model_c-s1.json \
  --weights output/weights/weights_s1.json \
  --output output/ensembles/ensemble_s1.json \
  --save-report output/ensembles/ensemble_s1_report.json \
  --task-evaluated 1 \
  --s1-strategy majority

python3 ../../src/ensemble-creation/prediction_aggregator.py \
  --inputs input/predictions/model_a-s2.json input/predictions/model_b-s2.json \
  --weights output/weights/weights_s2.json \
  --output output/ensembles/ensemble_s2.json \
  --save-report output/ensembles/ensemble_s2_report.json \
  --task-evaluated 2 \
  --s2-strategy cluster_vote \
  --entity-iou 0.5 \
  --min-votes 2

python3 ../../src/ensemble-creation/prediction_aggregator.py \
  --inputs input/predictions/model_a-s3.json input/predictions/model_b-s3.json \
  --weights output/weights/weights_s3.json \
  --output output/ensembles/ensemble_s3.json \
  --save-report output/ensembles/ensemble_s3_report.json \
  --task-evaluated 3 \
  --s2-strategy cluster_vote \
  --s3-strategy entity_aligned_vote \
  --entity-iou 0.5 \
  --relation-iou 0.5 \
  --min-votes 2

python3 ../../src/ensemble-creation/track2_scoring_program.py \
  --predictions output/ensembles/ensemble_s1.json \
  --gold input/gold.json \
  --task 1 \
  --output output/evaluation/ensemble_s1_results.json

python3 ../../src/ensemble-creation/track2_scoring_program.py \
  --predictions output/ensembles/ensemble_s2.json \
  --gold input/gold.json \
  --task 2 \
  --output output/evaluation/ensemble_s2_results.json

python3 ../../src/ensemble-creation/track2_scoring_program.py \
  --predictions output/ensembles/ensemble_s3.json \
  --gold input/gold.json \
  --task 3 \
  --output output/evaluation/ensemble_s3_results.json
```
