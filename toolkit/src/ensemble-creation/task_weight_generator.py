#!/usr/bin/env python3
"""
Generate task-specific weights for GRACE ensemble runs by evaluating prediction files.

The script reads all JSON prediction files in a directory, evaluates each file on
the task implied by its filename, and writes:

- weights_s1.json
- weights_s2.json
- weights_s3.json
- evaluation_scores.json

Filename convention expected:

- *-s1.json       -> evaluated for Subtask 1
- *-s2.json       -> evaluated for Subtask 2
- *-s3.json       -> evaluated for Subtask 3
- *-s3_gold.json  -> evaluated for Subtask 3
- *-global.json   -> evaluated for Subtasks 1, 2 and 3

Weights are keyed by filename stem, matching prediction_aggregator.py behavior.
Example:

  best-source-runs/grace_cm-gpt_5.4_mini-few-s1.json

becomes:

  "grace_cm-gpt_5.4_mini-few-s1": 0.7421
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_eval_module(evaluator_path: Path):
    if not evaluator_path.exists():
        raise FileNotFoundError(f"Evaluator script not found: {evaluator_path}")

    spec = importlib.util.spec_from_file_location("grace_eval_module", evaluator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import evaluator from: {evaluator_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def infer_tasks_from_name(path: Path) -> List[int]:
    stem = path.stem

    if stem.endswith("-global"):
        return [1, 2, 3]

    if stem.endswith("-s1"):
        return [1]

    if stem.endswith("-s2"):
        return [2]

    if stem.endswith("-s3") or stem.endswith("-s3_gold"):
        return [3]

    return []


def apply_transform(score: float, transform: str) -> float:
    if transform == "identity":
        return score

    if transform == "square":
        return score * score

    if transform == "sqrt":
        return math.sqrt(score)

    raise ValueError(f"Unsupported transform: {transform}")


def evaluate_file_for_tasks(
    eval_module: Any,
    predictions_path: Path,
    gold_path: Optional[Path],
    iou: float,
    tasks: List[int],
) -> Dict[str, float]:
    """
    Evaluate only the requested tasks.

    This avoids the problem of task-specific files having empty prediction fields
    for the other subtasks.
    """
    cases = eval_module._enrich_cases(
        eval_module._prepare_cases(predictions_path, gold_path)
    )

    scores: Dict[str, float] = {}

    if 1 in tasks:
        result = eval_module.evaluate_subtask1(cases)
        scores["s1"] = float(result["official_score"])

    if 2 in tasks:
        result = eval_module.evaluate_subtask2(cases, iou)
        scores["s2"] = float(result["official_score"])

    if 3 in tasks:
        result = eval_module.evaluate_subtask3(cases, iou)
        scores["s3"] = float(result["official_score"])

    return scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate GRACE task-specific weights from official evaluation scores."
    )

    parser.add_argument(
        "--pred-dir",
        type=Path,
        required=True,
        help="Directory containing prediction JSON files.",
    )
    parser.add_argument(
        "--evaluator",
        type=Path,
        required=True,
        help="Path to evaluate_track2_starter.py.",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=None,
        help="Optional separate gold file. If omitted, annotations inside each prediction file are used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where weights and score reports will be written.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold used by the evaluator for relaxed metrics.",
    )
    parser.add_argument(
        "--transform",
        choices=["identity", "square", "sqrt"],
        default="identity",
        help=(
            "Transform applied to official scores before using them as weights. "
            "Use identity to set weight = score."
        ),
    )
    parser.add_argument(
        "--include-unknown-as-global",
        action="store_true",
        help=(
            "If set, files that do not match -s1/-s2/-s3/-s3_gold/-global "
            "are evaluated as global files for all tasks."
        ),
    )

    args = parser.parse_args()

    if not args.pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {args.pred_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    eval_module = load_eval_module(args.evaluator)

    weights = {
        "s1": {},
        "s2": {},
        "s3": {},
    }

    score_report: Dict[str, Dict[str, Any]] = {}

    prediction_files = sorted(args.pred_dir.glob("*.json"))

    if not prediction_files:
        raise ValueError(f"No JSON files found in: {args.pred_dir}")

    for path in prediction_files:
        tasks = infer_tasks_from_name(path)

        if not tasks and args.include_unknown_as_global:
            tasks = [1, 2, 3]

        if not tasks:
            score_report[path.stem] = {
                "file": str(path),
                "status": "skipped",
                "reason": "Could not infer task from filename.",
            }
            continue

        try:
            scores = evaluate_file_for_tasks(
                eval_module=eval_module,
                predictions_path=path,
                gold_path=args.gold,
                iou=args.iou,
                tasks=tasks,
            )

            transformed_scores = {
                task_name: round(apply_transform(score, args.transform), 6)
                for task_name, score in scores.items()
            }

            for task_name, weight in transformed_scores.items():
                weights[task_name][path.stem] = weight

            score_report[path.stem] = {
                "file": str(path),
                "status": "ok",
                "tasks": tasks,
                "official_scores": scores,
                "weights": transformed_scores,
            }

        except Exception as exc:
            score_report[path.stem] = {
                "file": str(path),
                "status": "error",
                "tasks": tasks,
                "error": str(exc),
            }

    weights_s1_path = args.output_dir / "weights_s1.json"
    weights_s2_path = args.output_dir / "weights_s2.json"
    weights_s3_path = args.output_dir / "weights_s3.json"
    report_path = args.output_dir / "evaluation_scores.json"

    weights_s1_path.write_text(
        json.dumps(weights["s1"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    weights_s2_path.write_text(
        json.dumps(weights["s2"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    weights_s3_path.write_text(
        json.dumps(weights["s3"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(score_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Wrote: {weights_s1_path}")
    print(f"Wrote: {weights_s2_path}")
    print(f"Wrote: {weights_s3_path}")
    print(f"Wrote: {report_path}")

    errors = {
        name: row
        for name, row in score_report.items()
        if row["status"] == "error"
    }

    skipped = {
        name: row
        for name, row in score_report.items()
        if row["status"] == "skipped"
    }

    if skipped:
        print(f"\nSkipped {len(skipped)} files because their task could not be inferred.")

    if errors:
        print(f"\nWarning: {len(errors)} files failed during evaluation. See {report_path}")


if __name__ == "__main__":
    main()