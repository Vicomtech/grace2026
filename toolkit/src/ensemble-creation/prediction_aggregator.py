#!/usr/bin/env python3
"""
Aggregate multiple GRACE-format prediction files into a single scorer-ready GRACE file.

Main design
-----------
- The script uses the first input file as the structural reference.
- It preserves each case's original fields (`id`, `raw_text`, `metadata`, `annotations`).
- It overwrites only the `predictions` block with aggregated predictions.
- It can also write a detailed JSON report with voting summaries.

Supported aggregation
---------------------
Subtask 1 (sentence relevancy)
- majority
- weighted

Subtask 2 (entities)
- best_model
- exact_vote
- cluster_vote

Subtask 3 (relations)
- exact_vote
- entity_aligned_vote
- cluster_vote

Notes
-----
- Input files must be GRACE-style JSON arrays.
- The scorer accepts predictions embedded in the same GRACE file, so the output
  is directly scorer-ready.
- Relations are rebuilt so that `arg1_id` / `arg2_id` always reference IDs that
  exist in the aggregated `predictions.entities` block for the same case.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SENTENCE_LABELS = ["relevant", "not-relevant"]
ENTITY_TYPES = ["Premise", "Claim"]
RELATION_TYPES = ["Support", "Attack"]
DEFAULT_IOU = 0.5
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    type: str
    source_model: str
    weight: float
    entity_id: Optional[str] = None


@dataclass(frozen=True)
class RelationSpan:
    arg1_start: int
    arg1_end: int
    arg2_start: int
    arg2_end: int
    relation_type: str
    source_model: str
    weight: float


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array.")
    return data


def has_prediction_content(block: Dict[str, Any]) -> bool:
    return bool(
        block.get("sentence_relevancy") or block.get("entities") or block.get("relations")
    )


def normalize_text(text: str) -> str:
    text = " ".join(str(text).replace("\n", " ").split()).strip()
    return text


def normalize_relation_text(text: str) -> str:
    text = normalize_text(text)
    text = text.strip(" \t\r\n\"'“”‘’()[]{}<>.,;:!?")
    text = re.sub(r"[.。,;:!?]+$", "", text)
    text = re.sub(r"^[.。,;:!?()\[\]{}<>]+", "", text)
    return text


def tokenize_positions(text: str) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def token_set(token_positions: List[Tuple[int, int]], start: int, end: int) -> frozenset[int]:
    return frozenset(
        i for i, (ts, te) in enumerate(token_positions)
        if ts >= start and te <= end
    )


def token_iou_for_spans(
    token_positions: List[Tuple[int, int]],
    a_start: int,
    a_end: int,
    b_start: int,
    b_end: int,
) -> float:
    a = token_set(token_positions, a_start, a_end)
    b = token_set(token_positions, b_start, b_end)
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def path_alias(path: Path) -> str:
    return path.stem


def load_weights(weights_path: Optional[Path], aliases: List[str]) -> Dict[str, float]:
    weights = {alias: 1.0 for alias in aliases}
    if weights_path is None:
        return weights

    raw = json.loads(weights_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("--weights must point to a JSON object mapping model alias to weight.")

    for alias in aliases:
        if alias in raw:
            weights[alias] = float(raw[alias])
    return weights


def prepare_inputs(paths: List[Path]) -> Tuple[List[str], List[Dict[str, Dict[str, Any]]]]:
    aliases = [path_alias(p) for p in paths]
    loaded = []
    for alias, path in zip(aliases, paths):
        cases = load_json_array(path)
        by_id = {}
        for case in cases:
            pred_block = case.get("predictions", {})
            if not has_prediction_content(pred_block):
                pred_block = case.get("annotations", {})
            by_id[case["id"]] = {
                "full_case": case,
                "predictions": pred_block,
            }
        loaded.append(by_id)
    return aliases, loaded


def assert_compatible_structure(case_id: str, reference: Dict[str, Any], other: Dict[str, Any]) -> None:
    if reference["id"] != other["id"]:
        raise ValueError(f"Case ID mismatch for {case_id}.")
    if reference["raw_text"] != other["raw_text"]:
        raise ValueError(f"Case {case_id}: raw_text differs across input files.")
    if reference.get("metadata") != other.get("metadata"):
        raise ValueError(f"Case {case_id}: metadata differs across input files.")


def find_case_order(reference_cases: Dict[str, Dict[str, Any]], loaded: List[Dict[str, Dict[str, Any]]]) -> List[str]:
    case_ids = list(reference_cases.keys())
    for case_id in case_ids:
        for model_cases in loaded[1:]:
            if case_id not in model_cases:
                raise ValueError(f"Case '{case_id}' is missing from one input file.")
    return case_ids


def aggregate_sentence_labels(
    case_predictions: Dict[str, Dict[str, Any]],
    model_weights: Dict[str, float],
    strategy: str,
) -> Tuple[List[str], Dict[str, Any]]:
    model_names = list(case_predictions.keys())
    lengths = {m: len(case_predictions[m].get("sentence_relevancy", [])) for m in model_names}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Sentence label length mismatch across models: {lengths}")

    n = next(iter(lengths.values()), 0)
    final_labels: List[str] = []
    report_rows = []

    for i in range(n):
        label_weights = defaultdict(float)
        label_votes = defaultdict(int)

        for model_name in model_names:
            label = case_predictions[model_name]["sentence_relevancy"][i]
            if label not in SENTENCE_LABELS:
                continue
            label_votes[label] += 1
            if strategy == "weighted":
                label_weights[label] += model_weights[model_name]
            else:
                label_weights[label] += 1.0

        # tie-break in task-friendly order: relevant first, then not-relevant
        best_label = max(
            SENTENCE_LABELS,
            key=lambda lbl: (label_weights.get(lbl, 0.0), label_votes.get(lbl, 0), -SENTENCE_LABELS.index(lbl))
        )
        final_labels.append(best_label)
        report_rows.append({
            "sentence_index": i,
            "votes": dict(label_votes),
            "weighted_votes": {k: round(v, 6) for k, v in label_weights.items()},
            "selected": best_label,
        })

    return final_labels, {"strategy": strategy, "sentences": report_rows}


def collect_model_entities(case_predictions: Dict[str, Dict[str, Any]], model_weights: Dict[str, float]) -> List[Span]:
    entities: List[Span] = []
    for model_name, pred in case_predictions.items():
        for ent in pred.get("entities", []):
            entities.append(
                Span(
                    start=int(ent["start"]),
                    end=int(ent["end"]),
                    text=str(ent.get("text", "")),
                    type=str(ent["type"]),
                    source_model=model_name,
                    weight=model_weights[model_name],
                    entity_id=ent.get("id"),
                )
            )
    return entities


def exact_entity_key(span: Span) -> Tuple[int, int, str]:
    return (span.start, span.end, span.type)


def cluster_spans(
    spans: List[Span],
    raw_text: str,
    iou_threshold: float,
) -> List[List[Span]]:
    token_positions = tokenize_positions(raw_text)
    clusters: List[List[Span]] = []

    for span in spans:
        placed = False
        for cluster in clusters:
            rep = cluster[0]
            if span.type != rep.type:
                continue
            iou = token_iou_for_spans(token_positions, span.start, span.end, rep.start, rep.end)
            if iou >= iou_threshold:
                cluster.append(span)
                placed = True
                break
        if not placed:
            clusters.append([span])

    return clusters


def choose_cluster_representative(cluster: List[Span], model_weights: Dict[str, float]) -> Span:
    counts = defaultdict(int)
    total_weights = defaultdict(float)
    for span in cluster:
        key = (span.start, span.end, span.type, span.text)
        counts[key] += 1
        total_weights[key] += span.weight

    def score(span: Span) -> Tuple[float, int, int]:
        key = (span.start, span.end, span.type, span.text)
        return (
            total_weights[key],
            counts[key],
            -(span.end - span.start),
        )

    return max(cluster, key=score)


def aggregate_entities(
    case_predictions: Dict[str, Dict[str, Any]],
    raw_text: str,
    model_weights: Dict[str, float],
    strategy: str,
    min_votes: int,
    iou_threshold: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, str]]:
    spans = collect_model_entities(case_predictions, model_weights)
    report: Dict[str, Any] = {"strategy": strategy, "clusters": []}

    if strategy == "best_model":
        best_model = max(model_weights, key=lambda m: model_weights[m])
        best_entities = copy.deepcopy(case_predictions[best_model].get("entities", []))
        # Reassign IDs so downstream relations are rebuilt against a clean, deterministic set.
        output_entities: List[Dict[str, Any]] = []
        old_to_new = {}
        for idx, ent in enumerate(best_entities, start=1):
            new_id = f"agg_e{idx}"
            old_to_new[f"{best_model}:{ent['id']}"] = new_id
            output_entities.append({
                "id": new_id,
                "text": ent["text"],
                "start": int(ent["start"]),
                "end": int(ent["end"]),
                "type": ent["type"],
            })
        report["selected_model"] = best_model
        return output_entities, report, old_to_new

    if strategy == "exact_vote":
        grouped = defaultdict(list)
        for span in spans:
            grouped[exact_entity_key(span)].append(span)
        clusters = list(grouped.values())
    elif strategy == "cluster_vote":
        clusters = cluster_spans(spans, raw_text, iou_threshold)
    else:
        raise ValueError(f"Unsupported s2 strategy: {strategy}")

    output_entities: List[Dict[str, Any]] = []
    old_to_new: Dict[str, str] = {}

    sorted_clusters = sorted(
        clusters,
        key=lambda c: (
            -sum(s.weight for s in c),
            -len(c),
            choose_cluster_representative(c, model_weights).start,
            choose_cluster_representative(c, model_weights).end,
        )
    )

    entity_counter = 1
    for cluster in sorted_clusters:
        rep = choose_cluster_representative(cluster, model_weights)
        unique_models = sorted({s.source_model for s in cluster})
        total_weight = sum(model_weights[m] for m in unique_models)

        if len(unique_models) < min_votes:
            report["clusters"].append({
                "status": "dropped",
                "reason": "min_votes",
                "models": unique_models,
                "total_weight": round(total_weight, 6),
                "representative": {
                    "text": rep.text,
                    "start": rep.start,
                    "end": rep.end,
                    "type": rep.type,
                },
            })
            continue

        new_id = f"agg_e{entity_counter}"
        entity_counter += 1
        output_entities.append({
            "id": new_id,
            "text": rep.text if rep.text else raw_text[rep.start:rep.end],
            "start": rep.start,
            "end": rep.end,
            "type": rep.type,
        })

        for span in cluster:
            old_to_new[f"{span.source_model}:{span.entity_id}"] = new_id

        report["clusters"].append({
            "status": "kept",
            "models": unique_models,
            "vote_count": len(unique_models),
            "total_weight": round(total_weight, 6),
            "representative": {
                "id": new_id,
                "text": rep.text,
                "start": rep.start,
                "end": rep.end,
                "type": rep.type,
            },
        })

    output_entities.sort(key=lambda e: (e["start"], e["end"], e["type"], e["id"]))
    return output_entities, report, old_to_new


def containing_sentence(start: int, end: int, sentences: List[Dict[str, Any]]) -> Optional[int]:
    for i, s in enumerate(sentences):
        if start >= s["start"] and end <= s["end"]:
            return i
    best_idx, best_overlap = None, 0
    for i, s in enumerate(sentences):
        overlap = max(0, min(end, s["end"]) - max(start, s["start"]))
        if overlap > best_overlap:
            best_overlap, best_idx = overlap, i
    return best_idx


def gate_entities_by_s1(
    entities: List[Dict[str, Any]],
    sentence_labels: List[str],
    sentences: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[Dict[str, Any]]]:
    kept = []
    removed = []
    id_remap = {}
    counter = 1

    for ent in entities:
        sent_idx = containing_sentence(ent["start"], ent["end"], sentences)
        if sent_idx is not None and sent_idx < len(sentence_labels) and sentence_labels[sent_idx] != "relevant":
            removed.append(ent)
            continue
        new_id = f"agg_e{counter}"
        counter += 1
        id_remap[ent["id"]] = new_id
        kept.append(ent | {"id": new_id})

    return kept, id_remap, removed


def collect_model_relations(
    case_predictions: Dict[str, Dict[str, Any]],
    model_weights: Dict[str, float],
) -> List[RelationSpan]:
    rows: List[RelationSpan] = []
    for model_name, pred in case_predictions.items():
        entities = {e["id"]: e for e in pred.get("entities", [])}
        for rel in pred.get("relations", []):
            e1 = entities.get(rel.get("arg1_id"))
            e2 = entities.get(rel.get("arg2_id"))
            if e1 is None or e2 is None:
                continue
            rows.append(
                RelationSpan(
                    arg1_start=int(e1["start"]),
                    arg1_end=int(e1["end"]),
                    arg2_start=int(e2["start"]),
                    arg2_end=int(e2["end"]),
                    relation_type=str(rel["relation_type"]),
                    source_model=model_name,
                    weight=model_weights[model_name],
                )
            )
    return rows


def relation_cluster_key(rep: RelationSpan, cand: RelationSpan, token_positions: List[Tuple[int, int]], iou_threshold: float) -> bool:
    if rep.relation_type != cand.relation_type:
        return False
    arg1_iou = token_iou_for_spans(token_positions, rep.arg1_start, rep.arg1_end, cand.arg1_start, cand.arg1_end)
    arg2_iou = token_iou_for_spans(token_positions, rep.arg2_start, rep.arg2_end, cand.arg2_start, cand.arg2_end)
    return arg1_iou >= iou_threshold and arg2_iou >= iou_threshold


def find_aggregated_entity_for_span(
    entities: List[Dict[str, Any]],
    start: int,
    end: int,
    token_positions: List[Tuple[int, int]],
    exact_only: bool,
    iou_threshold: float,
) -> Optional[str]:
    candidates = []
    for ent in entities:
        if exact_only:
            if ent["start"] == start and ent["end"] == end:
                candidates.append((1.0, ent))
        else:
            iou = token_iou_for_spans(token_positions, start, end, ent["start"], ent["end"])
            if iou >= iou_threshold:
                candidates.append((iou, ent))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], x[1]["start"], x[1]["end"]))
    return candidates[0][1]["id"]


def aggregate_relations(
    case_predictions: Dict[str, Dict[str, Any]],
    aggregated_entities: List[Dict[str, Any]],
    model_weights: Dict[str, float],
    strategy: str,
    min_votes: int,
    iou_threshold: float,
    raw_text: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    token_positions = tokenize_positions(raw_text)
    report: Dict[str, Any] = {"strategy": strategy, "items": []}

    if strategy == "exact_vote":
        vote_by_key = defaultdict(lambda: {"models": set(), "weight": 0.0})
        for model_name, pred in case_predictions.items():
            entities = {e["id"]: e for e in pred.get("entities", [])}
            for rel in pred.get("relations", []):
                e1 = entities.get(rel.get("arg1_id"))
                e2 = entities.get(rel.get("arg2_id"))
                if e1 is None or e2 is None:
                    continue
                agg1 = find_aggregated_entity_for_span(
                    aggregated_entities, int(e1["start"]), int(e1["end"]), token_positions, True, iou_threshold
                )
                agg2 = find_aggregated_entity_for_span(
                    aggregated_entities, int(e2["start"]), int(e2["end"]), token_positions, True, iou_threshold
                )
                if agg1 is None or agg2 is None:
                    continue
                key = (agg1, agg2, rel["relation_type"])
                vote_by_key[key]["models"].add(model_name)
                vote_by_key[key]["weight"] += model_weights[model_name]

    elif strategy == "entity_aligned_vote":
        vote_by_key = defaultdict(lambda: {"models": set(), "weight": 0.0})
        for model_name, pred in case_predictions.items():
            entities = {e["id"]: e for e in pred.get("entities", [])}
            for rel in pred.get("relations", []):
                e1 = entities.get(rel.get("arg1_id"))
                e2 = entities.get(rel.get("arg2_id"))
                if e1 is None or e2 is None:
                    continue
                agg1 = find_aggregated_entity_for_span(
                    aggregated_entities, int(e1["start"]), int(e1["end"]), token_positions, False, iou_threshold
                )
                agg2 = find_aggregated_entity_for_span(
                    aggregated_entities, int(e2["start"]), int(e2["end"]), token_positions, False, iou_threshold
                )
                if agg1 is None or agg2 is None:
                    continue
                key = (agg1, agg2, rel["relation_type"])
                vote_by_key[key]["models"].add(model_name)
                vote_by_key[key]["weight"] += model_weights[model_name]

    elif strategy == "cluster_vote":
        raw_relations = collect_model_relations(case_predictions, model_weights)
        clusters: List[List[RelationSpan]] = []
        for rel in raw_relations:
            placed = False
            for cluster in clusters:
                rep = cluster[0]
                if relation_cluster_key(rep, rel, token_positions, iou_threshold):
                    cluster.append(rel)
                    placed = True
                    break
            if not placed:
                clusters.append([rel])

        vote_by_key = defaultdict(lambda: {"models": set(), "weight": 0.0})
        for cluster in clusters:
            unique_models = {r.source_model for r in cluster}
            if len(unique_models) < min_votes:
                continue
            rep = max(cluster, key=lambda r: (r.weight, -(r.arg1_end-r.arg1_start + r.arg2_end-r.arg2_start)))
            agg1 = find_aggregated_entity_for_span(
                aggregated_entities, rep.arg1_start, rep.arg1_end, token_positions, False, iou_threshold
            )
            agg2 = find_aggregated_entity_for_span(
                aggregated_entities, rep.arg2_start, rep.arg2_end, token_positions, False, iou_threshold
            )
            if agg1 is None or agg2 is None:
                continue
            key = (agg1, agg2, rep.relation_type)
            vote_by_key[key]["models"].update(unique_models)
            vote_by_key[key]["weight"] += sum(model_weights[m] for m in unique_models)
    else:
        raise ValueError(f"Unsupported s3 strategy: {strategy}")

    output_relations = []
    counter = 1
    for key, payload in sorted(
        vote_by_key.items(),
        key=lambda item: (-item[1]["weight"], -len(item[1]["models"]), item[0][0], item[0][1], item[0][2])
    ):
        if len(payload["models"]) < min_votes:
            report["items"].append({
                "status": "dropped",
                "reason": "min_votes",
                "relation": {"arg1_id": key[0], "arg2_id": key[1], "relation_type": key[2]},
                "vote_count": len(payload["models"]),
                "total_weight": round(payload["weight"], 6),
                "models": sorted(payload["models"]),
            })
            continue
        rel_id = f"agg_r{counter}"
        counter += 1
        output_relations.append({
            "id": rel_id,
            "arg1_id": key[0],
            "arg2_id": key[1],
            "relation_type": key[2],
        })
        report["items"].append({
            "status": "kept",
            "relation": {"id": rel_id, "arg1_id": key[0], "arg2_id": key[1], "relation_type": key[2]},
            "vote_count": len(payload["models"]),
            "total_weight": round(payload["weight"], 6),
            "models": sorted(payload["models"]),
        })

    return output_relations, report


def build_empty_predictions() -> Dict[str, Any]:
    return {
        "sentence_relevancy": [],
        "entities": [],
        "relations": [],
    }


def aggregate_case(
    case_id: str,
    base_case: Dict[str, Any],
    case_predictions: Dict[str, Dict[str, Any]],
    model_weights: Dict[str, float],
    s1_strategy: str,
    s2_strategy: str,
    s3_strategy: str,
    min_votes: int,
    entity_iou: float,
    relation_iou: float,
    gate_by_s1: bool,
    task_evaluated: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    raw_text = base_case["raw_text"]
    sentences = base_case["metadata"]["context_sentences"]

    sentence_relevancy: List[str] = []
    entities: List[Dict[str, Any]] = []
    relations: List[Dict[str, Any]] = []

    s1_report: Dict[str, Any] = {"strategy": None, "skipped": True}
    s2_report: Dict[str, Any] = {"strategy": None, "skipped": True}
    s3_report: Dict[str, Any] = {"strategy": None, "skipped": True}
    gating_report = {"enabled": False, "removed_entities": []}

     # Task 1 or all
    if task_evaluated in (1, 4):
        sentence_relevancy, s1_report = aggregate_sentence_labels(
            case_predictions, model_weights, s1_strategy
        )

    # Task 2 or all
    if task_evaluated in (2, 4):
        entities, s2_report, _entity_old_to_new = aggregate_entities(
            case_predictions, raw_text, model_weights, s2_strategy, min_votes, entity_iou
        )

        if gate_by_s1 and task_evaluated == 4:
            gating_report = {"enabled": True, "removed_entities": []}
            entities, id_remap, removed = gate_entities_by_s1(
                entities, sentence_relevancy, sentences
            )
            gating_report["removed_entities"] = removed

    # Task 3 or all
    if task_evaluated in (3, 4):
        # If only task 3 is being aggregated, relations still need entities to align spans.
        if task_evaluated == 3:
            entities, s2_report, _entity_old_to_new = aggregate_entities(
                case_predictions, raw_text, model_weights, s2_strategy, min_votes, entity_iou
            )

        relations, s3_report = aggregate_relations(
            case_predictions,
            entities,
            model_weights,
            s3_strategy,
            min_votes,
            relation_iou,
            raw_text,
        )

    output_case = copy.deepcopy(base_case)
    output_case["predictions"] = {
        "sentence_relevancy": sentence_relevancy,
        "entities": entities,
        "relations": relations,
    }

    report = {
        "case_id": case_id,
        "task_evaluated": task_evaluated,
        "subtask1": s1_report,
        "subtask2": s2_report,
        "subtask3": s3_report,
        "gating": gating_report,
    }
    return output_case, report


def aggregate_predictions(
    input_paths: List[Path],
    output_path: Path,
    save_report: Optional[Path],
    s1_strategy: str,
    s2_strategy: str,
    s3_strategy: str,
    min_votes: int,
    entity_iou: float,
    relation_iou: float,
    weights_path: Optional[Path],
    gate_by_s1: bool,
    task_evaluated: int,
) -> None:
    aliases, loaded = prepare_inputs(input_paths)
    model_weights = load_weights(weights_path, aliases)

    reference_cases = loaded[0]
    case_order = find_case_order(reference_cases, loaded)

    output_cases = []
    report = {
        "run_config": {
            "inputs": [str(p) for p in input_paths],
            "aliases": aliases,
            "weights": model_weights,
            "s1_strategy": s1_strategy,
            "s2_strategy": s2_strategy,
            "s3_strategy": s3_strategy,
            "min_votes": min_votes,
            "entity_iou": entity_iou,
            "relation_iou": relation_iou,
            "gate_by_s1": gate_by_s1,
            "task_evaluated": task_evaluated,
        },
        "case_reports": [],
    }

    for case_id in case_order:
        base_case = reference_cases[case_id]["full_case"]
        for model_cases in loaded[1:]:
            assert_compatible_structure(case_id, base_case, model_cases[case_id]["full_case"])

        case_predictions = {
            alias: model_cases[case_id]["predictions"]
            for alias, model_cases in zip(aliases, loaded)
        }

        aggregated_case, case_report = aggregate_case(
            case_id=case_id,
            base_case=base_case,
            case_predictions=case_predictions,
            model_weights=model_weights,
            s1_strategy=s1_strategy,
            s2_strategy=s2_strategy,
            s3_strategy=s3_strategy,
            min_votes=min_votes,
            entity_iou=entity_iou,
            relation_iou=relation_iou,
            gate_by_s1=gate_by_s1,
            task_evaluated=task_evaluated,
        )
        output_cases.append(aggregated_case)
        report["case_reports"].append(case_report)

    output_path.write_text(json.dumps(output_cases, indent=2, ensure_ascii=False), encoding="utf-8")
    if save_report is not None:
        save_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate multiple GRACE-format prediction files into one scorer-ready file."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="Input GRACE-format JSON files containing predictions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the aggregated GRACE-format JSON output.",
    )
    parser.add_argument(
        "--save-report",
        type=Path,
        default=None,
        help="Optional path to a JSON report with voting and clustering details.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Optional JSON file mapping input file stem to weight, e.g. {'model_a': 1.2}.",
    )
    parser.add_argument(
        "--s1-strategy",
        choices=["majority", "weighted"],
        default="weighted",
        help="Aggregation strategy for Subtask 1.",
    )
    parser.add_argument(
        "--s2-strategy",
        choices=["best_model", "exact_vote", "cluster_vote"],
        default="cluster_vote",
        help="Aggregation strategy for Subtask 2.",
    )
    parser.add_argument(
        "--s3-strategy",
        choices=["exact_vote", "entity_aligned_vote", "cluster_vote"],
        default="cluster_vote",
        help="Aggregation strategy for Subtask 3.",
    )
    parser.add_argument(
        "--min-votes",
        type=int,
        default=2,
        help="Minimum number of distinct models that must support an entity or relation.",
    )
    parser.add_argument(
        "--entity-iou",
        type=float,
        default=DEFAULT_IOU,
        help="IoU threshold for entity clustering/alignment.",
    )
    parser.add_argument(
        "--relation-iou",
        type=float,
        default=DEFAULT_IOU,
        help="IoU threshold for relation clustering/alignment.",
    )
    parser.add_argument(
        "--gate-by-s1",
        action="store_true",
        help="If set, remove aggregated entities that fall in sentences aggregated as not-relevant.",
    )
    parser.add_argument(
        "--task-evaluated",
        type=int,
        choices=[1, 2, 3, 4],
        default=4,
        help="Task to aggregate: 1=S1, 2=S2, 3=S3, 4=all.",
    )
    args = parser.parse_args()

    if len(args.inputs) < 2:
        raise ValueError("At least two input files are required for aggregation.")

    aggregate_predictions(
        input_paths=args.inputs,
        output_path=args.output,
        save_report=args.save_report,
        s1_strategy=args.s1_strategy,
        s2_strategy=args.s2_strategy,
        s3_strategy=args.s3_strategy,
        min_votes=args.min_votes,
        entity_iou=args.entity_iou,
        relation_iou=args.relation_iou,
        weights_path=args.weights,
        gate_by_s1=args.gate_by_s1,
        task_evaluated=args.task_evaluated,
    )


if __name__ == "__main__":
    main()
