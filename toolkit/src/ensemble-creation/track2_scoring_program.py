#!/usr/bin/env python3
"""
This is a modified version of the official evaluation script. The only change is the addition of a parameter so that you can evaluate tasks separately.
"""
"""
IberLEF 2026 – GRACE Shared Task, Track 2: Clinical Case Reasoning (MIR)
=======================================================================

Three subtasks are evaluated:

  Subtask 1 – Evidence Sentence Detection
    Binary classification of sentences as "relevant" or "not-relevant".
    Metrics: per-class, macro-avg, and micro-avg Precision / Recall / F1.
    Official ranking metric: F1 on the "relevant" class.

  Subtask 2 – Fine-Grained Evidence Span Detection
    Extraction of Premise/Claim text spans from the clinical case.
    Two evaluation scopes:
      Scope A (end-to-end): spans predicted in non-relevant sentences are FP.
      Scope B (oracle):     only spans within gold-relevant sentences count.
    Two match criteria per scope:
      Strict  – character-exact boundaries.
      Relaxed – token-level IoU ≥ τ (default 0.5).
    Metrics per scope/criterion: per-type (Premise/Claim), macro-avg, micro-avg P/R/F1.
    Official ranking metric: Scope A, Strict Micro F1.

  Subtask 3 – Argumentative Relation Detection
    Extraction of (Premise, Claim, relation_type) triplets.
    Two match criteria:
      Strict  – all three components match exactly (official ranking metric).
      Relaxed – same relation type + token IoU ≥ τ on both argument spans.
    Metrics: per-type, macro-avg, micro-avg P/R/F1.
    Official ranking metric: Strict Macro F1.

SPAN MATCHING
-------------
  Strict  — start offset AND end offset must be identical to gold.
  Relaxed — token-level Jaccard Index ≥ τ, where tokens are defined as
             word sequences (\\w+) and single punctuation characters.
             This mirrors the official evaluation formula (spec §2.2):
               J(A, B) = |A ∩ B| / |A ∪ B|  ≥  τ
             Tokenising at this level naturally absorbs minor boundary
             disagreements (trailing dots, leading articles, etc.).

  Matching is 1-to-1: each gold span can be claimed by at most one
  predicted span and vice versa (greedy bipartite assignment).

INPUT FORMAT
------------
  Both files are JSON arrays. Each element represents one clinical case:

    {
      "id":       "<case_id>",
      "raw_text": "<full case text used for tokenization>",
      "metadata": {
        "context_sentences": [{"sentence": "...", "start": 0, "end": 42}, ...]
      },
      "annotations": {
        "sentence_relevancy": ["relevant", "not-relevant", ...],
        "entities":  [{"id": "e1", "text": "...", "start": 91, "end": 123,
                       "type": "Premise"}, ...],
        "relations": [{"id": "r1", "arg1_id": "e1", "arg2_id": "e2",
                       "relation_type": "Support"}, ...]
      },
      "predictions": {   <-- same structure as annotations
        "sentence_relevancy": [...],
        "entities":  [...],
        "relations": [...]
      }
    }

  When --gold is provided the `predictions` block is read from the
  predictions file (or the `annotations` block as prediction if the
  `predictions` block is not present and the `annotations` block from
  the gold file.
  When --gold is omitted the `predictions` field of the input file is
  used as the system output and `annotations` as the gold standard.

USAGE
-----
  # Predictions embedded in a single file:
  python evaluate_track2_starter.py --predictions preds.json

  # Separate gold and prediction files:
  python evaluate_track2_starter.py --predictions preds.json --gold gold.json

  # Adjust the relaxed-match threshold and save results:
  python evaluate_track2_starter.py --predictions preds.json \\
      --iou 0.6 --output results.json
"""

import argparse
import copy
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, NamedTuple, Optional

from sklearn.metrics import precision_recall_fscore_support

# ── Constants ─────────────────────────────────────────────────────────────────

SENTENCE_LABELS = ["relevant", "not-relevant"]
ENTITY_TYPES = ["Premise", "Claim"]
RELATION_TYPES = ["Support", "Attack"]  # display order
DEFAULT_IOU = 0.5

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
THIN_RULE = "─" * 68
THICK_RULE = "━" * 68


# ── Tokenisation ──────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[tuple[int, int]]:
    """Return (char_start, char_end) pairs for every word and punctuation token."""
    return [(m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def _token_set(
        token_positions: list[tuple[int, int]],
        start: int,
        end: int,
) -> frozenset[int]:
    """Return the indices of tokens fully contained within the character span [start, end)."""
    return frozenset(
        i for i, (ts, te) in enumerate(token_positions)
        if ts >= start and te <= end
    )


def _token_iou(a: frozenset[int], b: frozenset[int]) -> float:
    """Compute the token-level Jaccard index. Returns 1.0 when both sets are empty."""
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _attach_token_sets(
        entities: list[dict],
        token_positions: list[tuple[int, int]],
) -> list[dict]:
    """Return entity dicts enriched with a 'tokens' frozenset for relaxed span matching."""
    return [
        e | {"tokens": _token_set(token_positions, e["start"], e["end"])}
        for e in entities
    ]


# ── Relation representation ───────────────────────────────────────────────────

class Relation(NamedTuple):
    """Flat representation of an argumentative relation, ready for matching."""
    arg1_start: int
    arg1_end: int
    arg2_start: int
    arg2_end: int
    relation_type: str
    arg1_tokens: frozenset
    arg2_tokens: frozenset


# ── Greedy 1-to-1 matching ────────────────────────────────────────────────────

def _greedy_match(
        gold: list,
        pred: list,
        match_fn: Callable,
) -> tuple[list, list, list]:
    """
    Pair each gold item with the first compatible predicted item (greedy, 1-to-1).

    Each item on either side is used at most once.
    Returns (matched_pairs, unmatched_gold, unmatched_pred).
    """
    used: set[int] = set()
    matched, unmatched_gold = [], []
    for g in gold:
        j = next(
            (j for j, p in enumerate(pred) if j not in used and match_fn(g, p)),
            None,
        )
        if j is not None:
            matched.append((g, pred[j]))
            used.add(j)
        else:
            unmatched_gold.append(g)
    unmatched_pred = [pred[j] for j in range(len(pred)) if j not in used]
    return matched, unmatched_gold, unmatched_pred


# ── Match predicates ──────────────────────────────────────────────────────────

def _strict_span(a: dict, b: dict) -> bool:
    """True when both spans share identical character offsets."""
    return a["start"] == b["start"] and a["end"] == b["end"]


def _strict_typed_span(a: dict, b: dict) -> bool:
    """True when both spans have the same entity type and identical character offsets."""
    return a["type"] == b["type"] and _strict_span(a, b)


def _relaxed_span(a: dict, b: dict, threshold: float) -> bool:
    """True when the token-level IoU between the two spans meets the threshold."""
    return _token_iou(a["tokens"], b["tokens"]) >= threshold


def _relaxed_typed_span(a: dict, b: dict, threshold: float) -> bool:
    """True when entity types match and token-level IoU meets the threshold."""
    return a["type"] == b["type"] and _relaxed_span(a, b, threshold)


def _strict_relation(a: Relation, b: Relation) -> bool:
    """True when both relations share the same type and exact argument offsets."""
    return (
            a.relation_type == b.relation_type
            and a.arg1_start == b.arg1_start and a.arg1_end == b.arg1_end
            and a.arg2_start == b.arg2_start and a.arg2_end == b.arg2_end
    )


def _relaxed_relation(a: Relation, b: Relation, threshold: float) -> bool:
    """True when relation types match and both argument spans meet the IoU threshold."""
    return (
            a.relation_type == b.relation_type
            and _token_iou(a.arg1_tokens, b.arg1_tokens) >= threshold
            and _token_iou(a.arg2_tokens, b.arg2_tokens) >= threshold
    )


# ── Metric helpers ────────────────────────────────────────────────────────────

def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Return (precision, recall, F1) from raw counts. All values are zero-safe."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _prf_dict(tp: int, fp: int, fn: int) -> dict:
    """Return a metric dict with rounded f1/precision/recall and the raw TP/FP/FN counts."""
    precision, recall, f1 = _prf(tp, fp, fn)
    return {
        "f1": round(f1, 4), "precision": round(precision, 4), "recall": round(recall, 4),
        "tp": tp, "fp": fp, "fn": fn,
    }


def _compute_prf_metrics(
        tp_by: dict[str, int],
        fp_by: dict[str, int],
        fn_by: dict[str, int],
        type_order: list[str],
) -> dict:
    """
    Compute per-type, macro-avg, and micro-avg P/R/F1 from pre-accumulated counts.

    type_order controls which labels appear and their display order.
    Macro is the unweighted mean of per-type values; micro aggregates all counts.
    """
    per_type: dict[str, dict] = {}
    f1s, precs, recs = [], [], []
    total_tp = total_fp = total_fn = 0

    for label in type_order:
        tp, fp, fn = tp_by[label], fp_by[label], fn_by[label]
        prec, rec, f1 = _prf(tp, fp, fn)
        per_type[label] = _prf_dict(tp, fp, fn)
        f1s.append(f1);
        precs.append(prec);
        recs.append(rec)
        total_tp += tp;
        total_fp += fp;
        total_fn += fn

    n = len(f1s)
    return {
        "per_type": per_type,
        "macro_avg": {
            "f1": round(sum(f1s) / n, 4) if n else 0.0,
            "precision": round(sum(precs) / n, 4) if n else 0.0,
            "recall": round(sum(recs) / n, 4) if n else 0.0,
        },
        "micro_avg": _prf_dict(total_tp, total_fp, total_fn),
    }


# ── Subtask 1 – Evidence Sentence Detection ───────────────────────────────────

def evaluate_subtask1(cases: list[dict]) -> dict:
    """
    Evaluate binary sentence relevance classification using sklearn.

    Collects sentence-level labels across all cases and computes per-class,
    macro, and micro P/R/F1. Official ranking metric: F1 on 'relevant'.
    """
    y_true, y_pred = [], []
    for case in cases:
        gold = case["annotations"]["sentence_relevancy"]
        pred = case["predictions"]["sentence_relevancy"]
        if len(gold) != len(pred):
            raise ValueError(
                f"Case {case['id']}: sentence_relevancy length mismatch "
                f"(gold={len(gold)}, pred={len(pred)})"
            )
        y_true.extend(gold)
        y_pred.extend(pred)

    prec_per, rec_per, f1_per, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=SENTENCE_LABELS, zero_division=0
    )
    prec_mic, rec_mic, f1_mic, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )

    per_class = {
        label: {
            "f1": round(float(f1_per[i]), 4),
            "precision": round(float(prec_per[i]), 4),
            "recall": round(float(rec_per[i]), 4),
        }
        for i, label in enumerate(SENTENCE_LABELS)
    }

    return {
        "official_score": round(float(f1_per[0]), 4),  # F1 on "relevant"
        "macro_avg": {
            "f1": round(float(f1_per.mean()), 4),
            "precision": round(float(prec_per.mean()), 4),
            "recall": round(float(rec_per.mean()), 4),
        },
        "micro_avg": {
            "f1": round(float(f1_mic), 4),
            "precision": round(float(prec_mic), 4),
            "recall": round(float(rec_mic), 4),
        },
    } | per_class


# ── Subtask 2 – Fine-Grained Evidence Span Detection ─────────────────────────

def _containing_sentence(start: int, end: int, sentences: list[dict]) -> Optional[int]:
    """
    Return the 0-based index of the sentence that fully contains [start, end).

    Falls back to the sentence with the greatest character overlap when no
    sentence fully contains the span.
    """
    for i, s in enumerate(sentences):
        if start >= s["start"] and end <= s["end"]:
            return i
    best_idx, best_overlap = None, 0
    for i, s in enumerate(sentences):
        overlap = max(0, min(end, s["end"]) - max(start, s["start"]))
        if overlap > best_overlap:
            best_overlap, best_idx = overlap, i
    return best_idx


def _filter_by_scope(
        gold: list[dict],
        pred: list[dict],
        oracle: bool,
        relevant_sent_indices: list[int],
        sentences: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Restrict entity lists to the active evaluation scope.

    End-to-end scope: all spans are kept unchanged.
    Oracle scope: only spans in gold-relevant sentences are kept; predictions
    outside those sentences are dropped without FP penalty.
    """
    if oracle:
        def in_scope(e: dict) -> bool:
            return _containing_sentence(e["start"], e["end"], sentences) in relevant_sent_indices

        return [e for e in gold if in_scope(e)], [e for e in pred if in_scope(e)]
    return gold, pred


def evaluate_subtask2(cases: list[dict], threshold: float = DEFAULT_IOU) -> dict:
    """
    Evaluate Premise/Claim span extraction for both scopes and match criteria.

    Counts are accumulated corpus-wide (not per-case) before computing
    per-type, macro, and micro metrics for stability.
    Official ranking metric: Scope A, Strict Micro F1.
    """
    # counts[scope][criterion][entity_type] = {"tp": int, "fp": int, "fn": int}
    counts: dict[str, dict[str, dict[str, dict[str, int]]]] = {
        scope: {
            criterion: {etype: {"tp": 0, "fp": 0, "fn": 0} for etype in ENTITY_TYPES}
            for criterion in ("strict", "relaxed")
        }
        for scope in ("end_to_end", "oracle")
    }

    for case in cases:
        sentences = case["metadata"]["context_sentences"]
        relevant_sent_idx = [
            i for i, label in enumerate(case["annotations"]["sentence_relevancy"])
            if label == "relevant"
        ]
        gold_entities = case["annotations"]["entities"]
        pred_entities = case["predictions"]["entities"]

        for scope, oracle in (("end_to_end", False), ("oracle", True)):
            scoped_gold, scoped_pred = _filter_by_scope(
                gold_entities, pred_entities, oracle, relevant_sent_idx, sentences
            )
            for criterion, match_fn in (
                    ("strict", _strict_typed_span),
                    ("relaxed", lambda a, b, _t=threshold: _relaxed_typed_span(a, b, _t)),
            ):
                matched, unmatched_gold, unmatched_pred = _greedy_match(
                    scoped_gold, scoped_pred, match_fn
                )
                c = counts[scope][criterion]
                for gold_e, _ in matched:    c[gold_e["type"]]["tp"] += 1
                for e in unmatched_pred:     c[e["type"]]["fp"] += 1
                for e in unmatched_gold:     c[e["type"]]["fn"] += 1

    def _scope_metrics(scope: str) -> dict:
        """Build strict and relaxed metric blocks for one evaluation scope."""
        result = {}
        for criterion in ("strict", "relaxed"):
            c = counts[scope][criterion]
            result[criterion] = _compute_prf_metrics(
                {t: c[t]["tp"] for t in ENTITY_TYPES},
                {t: c[t]["fp"] for t in ENTITY_TYPES},
                {t: c[t]["fn"] for t in ENTITY_TYPES},
                ENTITY_TYPES,
            )
        return result

    scope_a = _scope_metrics("end_to_end")
    scope_b = _scope_metrics("oracle")

    return {
        "official_score": scope_a["strict"]["micro_avg"]["f1"],  # Scope A, Strict Micro F1
        "scope_A_end_to_end": scope_a,
        "scope_B_oracle": scope_b,
    }


# ── Subtask 3 – Argumentative Relation Detection ──────────────────────────────

def _build_relations(relations: list[dict], entity_map: dict[str, dict]) -> list[Relation]:
    """
    Convert raw relation dicts to Relation namedtuples using pre-enriched entities.

    Relations whose argument IDs are absent from entity_map are silently skipped.
    """
    result = []
    for rel in relations:
        e1 = entity_map.get(rel["arg1_id"])
        e2 = entity_map.get(rel["arg2_id"])
        if e1 is not None and e2 is not None:
            result.append(Relation(
                e1["start"], e1["end"],
                e2["start"], e2["end"],
                rel["relation_type"],
                e1.get("tokens", frozenset()),
                e2.get("tokens", frozenset()),
            ))
    return result


def evaluate_subtask3(cases: list[dict], threshold: float = DEFAULT_IOU) -> dict:
    """
    Evaluate argumentative relation extraction with strict and relaxed matching.

    Counts are accumulated corpus-wide for stable macro/micro aggregation.
    Official ranking metric: Strict Macro F1.
    """
    # counts[criterion]["tp"/"fp"/"fn"][relation_type] = int
    counts: dict[str, dict[str, defaultdict]] = {
        criterion: {"tp": defaultdict(int), "fp": defaultdict(int), "fn": defaultdict(int)}
        for criterion in ("strict", "relaxed")
    }
    seen_types: set[str] = set()

    for case in cases:
        gold_rels = case["annotations"]["relations"]
        pred_rels = case["predictions"]["relations"]
        seen_types.update(r.relation_type for r in gold_rels + pred_rels)

        for criterion, match_fn in (
                ("strict", _strict_relation),
                ("relaxed", lambda a, b: _relaxed_relation(a, b, threshold)),
        ):
            matched, unmatched_gold, unmatched_pred = _greedy_match(
                gold_rels, pred_rels, match_fn
            )
            c = counts[criterion]
            for gold_r, _ in matched:      c["tp"][gold_r.relation_type] += 1
            for r in unmatched_pred:       c["fp"][r.relation_type] += 1
            for r in unmatched_gold:       c["fn"][r.relation_type] += 1

    # Known types first, unknown extras in alphabetical order
    type_order = sorted(
        seen_types,
        key=lambda t: (RELATION_TYPES.index(t) if t in RELATION_TYPES else 999, t),
    )

    def _criterion_metrics(criterion: str) -> dict:
        """Compute per-type, macro, and micro metrics for one match criterion."""
        c = counts[criterion]
        return _compute_prf_metrics(c["tp"], c["fp"], c["fn"], type_order)

    strict = _criterion_metrics("strict")
    relaxed = _criterion_metrics("relaxed")

    return {
        "official_score": strict["macro_avg"]["f1"],  # Strict Macro F1
        "strict": strict,
        "relaxed": relaxed,
    }


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _load_json_array(path: Path) -> list[dict]:
    """Load a JSON file and assert that its top-level value is an array."""
    files = [path] if path.is_file() else path.glob("**/*.json")
    data = []
    for f in files:
        _data = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(_data, list):
            raise ValueError(f"{f} must contain a JSON array.")
        data.extend(_data)
    return data


def _prepare_cases(predictions_path: Path, gold_path: Optional[Path]) -> list[dict]:
    """
    Assemble a unified list of cases, each with 'annotations' (gold) and 'predictions'.

    Two-file mode: annotations from gold_path, predictions from predictions_path
    matched by case ID.
    Single-file mode: predictions block used when non-empty; otherwise annotations
    are deep-copied as predictions for a self-consistency check.
    """
    pred_cases = _load_json_array(predictions_path)

    if gold_path is not None:
        gold_cases = _load_json_array(gold_path)

        pred_key = 'predictions' if any('predictions' in c for c in pred_cases) else 'annotations'
        pred_by_id = {c["id"]: c.get(pred_key, {}) for c in pred_cases}
        cases = []
        for case in gold_cases:
            pred_block = pred_by_id.get(case["id"])
            if pred_block is None:
                raise ValueError(
                    f"Case '{case['id']}' is in the gold file but missing "
                    f"from the predictions file."
                )
            cases.append(case | {"predictions": pred_block})
        return cases

    def _has_content(block: dict) -> bool:
        return bool(
            block.get("sentence_relevancy") or block.get("entities") or block.get("relations")
        )

    return [c | {
        "predictions": (
            c["predictions"]
            if _has_content(c.get("predictions", {}))
            else copy.deepcopy(c["annotations"])
        ),
    } for c in pred_cases]


def _enrich_cases(cases: list[dict]) -> list[dict]:
    """
    Attach token-index sets to entities and convert relation dicts to Relation tuples.

    Tokenisation is performed once per case from raw_text and reused for both
    the annotations and predictions blocks.
    """
    for case in cases:
        token_positions = _tokenize(case["raw_text"])
        for block_key in ("annotations", "predictions"):
            block = case[block_key]
            block["entities"] = _attach_token_sets(block["entities"], token_positions)
            entity_map = {e["id"]: e for e in block["entities"]}
            block["relations"] = _build_relations(block["relations"], entity_map)
    return cases


# ── Console report ────────────────────────────────────────────────────────────

def _format_row(label: str, metrics: dict, show_counts: bool = False) -> str:
    """Format a single metrics row for console output."""
    row = (
        f"  {label:<22}  "
        f"F1={metrics['f1']:.4f}  "
        f"P={metrics['precision']:.4f}  "
        f"R={metrics['recall']:.4f}"
    )
    if show_counts:
        row += f"  TP={metrics['tp']:4d}  FP={metrics['fp']:4d}  FN={metrics['fn']:4d}"
    return row


def _print_results(results: dict) -> None:
    """Print a formatted summary of the selected subtask results to stdout."""

    # ── Subtask 1 ─────────────────────────────────────────────────────────
    if "subtask1" in results:
        st1 = results["subtask1"]

        print(f"\n{THICK_RULE}")
        print("  SUBTASK 1 – Evidence Sentence Detection")
        print(f"  Official Score  (F1 'relevant'): {st1['official_score']:.4f}")
        print(THICK_RULE)
        print(f"\n  {'Class':<22}  {'F1':>8}  {'Precision':>12}  {'Recall':>10}")
        print(f"  {'-' * 56}")
        for key in (*SENTENCE_LABELS, "macro_avg", "micro_avg"):
            m = st1[key]
            print(f"  {key:<22}  {m['f1']:>8.4f}  {m['precision']:>12.4f}  {m['recall']:>10.4f}")

    # ── Subtask 2 ─────────────────────────────────────────────────────────
    if "subtask2" in results:
        st2 = results["subtask2"]

        print(f"\n{THICK_RULE}")
        print("  SUBTASK 2 – Fine-Grained Evidence Span Detection")
        print(f"  Official Score  (Scope A, Strict Micro F1): {st2['official_score']:.4f}")
        print(THICK_RULE)

        for scope_key, scope_label in (
                ("scope_A_end_to_end", "Scope A – End-to-End  [official]"),
                ("scope_B_oracle", "Scope B – Oracle       [complementary]"),
        ):
            print(f"\n  {scope_label}")
            for criterion_key, criterion_label in (
                    ("strict", "Strict (char-exact)"),
                    ("relaxed", "Relaxed (token IoU≥τ)"),
            ):
                crit = st2[scope_key][criterion_key]
                print(f"\n    {criterion_label}")
                print(
                    f"  {'Entity Type':<22}  {'F1':>8}  {'Precision':>12}  {'Recall':>10}"
                    f"  {'TP':>6}  {'FP':>6}  {'FN':>6}"
                )
                print(f"  {'-' * 72}")
                for etype in ENTITY_TYPES:
                    print(_format_row(etype, crit["per_type"][etype], show_counts=True))
                print(f"  {'-' * 72}")
                print(_format_row("Macro Avg", crit["macro_avg"]))
                print(_format_row("Micro Avg", crit["micro_avg"], show_counts=True))

    # ── Subtask 3 ─────────────────────────────────────────────────────────
    if "subtask3" in results:
        st3 = results["subtask3"]

        print(f"\n{THICK_RULE}")
        print("  SUBTASK 3 – Argumentative Relation Detection")
        print(f"  Official Score  (Strict Macro F1): {st3['official_score']:.4f}")
        print(THICK_RULE)

        for criterion_key, criterion_label in (
                ("strict", "Strict (char-exact)  [official]"),
                ("relaxed", "Relaxed (token IoU≥τ)  [complementary]"),
        ):
            block = st3[criterion_key]
            print(f"\n  {criterion_label}")
            print(
                f"  {'Relation Type':<22}  {'F1':>8}  {'Precision':>12}  {'Recall':>10}"
                f"  {'TP':>6}  {'FP':>6}  {'FN':>6}"
            )
            print(f"  {'-' * 72}")
            for rtype, m in block["per_type"].items():
                print(_format_row(rtype, m, show_counts=True))
            print(f"  {'-' * 72}")
            print(_format_row("Macro Avg", block["macro_avg"]))
            print(_format_row("Micro Avg", block["micro_avg"], show_counts=True))
            if "merged_attack" in block:
                print(_format_row("Merged Attack+PA", block["merged_attack"], show_counts=True))

    print(f"\n{THIN_RULE}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def evaluate(
        predictions_path: Path,
        gold_path: Optional[Path] = None,
        threshold: float = DEFAULT_IOU,
        task: str = "all",
) -> dict:
    """Load cases, enrich with token data, and run selected subtask evaluations."""
    cases = _enrich_cases(_prepare_cases(predictions_path, gold_path))

    results = {}

    if task in ("1", "all"):
        results["subtask1"] = evaluate_subtask1(cases)

    if task in ("2", "all"):
        results["subtask2"] = evaluate_subtask2(cases, threshold)

    if task in ("3", "all"):
        results["subtask3"] = evaluate_subtask3(cases, threshold)

    return results


def main() -> None:
    """Parse command-line arguments and run the selected evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="IberLEF 2026 – GRACE Track 2 Starter Kit Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--predictions", type=Path, required=True, metavar="FILE",
        help="JSON file with system predictions",
    )
    parser.add_argument(
        "--gold", type=Path, default=None, metavar="FILE",
        help="Separate JSON file with gold annotations (optional)",
    )
    parser.add_argument(
        "--iou", type=float, default=DEFAULT_IOU, metavar="τ",
        help=f"Token IoU threshold for relaxed span matching (default: {DEFAULT_IOU})",
    )
    parser.add_argument(
        "--task",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Task to evaluate: 1=S1, 2=S2, 3=S3, all=all subtasks.",
    )
    parser.add_argument(
        "--output", type=Path, default=None, metavar="FILE",
        help="Write full results as JSON to this path or directory (optional)",
    )
    args = parser.parse_args()

    results = evaluate(
        predictions_path=args.predictions,
        gold_path=args.gold,
        threshold=args.iou,
        task=args.task,
    )

    _print_results(results)

    if args.output:
        if args.output.suffix:
            output_file = args.output
            output_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            args.output.mkdir(parents=True, exist_ok=True)
            output_file = args.output / "results.json"

        summary = {}

        if "subtask1" in results:
            summary["Subtask 1: Relevant F1"] = results["subtask1"]["official_score"]

        if "subtask2" in results:
            summary["Subtask 2: Scope A Strict Micro F1"] = results["subtask2"]["official_score"]

        if "subtask3" in results:
            summary["Subtask 3: Strict Macro F1"] = results["subtask3"]["official_score"]

        if summary:
            summary["Overall"] = round(sum(summary.values()) / len(summary), 4)

        output_file.write_text(
            json.dumps(summary | results, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        print(f"  Results saved to: {output_file}\n")

if __name__ == "__main__":
    main()
