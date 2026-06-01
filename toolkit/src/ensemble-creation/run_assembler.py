#!/usr/bin/env python3
"""
Assemble one scorer-ready GRACE prediction file from task-specific prediction files.

Design
------
- S1 file provides predictions.sentence_relevancy.
- S2 file provides predictions.entities and is treated as the source of truth.
- S3 file provides predictions.relations.
- Final relations are validated/remapped so arg1_id / arg2_id refer to entities
  present in the final S2 entity block.

Typical usage
-------------
python3 run_assembler.py \
  --s1 ensemble_s1_best.json \
  --s2 ensemble_s2_best.json \
  --s3 ensemble_s3_best.json \
  --output final_submission.json \
  --save-report assembly_report.json
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array.")
    return data


def has_prediction_content(block: Dict[str, Any]) -> bool:
    return bool(
        block.get("sentence_relevancy")
        or block.get("entities")
        or block.get("relations")
    )


def get_prediction_block(case: Dict[str, Any]) -> Dict[str, Any]:
    pred = case.get("predictions", {})
    if isinstance(pred, dict) and has_prediction_content(pred):
        return pred

    ann = case.get("annotations", {})
    if isinstance(ann, dict):
        return ann

    return {}


def index_by_id(cases: List[Dict[str, Any]], path: Path) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}

    for case in cases:
        case_id = case.get("id")
        if not case_id:
            raise ValueError(f"{path}: found a case without an id.")

        if case_id in indexed:
            raise ValueError(f"{path}: duplicate case id '{case_id}'.")

        indexed[case_id] = case

    return indexed


def load_cases(path: Path) -> Dict[str, Dict[str, Any]]:
    return index_by_id(load_json_array(path), path)


def assert_same_structure(
    case_id: str,
    reference: Dict[str, Any],
    other: Dict[str, Any],
    source_name: str,
) -> None:
    if reference.get("id") != other.get("id"):
        raise ValueError(f"{case_id}: id mismatch in {source_name}.")

    if reference.get("raw_text") != other.get("raw_text"):
        raise ValueError(f"{case_id}: raw_text differs in {source_name}.")

    if reference.get("metadata") != other.get("metadata"):
        raise ValueError(f"{case_id}: metadata differs in {source_name}.")


def assert_same_case_set(
    reference_ids: List[str],
    source_cases: Dict[str, Dict[str, Any]],
    source_name: str,
) -> None:
    reference_set = set(reference_ids)
    source_set = set(source_cases)

    missing = sorted(reference_set - source_set)
    extra = sorted(source_set - reference_set)

    if missing:
        raise ValueError(f"{source_name}: missing case ids: {missing[:10]}")

    if extra:
        raise ValueError(f"{source_name}: extra case ids: {extra[:10]}")


def entity_key(entity: Dict[str, Any]) -> Tuple[int, int, str]:
    return (
        int(entity["start"]),
        int(entity["end"]),
        str(entity["type"]),
    )


def validate_unique_entity_ids(case_id: str, entities: List[Dict[str, Any]]) -> None:
    seen = set()

    for ent in entities:
        ent_id = ent.get("id")
        if not ent_id:
            raise ValueError(f"{case_id}: entity without id.")

        if ent_id in seen:
            raise ValueError(f"{case_id}: duplicate final entity id '{ent_id}'.")

        seen.add(ent_id)


def build_final_entity_lookup(
    entities: List[Dict[str, Any]],
) -> Tuple[set[str], Dict[Tuple[int, int, str], List[Dict[str, Any]]]]:
    ids = {str(ent["id"]) for ent in entities}

    by_span = defaultdict(list)
    for ent in entities:
        by_span[entity_key(ent)].append(ent)

    return ids, by_span


def remap_relation_arg(
    arg_id: str,
    final_entity_ids: set[str],
    s3_entities_by_id: Dict[str, Dict[str, Any]],
    final_entities_by_span: Dict[Tuple[int, int, str], List[Dict[str, Any]]],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (mapped_id, error_reason).

    If arg_id already exists in final S2 entities, no remapping is needed.
    Otherwise, try to map the S3 source entity to a final S2 entity using
    exact start/end/type.
    """

    if arg_id in final_entity_ids:
        return arg_id, None

    source_entity = s3_entities_by_id.get(arg_id)
    if source_entity is None:
        return None, f"S3 relation references unknown S3 entity id '{arg_id}'"

    key = entity_key(source_entity)
    matches = final_entities_by_span.get(key, [])

    if not matches:
        return None, (
            "No final S2 entity matches S3 entity "
            f"id='{arg_id}', span=({key[0]}, {key[1]}), type='{key[2]}'"
        )

    # Deterministic fallback if duplicated spans exist.
    matches = sorted(matches, key=lambda ent: str(ent["id"]))
    return str(matches[0]["id"]), None


def assemble_relations_for_case(
    case_id: str,
    final_entities: List[Dict[str, Any]],
    s3_predictions: Dict[str, Any],
    drop_unmapped_relations: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    validate_unique_entity_ids(case_id, final_entities)

    final_entity_ids, final_entities_by_span = build_final_entity_lookup(final_entities)

    s3_entities = s3_predictions.get("entities", [])
    s3_entities_by_id = {
        str(ent["id"]): ent
        for ent in s3_entities
        if "id" in ent
    }

    raw_relations = s3_predictions.get("relations", [])

    output_relations: List[Dict[str, Any]] = []
    seen_relation_keys = set()

    report = {
        "relations_in": len(raw_relations),
        "relations_out": 0,
        "dropped": [],
        "deduplicated": 0,
    }

    for rel in raw_relations:
        relation_type = rel.get("relation_type")
        arg1_id = rel.get("arg1_id")
        arg2_id = rel.get("arg2_id")

        if not relation_type or not arg1_id or not arg2_id:
            reason = "Relation is missing relation_type, arg1_id or arg2_id."
            if drop_unmapped_relations:
                report["dropped"].append({"relation": rel, "reason": reason})
                continue
            raise ValueError(f"{case_id}: {reason}")

        mapped_arg1, err1 = remap_relation_arg(
            str(arg1_id),
            final_entity_ids,
            s3_entities_by_id,
            final_entities_by_span,
        )
        mapped_arg2, err2 = remap_relation_arg(
            str(arg2_id),
            final_entity_ids,
            s3_entities_by_id,
            final_entities_by_span,
        )

        if err1 or err2:
            reason = "; ".join(err for err in [err1, err2] if err)

            if drop_unmapped_relations:
                report["dropped"].append({"relation": rel, "reason": reason})
                continue

            raise ValueError(f"{case_id}: cannot map S3 relation: {reason}")

        if mapped_arg1 == mapped_arg2:
            reason = f"Relation maps both arguments to the same final entity '{mapped_arg1}'."

            if drop_unmapped_relations:
                report["dropped"].append({"relation": rel, "reason": reason})
                continue

            raise ValueError(f"{case_id}: {reason}")

        relation_key = (mapped_arg1, mapped_arg2, str(relation_type))

        if relation_key in seen_relation_keys:
            report["deduplicated"] += 1
            continue

        seen_relation_keys.add(relation_key)

        output_relations.append({
            "id": f"asm_r{len(output_relations) + 1}",
            "arg1_id": mapped_arg1,
            "arg2_id": mapped_arg2,
            "relation_type": str(relation_type),
        })

    report["relations_out"] = len(output_relations)
    return output_relations, report


def assemble_case(
    case_id: str,
    base_case: Dict[str, Any],
    s1_case: Dict[str, Any],
    s2_case: Dict[str, Any],
    s3_case: Dict[str, Any],
    drop_unmapped_relations: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    s1_predictions = get_prediction_block(s1_case)
    s2_predictions = get_prediction_block(s2_case)
    s3_predictions = get_prediction_block(s3_case)

    sentence_relevancy = copy.deepcopy(
        s1_predictions.get("sentence_relevancy", [])
    )
    entities = copy.deepcopy(
        s2_predictions.get("entities", [])
    )

    relations, relation_report = assemble_relations_for_case(
        case_id=case_id,
        final_entities=entities,
        s3_predictions=s3_predictions,
        drop_unmapped_relations=drop_unmapped_relations,
    )

    output_case = copy.deepcopy(base_case)
    output_case["predictions"] = {
        "sentence_relevancy": sentence_relevancy,
        "entities": entities,
        "relations": relations,
    }

    report = {
        "case_id": case_id,
        "sentence_relevancy_count": len(sentence_relevancy),
        "entity_count": len(entities),
        "relation_report": relation_report,
    }

    return output_case, report


def assemble_run(
    s1_path: Path,
    s2_path: Path,
    s3_path: Path,
    output_path: Path,
    reference_path: Optional[Path],
    save_report_path: Optional[Path],
    drop_unmapped_relations: bool,
) -> None:
    s1_cases = load_cases(s1_path)
    s2_cases = load_cases(s2_path)
    s3_cases = load_cases(s3_path)

    if reference_path is not None:
        reference_cases = load_cases(reference_path)
        base_source = str(reference_path)
    else:
        # S2 is the safest default because final entities come from S2.
        reference_cases = s2_cases
        base_source = str(s2_path)

    case_order = list(reference_cases.keys())

    assert_same_case_set(case_order, s1_cases, "S1 file")
    assert_same_case_set(case_order, s2_cases, "S2 file")
    assert_same_case_set(case_order, s3_cases, "S3 file")

    output_cases = []
    case_reports = []

    for case_id in case_order:
        base_case = reference_cases[case_id]

        assert_same_structure(case_id, base_case, s1_cases[case_id], "S1 file")
        assert_same_structure(case_id, base_case, s2_cases[case_id], "S2 file")
        assert_same_structure(case_id, base_case, s3_cases[case_id], "S3 file")

        output_case, case_report = assemble_case(
            case_id=case_id,
            base_case=base_case,
            s1_case=s1_cases[case_id],
            s2_case=s2_cases[case_id],
            s3_case=s3_cases[case_id],
            drop_unmapped_relations=drop_unmapped_relations,
        )

        output_cases.append(output_case)
        case_reports.append(case_report)

    output_path.write_text(
        json.dumps(output_cases, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if save_report_path is not None:
        report = {
            "run_config": {
                "s1": str(s1_path),
                "s2": str(s2_path),
                "s3": str(s3_path),
                "reference": base_source,
                "output": str(output_path),
                "drop_unmapped_relations": drop_unmapped_relations,
            },
            "case_reports": case_reports,
        }

        save_report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble task-specific GRACE prediction files into one final scorer-ready file."
    )

    parser.add_argument(
        "--s1",
        type=Path,
        required=True,
        help="GRACE JSON file providing predictions.sentence_relevancy.",
    )
    parser.add_argument(
        "--s2",
        type=Path,
        required=True,
        help="GRACE JSON file providing final predictions.entities.",
    )
    parser.add_argument(
        "--s3",
        type=Path,
        required=True,
        help="GRACE JSON file providing predictions.relations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to final assembled GRACE JSON output.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help=(
            "Optional structural reference file. If omitted, the S2 file is used "
            "as the base because S2 entities are authoritative."
        ),
    )
    parser.add_argument(
        "--save-report",
        type=Path,
        default=None,
        help="Optional JSON report with relation remapping/dropping details.",
    )
    parser.add_argument(
        "--drop-unmapped-relations",
        action="store_true",
        help=(
            "Drop S3 relations whose arguments cannot be mapped to final S2 entities. "
            "By default, the script fails loudly instead."
        ),
    )

    args = parser.parse_args()

    assemble_run(
        s1_path=args.s1,
        s2_path=args.s2,
        s3_path=args.s3,
        output_path=args.output,
        reference_path=args.reference,
        save_report_path=args.save_report,
        drop_unmapped_relations=args.drop_unmapped_relations,
    )


if __name__ == "__main__":
    main()
