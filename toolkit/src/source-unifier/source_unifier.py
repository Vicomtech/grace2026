#!/usr/bin/env python3
"""
Unify GRACE and CASIMEDICOS datasets into the GRACE JSON format.

Notes
- Text is preserved as-is from the source data as much as possible.
- CASIMEDICOS rationale is included by default because some relations point to explanation sentences.
- CASIMEDICOS sentence relevancy is derived from entities that participate in valid relations.
- Any relation type is kept as long as its endpoints can be matched to entities.
- The script expects:
    1) a GRACE JSON file (list[dict])
    2) a CASIMEDICOS main file (JSONL: one JSON object per line, can also be JSON)
    3) a CASIMEDICOS relations file (JSONL: one JSON object per line, can also be JSON)

Example:
    python3 source_unifier.py \
        --grace grace_test_data.json \
        --casimedicos-main casimedicos_test_all.jsonl \
        --casimedicos-relations casimedicos_test_relations.jsonl \
        --output unified.json
"""

from __future__ import annotations

import argparse
import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# No es necesario establecer un placeholder porque cambiamos la estrategia para anotar la s1
# RELEVANCY_PLACEHOLDER = "unknown" 
DEFAULT_INCLUDE_CASIMEDICOS_RATIONALE = True


@dataclass
class SpanEntity:
    id: str
    text: str
    start: int
    end: int
    type: str


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl_or_json_objects(path: Path) -> List[Dict[str, Any]]:
    """
    Loads files that may be either:
    - standard JSON array/object
    - JSONL / one-JSON-object-per-line
    - concatenated JSON objects, one per line
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    records: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def is_valid_casimedicos_record(record: Dict[str, Any]) -> bool:
    return isinstance(record.get("text"), list) and isinstance(record.get("labels"), list)


def normalize_relation_text(text: str) -> str:
    text = text.replace("\n", " ").strip()
    text = " ".join(text.split())

    # remove surrounding quotes/brackets/punctuation
    text = text.strip(" \t\r\n\"'“”‘’()[]{}<>.,;:!?")

    # remove repeated trailing punctuation after stripping
    text = re.sub(r"[.。,;:!?]+$", "", text)

    # remove repeated leading punctuation too
    text = re.sub(r"^[.。,;:!?()\\[\\]{}<>]+", "", text)

    return text


def tokenize_like_source(sentence_tokens: List[str]) -> str:
    """
    Reconstructs sentence text without linguistic preprocessing.
    We preserve token order and simply join with spaces because the source
    stores tokens already split.
    """
    return " ".join(sentence_tokens)

def find_entity_candidates_with_fallback(
    normalized_text: str,
    ent_index: Dict[str, List[SpanEntity]],
) -> List[SpanEntity]:
    # 1) exact match first
    exact = ent_index.get(normalized_text, [])
    if exact:
        return exact

    # 2) partial match fallback
    partial_matches: List[SpanEntity] = []
    for ent_norm, ents in ent_index.items():
        if normalized_text in ent_norm or ent_norm in normalized_text:
            partial_matches.extend(ents)

    # 3) prefer longer entities first (usually more specific)
    partial_matches = sorted(
        partial_matches,
        key=lambda e: (-len(normalize_relation_text(e.text)), e.start, e.end)
    )

    return partial_matches


def build_raw_text_and_offsets(sentences_tokens: List[List[str]]) -> Tuple[str, List[Dict[str, Any]]]:
    raw_parts: List[str] = []
    sentence_objs: List[Dict[str, Any]] = []
    cursor = 0

    for idx, tokens in enumerate(sentences_tokens):
        sentence_text = tokenize_like_source(tokens)
        start = cursor
        end = start + len(sentence_text)
        sentence_objs.append({
            "sentence": sentence_text,
            "start": start,
            "end": end,
        })
        raw_parts.append(sentence_text)
        cursor = end
        if idx < len(sentences_tokens) - 1:
            raw_parts.append("\n")
            cursor += 1

    return "".join(raw_parts), sentence_objs


def extract_correct_choice_id(sentences_tokens: List[List[str]]) -> Optional[str]:
    for tokens in sentences_tokens:
        if len(tokens) >= 3 and tokens[0] == "CORRECT" and tokens[1] == "ANSWER:":
            return tokens[2]
    return None


def is_choice_sentence(tokens: List[str]) -> bool:
    return bool(tokens) and re.fullmatch(r"\d+-", tokens[0]) is not None


def is_correct_answer_sentence(tokens: List[str]) -> bool:
    return len(tokens) >= 2 and tokens[0] == "CORRECT" and tokens[1] == "ANSWER:"


def is_header_sentence(tokens: List[str]) -> bool:
    if not tokens:
        return False
    joined = " ".join(tokens)
    return (
        (len(tokens) >= 2 and tokens[0] == "QUESTION" and tokens[1] == "TYPE:")
        or joined == "CLINICAL CASE:"
    )


def extract_choice_text(tokens: List[str]) -> Tuple[str, str]:
    choice_id = tokens[0].rstrip("-")
    choice_text = " ".join(tokens[1:])
    return choice_id, choice_text


def collect_choice_offsets(raw_text: str, sentences_tokens: List[List[str]]) -> List[Dict[str, Any]]:
    choices: List[Dict[str, Any]] = []
    search_from = 0

    for tokens in sentences_tokens:
        if not is_choice_sentence(tokens):
            continue
        choice_id, choice_text = extract_choice_text(tokens)
        start = raw_text.find(choice_text, search_from)
        if start == -1:
            raise ValueError(f"Could not locate choice text in raw_text: {choice_text!r}")
        end = start + len(choice_text)
        choices.append({
            "id": choice_id,
            "text": choice_text,
            "start": start,
            "end": end,
        })
        search_from = end
    return choices


def bio_tags_to_entities(raw_text: str, sentences_tokens: List[List[str]], labels: List[List[str]]) -> List[SpanEntity]:
    entities: List[SpanEntity] = []
    cursor = 0

    for sent_idx, (tokens, sent_labels) in enumerate(zip(sentences_tokens, labels)):
        if len(tokens) != len(sent_labels):
            raise ValueError(
                f"Token/label length mismatch in sentence {sent_idx}: "
                f"{len(tokens)} tokens vs {len(sent_labels)} labels"
            )

        sentence_text = tokenize_like_source(tokens)
        sentence_start = raw_text.find(sentence_text, cursor)
        if sentence_start == -1:
            raise ValueError(f"Could not locate sentence in raw_text: {sentence_text!r}")

        token_positions: List[Tuple[int, int]] = []
        local_pos = 0
        for token in tokens:
            start = sentence_start + local_pos
            end = start + len(token)
            token_positions.append((start, end))
            local_pos += len(token)
            if local_pos < len(sentence_text):
                local_pos += 1

        i = 0
        while i < len(tokens):
            label = sent_labels[i]
            if label.startswith("B-"):
                ent_type = label[2:]
                start_token = i
                end_token = i
                i += 1
                while i < len(tokens) and sent_labels[i] == f"I-{ent_type}":
                    end_token = i
                    i += 1
                start = token_positions[start_token][0]
                end = token_positions[end_token][1]
                text = raw_text[start:end]
                entities.append(
                    SpanEntity(
                        id=uuid.uuid4().hex[:10],
                        text=text,
                        start=start,
                        end=end,
                        type=ent_type,
                    )
                )
            else:
                i += 1

        cursor = sentence_start + len(sentence_text)

    return entities


def remap_claim_ids_to_choice_ids(entities: List[SpanEntity], choices: List[Dict[str, Any]]) -> List[SpanEntity]:
    """
    In GRACE, claims normally reuse the choice ids ("1", "2", ...).
    We align CASIMEDICOS claims to those ids whenever exact span match exists.
    """
    choice_by_span = {
        (choice["start"], choice["end"], choice["text"]): choice["id"]
        for choice in choices
    }

    remapped: List[SpanEntity] = []
    for ent in entities:
        if ent.type == "Claim":
            choice_id = choice_by_span.get((ent.start, ent.end, ent.text))
            if choice_id is not None:
                remapped.append(SpanEntity(choice_id, ent.text, ent.start, ent.end, ent.type))
                continue
        remapped.append(ent)
    return remapped


def build_relation_lookup(rel_records: List[Dict[str, Any]]) -> Dict[str, List[List[str]]]:
    merged: Dict[str, List[List[str]]] = {}
    for obj in rel_records:
        for key, value in obj.items():
            merged[key] = value
    return merged


def build_text_to_entities_index(entities: List[SpanEntity]) -> Dict[str, List[SpanEntity]]:
    index: Dict[str, List[SpanEntity]] = {}
    for ent in entities:
        norm = normalize_relation_text(ent.text)
        index.setdefault(norm, []).append(ent)
    return index


def choose_relation_endpoint(candidates: List[SpanEntity], preferred_type: Optional[str] = None) -> SpanEntity:
    if preferred_type is not None:
        typed = [c for c in candidates if c.type == preferred_type]
        if typed:
            return sorted(typed, key=lambda x: (x.start, x.end))[0]
    return sorted(candidates, key=lambda x: (x.start, x.end))[0]


def sentence_belongs_to_context(tokens: List[str]) -> bool:
    """
    GRACE-style context should contain only the case/question stem.
    Exclude:
    - headers
    - answer choices
    - CORRECT ANSWER line
    - rationale/explanation after correct answer
    """
    if is_header_sentence(tokens):
        return False
    if is_choice_sentence(tokens):
        return False
    if is_correct_answer_sentence(tokens):
        return False
    return True


def build_context_from_raw(
    all_sentence_objs: List[Dict[str, Any]],
    all_sentences_tokens: List[List[str]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Build GRACE-like metadata.context and metadata.context_sentences
    from only the stem/context/question sentences before choices.
    Offsets remain relative to raw_text.
    """
    context_sentences: List[Dict[str, Any]] = []
    for sent_obj, tokens in zip(all_sentence_objs, all_sentences_tokens):
        if is_choice_sentence(tokens) or is_correct_answer_sentence(tokens):
            break
        if sentence_belongs_to_context(tokens):
            context_sentences.append(sent_obj)

    context = "\n".join(sent["sentence"] for sent in context_sentences)
    return context, context_sentences

def compute_sentence_relevancy_from_relations(
    context_sentences: List[Dict[str, Any]],
    entities: List[SpanEntity],
    relations: List[Dict[str, Any]],
) -> List[str]:
    """
    A context sentence is relevant if:
    - it contains at least one Premise/Claim entity
    - and that entity participates in at least one valid relation.

    Otherwise, it is not-relevant.
    """
    related_entity_ids = set()

    for rel in relations:
        related_entity_ids.add(rel["arg1_id"])
        related_entity_ids.add(rel["arg2_id"])

    sentence_relevancy: List[str] = []

    for sent in context_sentences:
        sent_start = sent["start"]
        sent_end = sent["end"]

        is_relevant = False

        for ent in entities:
            if ent.type not in {"Premise", "Claim"}:
                continue

            entity_inside_sentence = (
                ent.start >= sent_start
                and ent.end <= sent_end
            )

            entity_has_relation = ent.id in related_entity_ids

            if entity_inside_sentence and entity_has_relation:
                is_relevant = True
                break

        sentence_relevancy.append(
            "relevant" if is_relevant else "not-relevant"
        )

    return sentence_relevancy


def convert_casimedicos_record(
    record: Dict[str, Any],
    relation_lookup: Dict[str, List[List[str]]],
    include_rationale: bool = DEFAULT_INCLUDE_CASIMEDICOS_RATIONALE,
) -> Dict[str, Any]:
    record_id = record["id"]
    sentences_tokens = record["text"]
    labels = record["labels"]

    if not isinstance(sentences_tokens, list) or not isinstance(labels, list):
        raise ValueError(
            f"Record {record_id} has invalid text/labels format. "
            f"Expected list-of-lists, got text={type(sentences_tokens).__name__}, "
            f"labels={type(labels).__name__}"
        )

    if include_rationale:
        selected_sentences_tokens = sentences_tokens
        selected_labels = labels
    else:
        selected_sentences_tokens = []
        selected_labels = []
        for sent_tokens, sent_labels in zip(sentences_tokens, labels):
            selected_sentences_tokens.append(sent_tokens)
            selected_labels.append(sent_labels)
            if is_correct_answer_sentence(sent_tokens):
                break

    raw_text, all_sentence_objs = build_raw_text_and_offsets(selected_sentences_tokens)
    correct_choice_id = extract_correct_choice_id(selected_sentences_tokens)
    choices = collect_choice_offsets(raw_text, selected_sentences_tokens)

    rel_items = relation_lookup.get(record_id, [])

    entities = bio_tags_to_entities(raw_text, selected_sentences_tokens, selected_labels)
    entities = remap_claim_ids_to_choice_ids(entities, choices)

    ent_index = build_text_to_entities_index(entities)

    relations: List[Dict[str, Any]] = []
    for premise_text, claim_text, rel_type in rel_items:
        premise_norm = normalize_relation_text(premise_text)
        claim_norm = normalize_relation_text(claim_text)

        premise_candidates = find_entity_candidates_with_fallback(premise_norm, ent_index)
        claim_candidates = find_entity_candidates_with_fallback(claim_norm, ent_index)

        if not premise_candidates:
            print(
                f"[WARN] Skipping relation in record {record_id}: "
                f"could not match premise {premise_text!r}"
            )
            continue

        if not claim_candidates:
            print(
                f"[WARN] Skipping relation in record {record_id}: "
                f"could not match claim {claim_text!r}"
            )
            continue

        source_ent = choose_relation_endpoint(premise_candidates, preferred_type="Premise")
        target_ent = choose_relation_endpoint(claim_candidates, preferred_type="Claim")

        relations.append({
            "id": uuid.uuid4().hex[:8],
            "relation_type": rel_type,
            "arg1_id": source_ent.id,
            "arg2_id": target_ent.id,
        })

    context, context_sentences = build_context_from_raw(all_sentence_objs, selected_sentences_tokens)

    # Parcheamos para que la relevancy se saque a partir de las entidades y relaciones
    sentence_relevancy = compute_sentence_relevancy_from_relations(
    context_sentences=context_sentences,
    entities=entities,
    relations=relations)

    grace_record = {
        "id": record_id,
        "origin": "CASIMEDICOS",
        "raw_text": raw_text,
        "metadata": {
            "context": context,
            "context_sentences": context_sentences,
            "choices": choices,
            "correct_choice_id": correct_choice_id,
        },
        "annotations": {
            "sentence_relevancy": sentence_relevancy,
            "entities": [
                {
                    "id": ent.id,
                    "text": ent.text,
                    "start": ent.start,
                    "end": ent.end,
                    "type": ent.type,
                }
                for ent in entities
            ],
            "relations": relations,
        },
        "predictions": {
            "sentence_relevancy": [],
            "entities": [],
            "relations": [],
        },
    }
    return grace_record


def validate_grace_record(record: Dict[str, Any]) -> None:
    raw_text = record["raw_text"]
    origin = record.get("origin", "grace")

    for sent in record["metadata"]["context_sentences"]:
        assert raw_text[sent["start"]:sent["end"]] == sent["sentence"], (
            "Context sentence offsets do not match text",
            record["id"],
            sent,
        )

    for choice in record["metadata"]["choices"]:
        assert raw_text[choice["start"]:choice["end"]] == choice["text"], (
            "Choice offsets do not match text",
            record["id"],
            choice,
        )

    for ent in record["annotations"]["entities"]:
        assert raw_text[ent["start"]:ent["end"]] == ent["text"], (
            "Entity offsets do not match text",
            record["id"],
            ent,
        )

    context_sentences = record["metadata"]["context_sentences"]
    sentence_relevancy = record["annotations"]["sentence_relevancy"]
    assert len(context_sentences) == len(sentence_relevancy), (
        "sentence_relevancy length must match context_sentences length",
        record["id"],
    )

    entity_by_id = {ent["id"]: ent for ent in record["annotations"]["entities"]}
    for rel in record["annotations"]["relations"]:
        assert rel["arg1_id"] in entity_by_id, (
            "Relation arg1_id missing entity",
            record["id"],
            rel,
        )
        assert rel["arg2_id"] in entity_by_id, (
            "Relation arg2_id missing entity",
            record["id"],
            rel,
        )

        # Only enforce Premise -> Claim for original GRACE records
        if origin == "grace":
            assert entity_by_id[rel["arg1_id"]]["type"] == "Premise", (
                "Relation arg1 must be Premise",
                record["id"],
                rel,
            )
            assert entity_by_id[rel["arg2_id"]]["type"] == "Claim", (
                "Relation arg2 must be Claim",
                record["id"],
                rel,
            )


def unify_datasets(
    grace_path: Path,
    casimedicos_main_path: Path,
    casimedicos_relations_path: Path,
    output_path: Path,
    include_casimedicos_rationale: bool = DEFAULT_INCLUDE_CASIMEDICOS_RATIONALE,
) -> None:
    grace_data = load_json(grace_path)
    if not isinstance(grace_data, list):
        raise ValueError("GRACE file must be a JSON list.")
    
    for record in grace_data:
        record["origin"] = record.get("origin", "GRACE")

    casimedicos_records = load_jsonl_or_json_objects(casimedicos_main_path)
    casimedicos_rel_records = load_jsonl_or_json_objects(casimedicos_relations_path)
    relation_lookup = build_relation_lookup(casimedicos_rel_records)

    valid_casimedicos_records = []
    for record in casimedicos_records:
        if is_valid_casimedicos_record(record):
            valid_casimedicos_records.append(record)
        else:
            print(f"[WARN] Skipping malformed CASIMEDICOS record: {record.get('id')}")

    converted_casimedicos = [
        convert_casimedicos_record(
            record,
            relation_lookup,
            include_rationale=include_casimedicos_rationale,
        )
        for record in valid_casimedicos_records
    ]

    unified = list(grace_data) + converted_casimedicos

    for record in unified:
        validate_grace_record(record)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(unified, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unify datasets into GRACE format")
    parser.add_argument("--grace", required=True, type=Path, help="Path to GRACE JSON file")
    parser.add_argument(
        "--casimedicos-main",
        required=True,
        type=Path,
        help="Path to CASIMEDICOS main JSONL file",
    )
    parser.add_argument(
        "--casimedicos-relations",
        required=True,
        type=Path,
        help="Path to CASIMEDICOS relations JSONL file",
    )
    parser.add_argument("--output", required=True, type=Path, help="Path to output unified JSON")
    parser.add_argument(
        "--exclude-casimedicos-rationale",
        action="store_true",
        help="If set, truncate CASIMEDICOS records at the 'CORRECT ANSWER:' line.",
    )
    args = parser.parse_args()

    unify_datasets(
        grace_path=args.grace,
        casimedicos_main_path=args.casimedicos_main,
        casimedicos_relations_path=args.casimedicos_relations,
        output_path=args.output,
        include_casimedicos_rationale=not args.exclude_casimedicos_rationale,
    )


if __name__ == "__main__":
    main()
