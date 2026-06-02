# src/casimedicos/splits.py
# ------------------------------------------------------------------------------------------------------
# multilingual split generation for CasiMedicos-Arg
# ------------------------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# apr-2026

import json
import random
import logging
from pathlib import Path
from itertools import combinations
import shutil
from typing import List

class SplitGenerator:
    def __init__(self, raw_dir: Path, relations_dir: Path, splits_dir: Path, all_langs: List[str]):
        self.raw_dir = raw_dir
        self.relations_dir = relations_dir
        self.splits_dir = splits_dir
        self.all_langs = all_langs
        
        # into casimedicos/splits directory
        self.splits = ["train", "dev", "test"]

    def generate_splits(self):
        """Monolingual and multilingual combination generation of the CasiMedicos data splits."""
        if self.splits_dir.exists():
            shutil.rmtree(self.splits_dir)

        for split in self.splits:
            (self.splits_dir / split).mkdir(parents=True, exist_ok=True)

        self._generate_mono()
        self._generate_multi()
        logging.info(">>> Split generation complete")

    # --- monolingual raw/ CasiMedicos data [both _ordered.jsonl and _relations.jsonl] into splits/ ---

    def _generate_mono(self):
        """Transfers source files from raw/ and relations/ into the unified splits/ directory."""
        import json
        for lang in self.all_langs:
            for split in self.splits:
                split_target_dir = self.splits_dir / split
                split_name = "validation" if split == "dev" else split

                ordered_tgt = split_target_dir / f"{split}_{lang}_ordered.jsonl"
                relations_tgt = split_target_dir / f"{split}_{lang}_relations.jsonl"
                
                # ** path resolution > jsonl or json **
                ordered_src_options = [
                    self.raw_dir / lang / f"{split_name}_{lang}_ordered.jsonl",
                    self.raw_dir / f"{split_name}_{lang}_ordered.jsonl",
                    self.raw_dir / lang / f"{split_name}_ordered.jsonl",
                    self.raw_dir / lang / f"{split_name}_{lang}_ordered.json",
                    self.raw_dir / f"{split_name}_{lang}_ordered.json",
                    self.raw_dir / lang / f"{split_name}_ordered.json"
                ]
                ordered_src = next((path for path in ordered_src_options if path.exists()), None)

                if ordered_src:
                    valid_lines = []
                    with open(ordered_src, "r", encoding="utf-8") as f:
                        if ordered_src.suffix == ".json":
                            # [ json format ]
                            try:
                                data = json.load(f)
                                if isinstance(data, dict):
                                    if "id" in data: data = [data]
                                    else: data = [{k: v} for k, v in data.items()]
                                elif not isinstance(data, list):
                                    data = []
                                
                                # [ normalize > jsonl]
                                for line_num, record in enumerate(data, 1):
                                    line_str = json.dumps(record, ensure_ascii=False)
                                    normalized_line = self._normalize_ordered(line_str, lang, ordered_src.name, line_num)
                                    if normalized_line:
                                        valid_lines.append(normalized_line)
                            except json.JSONDecodeError:
                                logging.warning(f"\t(!) > Failed to parse JSON file: {ordered_src.name}")
                        else:
                            # [ jsonl format ]
                            for line_num, line in enumerate(f, 1):
                                normalized_line = self._normalize_ordered(line, lang, ordered_src.name, line_num)
                                if normalized_line:
                                    valid_lines.append(normalized_line)
                    
                    if valid_lines:
                        with open(ordered_tgt, "w", encoding="utf-8") as f:
                            f.writelines(valid_lines)
                    else:
                        logging.warning(f"\t(!) > All records dropped in {ordered_src.name}")
                else:
                    logging.warning(f"\t(!) > Missing monolingual ordered source for: {lang} ({split})")
                
                # ** path resolution > jsonl or json [relations] **
                relations_src_options = [
                    self.relations_dir / f"{split_name}_{lang}_relations.jsonl",
                    self.relations_dir / lang / f"{split_name}_relations.jsonl",
                    self.relations_dir / lang / f"{split_name}_{lang}_relations.jsonl",
                    self.relations_dir / f"{split_name}_{lang}_relations.json",
                    self.relations_dir / lang / f"{split_name}_relations.json",
                    self.relations_dir / lang / f"{split_name}_{lang}_relations.json"
                ]
                relations_src = next((path for path in relations_src_options if path.exists()), None)

                if relations_src:
                    valid_rels = []
                    with open(relations_src, "r", encoding="utf-8") as f:
                        if relations_src.suffix == ".json":
                            try:
                                data = json.load(f)
                                if isinstance(data, dict):
                                    # {"case_id": [rels]}
                                    for line_num, (k, v) in enumerate(data.items(), 1):
                                        line_str = json.dumps({k: v}, ensure_ascii=False)
                                        normalized_rel = self._normalize_relations(line_str, lang, relations_src.name, line_num)
                                        if normalized_rel:
                                            valid_rels.append(normalized_rel)
                                elif isinstance(data, list):
                                    for line_num, record in enumerate(data, 1):
                                        line_str = json.dumps(record, ensure_ascii=False)
                                        normalized_rel = self._normalize_relations(line_str, lang, relations_src.name, line_num)
                                        if normalized_rel:
                                            valid_rels.append(normalized_rel)
                            except json.JSONDecodeError:
                                logging.warning(f"\t(!) > Failed to parse JSON file: {relations_src.name}")
                        else:
                            for line_num, line in enumerate(f, 1):
                                normalized_rel = self._normalize_relations(line, lang, relations_src.name, line_num)
                                if normalized_rel:
                                    valid_rels.append(normalized_rel)
                                
                    if valid_rels:
                        with open(relations_tgt, "w", encoding="utf-8") as f:
                            f.writelines(valid_rels)
                    else:
                        logging.warning(f"\t(!) > All records dropped in {relations_src.name}")
                else:
                    logging.warning(f"\t(!) > Missing monolingual relations source for: {lang} ({split})")
                    
    # --- multilingual permutations of raw CasiMedicos data [both _ordered.jsonl and _relations.jsonl] into splits/ ---

    def _generate_multi(self):
        """Generates all bilingual and multilingual .jsonl combinations."""
        bilingual_combos = list(combinations(self.all_langs, 2))
        multilingual_combo = tuple(self.all_langs)
        all_combos = bilingual_combos + [multilingual_combo]

        for split in self.splits:
            split_dir = self.splits_dir / split
            if not split_dir.exists():
                continue

            for suffix in ["_ordered.jsonl", "_relations.jsonl"]:
                for combo in all_combos:
                    
                    # ** naming convention: train_es-fr.jsonl, dev_es-it.jsonl, test_all.jsonl, etc. **
                    if len(combo) == len(self.all_langs):
                        combo_name = "all"
                    else:
                        combo_name = "-".join(combo)

                    combo_file_path = split_dir / f"{split}_{combo_name}{suffix}"

                    combined_lines = []
                    for lang in combo:
                        source_file = split_dir / f"{split}_{lang}{suffix}"
                        if source_file.exists():
                            with open(source_file, "r", encoding="utf-8") as f:
                                combined_lines.extend(f.readlines())

                    if combined_lines:
                        # ** reproduction seed for shuffling train data **
                        if split == "train":
                            random.seed(42)
                            random.shuffle(combined_lines)
                        
                        with open(combo_file_path, "w", encoding="utf-8") as f:
                            f.writelines(combined_lines)
                        logging.info(f"\t>>> Generated {combo_file_path.name}")
    
    # --- formatting ---

    def _normalize_ordered(self, line: str, lang: str, file_name: str, line_num: int) -> str:
        """Enforce flat schema: {"id": "...", "text": [...], "labels": [...]}"""
        line = line.strip()
        if not line:
            return ""

        try:
            record = json.loads(line)

            if not isinstance(record, dict) or not record:
                logging.warning(f"\t> (!) [Line {line_num} | {file_name}] Empty record")
                return ""

            # ** correct format **
            if set(record.keys()) == {"id", "text", "labels"}:
                case_id = str(record["id"])

                if not case_id.endswith(f"_{lang}"):
                    case_id = f"{case_id}_{lang}"

                text = record["text"]
                labels = record["labels"]

            # ** to-be-fixed: nested {"id_lang": {...}} **
            elif len(record) == 1:
                case_id, inner = next(iter(record.items()))

                if not isinstance(inner, dict) or "text" not in inner or "labels" not in inner:
                    return ""

                case_id = str(case_id)
                if not case_id.endswith(f"_{lang}"):
                    case_id = f"{case_id}_{lang}"

                text = inner["text"]
                labels = inner["labels"]

            else:
                logging.warning(f"\t> (!) [Line {line_num} | {file_name}] Unrecognized format")
                return ""

            """ debugging
            if not isinstance(text, list) or not isinstance(labels, list):
                logging.warning(f"\t> (!) [Line {line_num} | {file_name}] text/labels not lists")
                return ""

            if len(text) != len(labels):
                logging.warning(f"\t> (!) [Line {line_num} | {file_name}] text/labels length mismatch")
                return ""
            """
            return json.dumps({
                "id": case_id,
                "text": text,
                "labels": labels
            }, ensure_ascii=False) + "\n"

        except json.JSONDecodeError:
            logging.warning(f"\t> (!) [Line {line_num} | {file_name}] JSON decode error")
            return ""
    
    def _normalize_relations(self, line: str, lang: str, file_name: str, line_num: int) -> str:
        """format > "N_M_it": [["head", "tail", "Support"], ...]}"""
        line = line.strip()
        if not line:
            return ""
            
        try:
            record = json.loads(line)
            
            if isinstance(record, list):
                if not record: return ""
                record = record
                
            if not isinstance(record, dict) or not record:
                return ""
                
            normalized_record = {}
            for case_id, rels in record.items():
                case_id_str = str(case_id)
                if not case_id_str.endswith(f"_{lang}"):
                    case_id_str = f"{case_id_str}_{lang}"
                normalized_record[case_id_str] = rels
                
            return json.dumps(normalized_record, ensure_ascii=False) + "\n"
            
        except json.JSONDecodeError:
            return ""