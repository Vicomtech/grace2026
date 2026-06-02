# relations.py
# ----------------------------------------------------------------------------------------
# relation mapping and alignment for https://huggingface.co/datasets/HiTZ/casimedicos-arg
# ----------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# mar-2026

import re
import json
import logging
from typing import Dict, List, Any, Tuple
import pandas as pd
from pathlib import Path

class RelationAligner:
    """Cross-lingual relation aligner for the CasiMedicos dataset."""
    
    def __init__(self, source_lang: str = "en", raw_dir=None, relations_dir=None):
        self.source_lang = source_lang
        repo_root = Path(__file__).resolve().parents[3]
        raw_dir = raw_dir or repo_root / "casimedicos-dataset" / "raw"
        relations_dir = relations_dir or raw_dir / "relations"
        self.raw_dir = Path(raw_dir)
        self.relations_dir = Path(relations_dir)

    def align_split(self, target_lang: str, split: str, source_jsonl_path: Path, manual_fixes_path: Path = None) -> List[Dict[str, Any]]:
        logging.info(f"> Loading local JSON for {split} split ({self.source_lang} -> {target_lang})...")
        
        try:
            df_src = self._load_df(self.source_lang, split)
            df_tgt = self._load_df(target_lang, split)
        except Exception as e:
            logging.error(f"\t (!) > Failed to load JSON files: {e}")
            return []
            
        translation_map = self._build_translation_map(df_src, df_tgt)
        aligned_relations, skipped_relations, base_stats = self._map_relations(translation_map, source_jsonl_path, target_lang, split)
        
        # manually-validated skipped relations
        if manual_fixes_path and manual_fixes_path.exists():
            aligned_relations, skipped_relations = self._apply_manual_fixes(
                aligned_relations, skipped_relations, manual_fixes_path, target_lang, split
            )
        
        # stats after manual injections
        final_matched = sum(len(list(record.values())) for record in aligned_relations)
        final_skipped = sum(len(list(record.values())) for record in skipped_relations)
        final_total = final_matched + final_skipped
        final_cov = (final_matched / final_total) * 100 if final_total > 0 else 0.0

        logging.info(
            f"\n\t --- CasiMedicos relations corpus: [{self.source_lang}] -> [{target_lang}] ({split}) ---\n"
            f"\t  Total cases:         {base_stats['total_cases']}\n"
            f"\t  Related cases:       {base_stats['cases_with_relations']}\n"
            f"\t  Total relations:     {final_total}\n"
            f"\t  Matched relations:   {final_matched}\n"
            f"\t  Skipped relations:   {final_skipped}\n"
            f"\t  Final coverage:      {final_cov:.2f}%\n"
            f"\t -------------------------------------------------------"
        )
        
        if skipped_relations: 
            skipped_path = self.relations_dir / target_lang.lower() / "skipped" / f"skipped_{split}.jsonl"
            self.save(skipped_relations, skipped_path)
            self._print_skipped_relations(skipped_relations, target_lang, split)        
            
        return aligned_relations
    
    # --- core logic -------------------------------------------------------------------------

    def _apply_manual_fixes(self, aligned_relations: List[Dict[str, Any]], skipped_relations: List[Dict[str, Any]], manual_fixes_path: Path, target_lang: str, split: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """loads the manual fixes JSON and injects them, printing before and after states."""
        try:
            with open(manual_fixes_path, 'r', encoding='utf-8') as f:
                manual_fixes = json.load(f)
        except Exception as e:
            logging.error(f"\t (!) > Failed to load manual fixes JSON: {e}")
            return aligned_relations, skipped_relations

        lang_key = target_lang.lower()
        fixes_for_split = manual_fixes.get(lang_key, {}).get(split, {})
        
        if not fixes_for_split:
            return aligned_relations, skipped_relations

        logging.info(f"\n\t >>> Manual relation injection: [{lang_key}] ({split}) ---")
        
        aligned_dict = {list(item.keys())[0]: list(item.values())[0] for item in aligned_relations}
        skipped_dict = {list(item.keys())[0]: list(item.values())[0] for item in skipped_relations}

        for case_id, fixes in fixes_for_split.items():
            case_id_str = str(case_id)
            
            
            if case_id_str in skipped_dict and skipped_dict[case_id_str]:
                logging.info(f"\t  - Case {case_id_str}:")
                for rel in skipped_dict[case_id_str]:
                    logging.info(f"\t    - Before (skipped): {rel[0][:50]}... /// {rel[1][:50]}...")

            if case_id_str not in aligned_dict:
                aligned_dict[case_id_str] = []

            for h, t, label in fixes:
                logging.info(f"\t    + After (injected):    {h[:50]}... -> {t[:50]}... [{label}]")
                if [h, t, label] not in aligned_dict[case_id_str]:
                    aligned_dict[case_id_str].append([h, t, label])
            
            if case_id_str in skipped_dict:
                del skipped_dict[case_id_str]

        updated_aligned = [{k: v} for k, v in aligned_dict.items()]
        updated_skipped = [{k: v} for k, v in skipped_dict.items() if v]

        return updated_aligned, updated_skipped

    def _build_translation_map(self, df_src: pd.DataFrame, df_tgt: pd.DataFrame) -> Dict[str, List[Tuple[str, str]]]:
        """creates a lookup map between normalized source sentences and target sentences."""
        src_cases = df_src.set_index('id')['text'].to_dict()
        tgt_cases = df_tgt.set_index('id')['text'].to_dict()
        translation_map = {}
        
        for case_id, src_text_list in src_cases.items():
            if case_id not in tgt_cases:
                continue
                
            translation_map[case_id] = []
            tgt_text_list = tgt_cases[case_id]
            
            for sent_src_tokens, sent_tgt_tokens in zip(src_text_list, tgt_text_list):
                tgt_str = re.sub(r'\s+([.,;:?!)])', r'\1', " ".join(sent_tgt_tokens))
                src_key = self._clean("".join(sent_src_tokens))
                translation_map[case_id].append((src_key, tgt_str))
                
        return translation_map

    def _map_relations(self, translation_map: Dict[str, List[Tuple[str, str]]], source_jsonl_path: Path, target_lang: str, split: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
        """translates relations using the map and returns them along with base case statistics."""
        aligned_relations = []
        skipped_relations = []
        
        base_stats = {
            "total_cases": 0, "cases_with_relations": 0
        }
        
        with open(source_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line.strip())
                if isinstance(record, list): 
                    record = record[0]

                for case_id, relations in record.items():
                    base_stats["total_cases"] += 1
                    case_id_str = str(case_id)
                    
                    if not relations:
                        aligned_relations.append({case_id_str: []})
                        continue

                    base_stats["cases_with_relations"] += 1
                    cmap = translation_map.get(case_id_str, [])
                    translated_relations = []
                    skipped_for_case = []

                    for src_rel_str, tgt_rel_str, label in relations:
                        norm_src = self._clean(src_rel_str)
                        norm_tgt = self._clean(tgt_rel_str)

                        tgt_match_1 = self._find_match(norm_src, cmap)
                        tgt_match_2 = self._find_match(norm_tgt, cmap)

                        if tgt_match_1 and tgt_match_2:
                            translated_relations.append([tgt_match_1, tgt_match_2, label])
                        else:
                            skipped_for_case.append([src_rel_str, tgt_rel_str, label])

                    if translated_relations or not skipped_for_case: 
                        # ensure the case exists in aligned_relations even if empty (but had relations originally)
                        aligned_relations.append({case_id_str: translated_relations})
                    if skipped_for_case:
                        skipped_relations.append({case_id_str: skipped_for_case})
                        
        return aligned_relations, skipped_relations, base_stats

    # --- heuristics & utilities -------------------------------------------------------------

    def _find_match(self, norm_rel: str, translations: List[Tuple[str, str]]) -> str:
        """finds the best target sentence match using exact, substring, or heuristic logic."""
        if not norm_rel: return None
        
        for src_key, tgt_str in translations:
            if norm_rel == src_key: return tgt_str
            
        for src_key, tgt_str in translations:
            if len(norm_rel) > 4 and norm_rel in src_key: return tgt_str
            if len(src_key) > 4 and src_key in norm_rel: return tgt_str
            
        if "answer" in norm_rel and "correct" in norm_rel:
            digits = [c for c in norm_rel if c.isdigit()]
            if digits:
                for src_key, tgt_str in translations:
                    if "correct" in src_key and "answer" in src_key and digits[0] in src_key:
                        return tgt_str
                        
        return None

    # --- io ---------------------------------------------------------------------------------

    def _print_skipped_relations(self, skipped_relations: List[Dict[str, Any]], target_lang: str, split: str) -> None:
        """prints all unrecovered relations for debugging."""
        logging.warning(f"\n\n\t [!] Printing all skipped relations for {target_lang} - {split}:")
        for record in skipped_relations:
            for case_id, relations in record.items():
                for rel in relations:
                    logging.warning(f"\t\t Case {case_id} | {rel[0].strip()} <-> {rel[1].strip()}")

    def _load_df(self, lang: str, split: str) -> pd.DataFrame:
        """loads the structured local .json/.jsonl files (deals with structural mismatches)."""
        lang_lower = lang.lower()
        ext = "json" if lang=="en" else "jsonl"
        file_name = f"{split}_{lang_lower}_ordered.{ext}"
        file_path = self.raw_dir / lang_lower / file_name
                    
        if not file_path.exists():
            raise FileNotFoundError(f"(!) > Missing file: {file_path}")

        records = []
        if ext == "jsonl":
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    
                    item = json.loads(line)
                    if "id" in item or "case_id" in item:
                        item_id = str(item.get("id", item.get("case_id", "")))
                        records.append({"id": item_id, "text": item.get("text", [])})
                    else:
                        for k, v in item.items():
                            if isinstance(v, dict):
                                records.append({"id": str(k), "text": v.get("text", [])})
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                
            if isinstance(raw_data, dict):
                for k, v in raw_data.items():
                    records.append({"id": str(k), "text": v.get("text", [])})
            elif isinstance(raw_data, list):
                for item in raw_data:
                    item_id = str(item.get("id", item.get("case_id", "")))
                    records.append({"id": item_id, "text": item.get("text", [])})
                
        df = pd.DataFrame(records)
                
        if df.empty or 'id' not in df.columns:
            raise KeyError(f"(!) > Failed to parse 'id' from {file_path}... >>> SKIPPING LANG SPLIT")
            
        return df
            
    def _clean(self, text: str) -> str:
        """normalizes text for robust string matching."""
        return re.sub(r'[^a-z0-9]', '', str(text).lower())
    
    def save(self, data: List[Dict[str, Any]], output_path: Path) -> None:
        """saves mapped relations to a jsonl file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        logging.info(f">>> Saved aligned relations to: {output_path}\n\n")
