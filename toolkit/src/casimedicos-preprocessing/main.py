# main.py
# ------------------------------------------------------------------------------------------------------
# casimedicos relation alignment and multilingual split generation (pre-processing for GRACE IBERLEF26)
# ------------------------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# apr-2026

import argparse
import logging
from pathlib import Path

try:
    from .relations import RelationAligner
    from .splits import SplitGenerator
    from .config import *
except ImportError:
    from relations import RelationAligner
    from splits import SplitGenerator
    from config import *

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Dataset preprocessing for CasiMedicos-Arg")
    parser.add_argument("--align", action="store_true", help="Run multi-lingual relation alignment")
    parser.add_argument("--split", action="store_true", help="Run multi-lingual split generation")
    parser.add_argument("--dataset-dir", type=Path, default=BASE_DATA_DIR, help="Processed CasiMedicos dataset directory")
    parser.add_argument("--raw-dir", type=Path, default=None, help="Raw CasiMedicos input directory")
    parser.add_argument("--relations-dir", type=Path, default=None, help="Relations directory")
    parser.add_argument("--splits-dir", type=Path, default=None, help="Output split directory")
    parser.add_argument("--source-lang", default=SOURCE_LANG, help="Source language for relation alignment")
    parser.add_argument("--target-langs", nargs="+", default=TARGET_LANGS, help="Target languages for relation alignment")
    parser.add_argument("--manual-fixes", type=Path, default=None, help="Manual relation fixes JSON")
    args = parser.parse_args()

    raw_dir = args.raw_dir or args.dataset_dir / "raw"
    relations_dir = args.relations_dir or raw_dir / "relations"
    splits_dir = args.splits_dir or args.dataset_dir / "splits"
    manual_fixes = args.manual_fixes or relations_dir / "fix_relations.json"
    source_lang = args.source_lang
    target_langs = args.target_langs
    split_sources = {
        "train": relations_dir / source_lang / "train_relations.jsonl",
        "validation": relations_dir / source_lang / "validation_relations.jsonl",
        "test": relations_dir / source_lang / "test_relations.jsonl"
    }

    if args.align:
        logging.info(f"[{source_lang}] Relation alignment for target languages: {', '.join(target_langs)}")
        aligner = RelationAligner(source_lang=source_lang, raw_dir=raw_dir, relations_dir=relations_dir)
        
        for lang in target_langs:
            for split, relations_path in split_sources.items():
                if not relations_path.exists():
                    logging.warning(f"\t(!) > Missing {relations_path}... >>> SKIPPED")
                    continue
                    
                aligned_data = aligner.align_split(lang, split, relations_path, manual_fixes)
                out_path = relations_dir / OUTPUT_JSONL.format(lang=lang, jsonl_split=split)
                aligner.save(aligned_data, out_path)

    if args.split:
        all_langs = [source_lang] + target_langs
        logging.info(f"Multilingual split generation for languages: {', '.join(all_langs)}")
        
        generator = SplitGenerator(
            raw_dir=raw_dir,
            relations_dir=relations_dir,
            splits_dir=splits_dir,
            all_langs=all_langs
        )
        generator.generate_splits()

if __name__ == "__main__":
    main()
