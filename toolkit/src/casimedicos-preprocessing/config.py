# toolkit/src/casimedicos-preprocessing/config.py
# ----------------------------------------------------------
# configurations and paths
# ----------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# apr-2026

from typing import List, Dict
from pathlib import Path

# --- general path directories and settings ---
REPO_ROOT: Path = Path(__file__).resolve().parents[3]
BASE_DATA_DIR: Path = REPO_ROOT / "casimedicos-dataset"
SPLITS_DATA_DIR: Path = BASE_DATA_DIR / "splits"
RAW_DATA_DIR: Path = BASE_DATA_DIR / "raw"
RELATIONS_DIR: Path = RAW_DATA_DIR / "relations"

# --- relation alignment settings ---
SOURCE_LANG: str = "en"
TARGET_LANGS: List[str] = ["es", "fr", "it"]

SPLITS: Dict[str, Path] = {
    "train": RELATIONS_DIR / SOURCE_LANG / "train_relations.jsonl",
    "validation": RELATIONS_DIR / SOURCE_LANG / "validation_relations.jsonl",
    "test": RELATIONS_DIR / SOURCE_LANG / "test_relations.jsonl"
}

OUTPUT_JSONL: str = "{lang}/{jsonl_split}_relations.jsonl"
MANUAL_FIX_JSON: Path = RELATIONS_DIR / "fix_relations.json"
