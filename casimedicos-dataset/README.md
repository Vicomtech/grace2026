# Processed CasiMedicos-Arg Dataset

This folder contains the CasiMedicos-Arg data as used in our GRACE2026
experiments. It is not a new dataset; it is a processed version of the public
[HiTZ/casimedicos-arg](https://huggingface.co/datasets/HiTZ/casimedicos-arg)
release prepared for GRACE-style argument mining experiments.

## Source

CasiMedicos-Arg is a multilingual medical question-answering dataset with
argument components and support/attack relations for English, Spanish, French,
and Italian clinical cases. The Hugging Face release is distributed under
CC-BY-4.0. Please cite the original dataset paper when using these files:

[CasiMedicos-Arg: A Medical Question Answering Dataset Annotated with
Explanatory Argumentative Structures](https://aclanthology.org/2024.emnlp-main.1026/)
(Sviridova et al., EMNLP 2024).

## Processing

Starting from the Hugging Face dataset, we prepared the files in this folder by:

- normalizing the upstream tokenized `text` and BIO `labels` records into JSONL;
- keeping the original train/test splits and mapping upstream `validation` to
  local `dev`;
- aligning English support/attack relation files to Spanish, French, and Italian
  records through sentence-level cross-language matching;
- applying manually checked relation fixes from `raw/relations/fix_relations.json`;
- preserving skipped alignment candidates and processing reports under `raw/`;
- adding language suffixes to record IDs where needed;
- generating monolingual, bilingual, and all-language split files under
  `splits/{train,dev,test}`.

The preprocessing code is available in
[`../toolkit/src/casimedicos-preprocessing`](../toolkit/src/casimedicos-preprocessing).

The resulting files are used by the GRACE toolkit source unifier to create
GRACE-style JSON records with `origin: "CASIMEDICOS"`. The final unified files
used in our experiments are not committed here because the official GRACE
competition data has not yet been released by the organizers.

## Layout

- `raw/`: normalized source files, aligned relation files, manual fixes, skipped
  relation candidates, and processing reports.
- `splits/`: ready-to-use JSONL files for each split. Each split contains
  `*_ordered.jsonl` files with token/BIO annotations and `*_relations.jsonl`
  files with support/attack relations.
- `splits/*/*_all_*.jsonl`: multilingual combinations across all available
  languages.

## Citation

If you use this dataset please cite:

```bibtex
@inproceedings{sviridova-etal-2024-casimedicos,
    title = "{C}asi{M}edicos-Arg: A Medical Question Answering Dataset Annotated with Explanatory Argumentative Structures",
    author = "Sviridova, Ekaterina and
      Yeginbergen, Anar and
      Estarrona, Ainara and
      Cabrio, Elena and
      Villata, Serena and
      Agerri, Rodrigo",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    year = "2024",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.1026/",
    doi = "10.18653/v1/2024.emnlp-main.1026",
    pages = "18463--18475"
}
```
