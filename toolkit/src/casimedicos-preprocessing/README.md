# CasiMedicos Preprocessing

Toolkit-compatible preprocessing scripts for the processed CasiMedicos-Arg data
used in the GRACE2026 experiments.

This module preserves the original preprocessing implementation from
`icl/src/casimedicos` and adapts its paths to the published repository layout:

```text
grace2026/
  casimedicos-dataset/
    raw/
    splits/
  toolkit/
    src/casimedicos-preprocessing/
```

## Inputs

By default, the script reads from `../../../casimedicos-dataset/raw` relative to
this directory. Expected inputs are:

- `raw/{en,es,fr,it}/`: ordered token/BIO files from CasiMedicos-Arg;
- `raw/relations/en/`: English relation files used as the alignment source;
- `raw/relations/fix_relations.json`: manually checked relation fixes.

## Usage

From this directory:

```bash
python main.py --align --split
```

The default command:

- aligns English relations to Spanish, French, and Italian;
- writes aligned relations under `casimedicos-dataset/raw/relations/{lang}/`;
- writes skipped relation candidates under
  `casimedicos-dataset/raw/relations/{lang}/skipped/`;
- regenerates `casimedicos-dataset/splits/{train,dev,test}/`.

`--split` recreates the target `splits/` directory.

## Custom Paths

```bash
python main.py \
  --align \
  --split \
  --dataset-dir ../../../casimedicos-dataset
```

You can also override individual directories:

```bash
python main.py \
  --align \
  --raw-dir ../../../casimedicos-dataset/raw \
  --relations-dir ../../../casimedicos-dataset/raw/relations
```

## Options

- `--align`: run cross-lingual relation alignment.
- `--split`: generate monolingual, bilingual, and all-language split files.
- `--source-lang`: source language for relation alignment, default `en`.
- `--target-langs`: target languages, default `es fr it`.
- `--manual-fixes`: manual fixes JSON, default
  `raw/relations/fix_relations.json`.
