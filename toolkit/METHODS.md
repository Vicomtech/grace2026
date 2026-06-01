# Method Notes

This toolkit supports the data preparation and submission assembly workflow used for GRACE.

## Data Splitting

GRACE-format JSON arrays were split into deterministic train/evaluation subsets with `src/data-splitter/data_splitter.py`. The script optionally shuffles records with a fixed seed and writes one JSON file per requested split.

## Source Unification

Additional Spanish argumentation examples were obtained from CASIMEDICOS (`HiTZ/casimedicos-arg`) and converted into the GRACE schema with `src/source-unifier/source_unifier.py`.

The conversion reconstructs raw text from CASIMEDICOS tokenized sentences, maps BIO tags to GRACE entities, remaps answer-option claims to their choice IDs when possible, and links relations by matching relation endpoint text to extracted entities. CASIMEDICOS rationale text is kept by default to preserve relations that point to post-answer explanations.

## Task-Specific Ensembling

Model predictions were ensembled independently for the three subtasks:

- S1 sentence relevance: majority or weighted voting over sentence labels.
- S2 entity extraction: exact voting, best-model selection, or token-IoU span clustering.
- S3 relation extraction: exact voting, relation clustering, or entity-aligned voting over aggregated entity spans.

Task weights were generated from official scorer outputs with `task_weight_generator.py`, using one weight file per subtask.

## Submission Assembly

Final runs were assembled with `run_assembler.py`. The assembled file takes sentence relevance from the chosen S1 output, entities from the chosen S2 output, and relations from the chosen S3 output. Relations are validated and remapped so their arguments refer to the final S2 entity IDs.
