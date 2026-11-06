# Native multilingual extension

This extension adds native Bulgarian, Russian, Hindi and Portugal-publisher
Portuguese evaluations. It tests multilingual components using original auxiliary
labels; it does not supply new official SemEval narrative annotations.

## Sources and tasks

- Slavic Persuasion Techniques Corpus v1.0: Bulgarian parliamentary debates and
  Russian social media, with 25 fine techniques in six parent groups. Sources:
  https://github.com/SlavicNLP/SlavicPersuasionTechniques and
  https://aclanthology.org/2026.lrec-1.289/ (Figure 1 for the hierarchy).
  Use the actual ZIP in `LREC-2026-paper-corpus/`, not the HF documentation-only
  viewer. The archive includes publisher-normalized sentences and document/offset
  keys. We join those supplied sentences to labels by exact key; we do not repair
  inconsistent raw-document offsets or reuse them as verified evidence spans.
- H-Prop-News v1.0: 5,500 native Hindi news articles with human binary propaganda
  labels (`-1`, `1`), from https://zenodo.org/records/5828240. The translated
  H-Prop corpus is not used. The release misspells the URL column `rrticle_url`;
  the preparer explicitly handles that released schema. No explicit permissive
  dataset license was established from its README; raw data are kept local and
  not redistributed.
- PartisanLens: hyperpartisanship labels on immigration headlines, from
  https://github.com/MichJoM/PartisanLens. The actual Portuguese data include
  Brazilian and other-country publishers. We retain a fixed allowlist of 18
  Portugal publisher domains and exclude all others. This is a publisher-location
  proxy for European Portuguese, not independent linguistic validation of every
  headline or syndicated text. The repository carries Apache-2.0 licensing;
  source texts retain their underlying rights. Rationales are not used.

Immutable repository commits, URLs and SHA256 checksums for every downloaded
file are in `experiments/configs/multilingual-sources.json`. All data, article
identifiers and trained weights remain in ignored local folders. Only aggregate
scores, source metadata, protocols and plots are exported.

## Split and compute protocol

Publisher partitions are pooled and replaced with researcher-defined 70/15/15
source-group splits, seed 42. Source groups connected by whitespace-normalized
exact duplicate text stay together. Conflicting duplicate labels are excluded;
consistent duplicates are collapsed after connecting their source groups.
Examples must contain at least five whitespace-separated words, at most 6,000
characters and no Unicode replacement character. Complete examples are retained;
text is not silently truncated. These filters change the target population.

The final compute-limited pilots are:

| Task | Train | Development | Test |
| --- | --- | --- | --- |
| Bulgarian persuasion | 500 | 150 | 150 |
| Russian persuasion | 500 | 150 | 150 |
| Hindi propaganda | 400 | 100 | 150 |
| Portugal-publisher hyperpartisanship | 151 | 32 | 33 |

Sampling occurs inside each language/partition before training. It preserves
binary-label proportions, or labelled/unlabelled proportions for the multi-label
Slavic task, when capping is necessary. Source-group and exact-text disjointness
are asserted again after capping. Translations and semantic near duplicates are
not established. Test groups are few, especially Bulgarian and Portuguese.

The full native inventory is fixed before training. Slavic macro F1 includes all
25 labels even when a label is absent in a pilot test set. Empty technique sets
remain valid native negatives. Parent descriptions use publisher label names.
Hindi and Portuguese are independent binary tasks; identity parents are only
schema bookkeeping. No graph is trained on either binary taxonomy.

## Experiments and evaluation

All three datasets compare TF-IDF + logistic regression, LSA + logistic regression
and frozen multilingual MiniLM + logistic regression. The bilingual Slavic matrix
also runs full, self-loop-only, shuffled-edge and random-label-vector GNNs at seeds
13/42/87. Each model receives identical splits. Graph bars show sample standard
deviation across seeds, not document-level confidence intervals.

Slavic transfer models train on 500 source-language sentences and select settings
on only that language's development set. The same fitted model predicts both
language test sets. Both directions compare MiniLM linear and full graph heads,
seed 42. Target development labels are never used. Language is confounded with
parliament/social-media domain. Bilingual models receive twice as many training
examples as the source-only models; the comparison does not isolate the effect
of multilingual training alone.

Hindi and Portuguese additionally report a training-majority-class baseline.
Macro F1, sample F1, micro F1, exact match and per-language/per-label measures
are recorded where applicable. Scores from different tasks are not averaged into
a single multilingual score. Small samples and class imbalance limit conclusions;
Portuguese test accuracy alone would be particularly misleading.

Spark SQL `local[2]` validates all nine prepared files before the standard training
stage: schema, valid languages, nonempty inputs, and cross-split article ID, text
hash and connected source-group overlap. It is a local audit, not a cluster
scalability result. Its aggregate audit contains no source text or IDs.

## Reproduce

After the root README dependency/model setup, run from the repository root:

```powershell
python -m experiments.multilingual --stage all
python -m experiments.catalog
python -m experiments.website_studies
```

Stages can also be run separately: `prepare`, `audit`, `train`, `transfer`,
`export`. The `train` stage always reruns the Spark gate. Classifier matrices
resume only matching implementation/configuration/input signatures. Completed
transfer outputs are reused only after source and input hash verification.
For changed experiments, preserve previous outputs and select fresh output
locations in the relevant configuration/code.

Prepared data and protocol: `data/public/multilingual/`. Matrix checkpoints:
`artifacts/native-multilingual-{slavic,hindi,portuguese}/`. Transfer:
`artifacts/multilingual-transfer/`. Spark audit: `artifacts/multilingual-audit/`.
Public aggregates: `results/multilingual.json`. Study 02 on the website includes
three additional plots and tables without increasing the five featured studies.

These resources strengthen native-language evaluation of the architecture.
They do not turn persuasion, propaganda or hyperpartisanship annotations into
official narrative labels, and they do not establish multilingual explanation
quality. The official-taxonomy article lab remains a separate zero-shot mode.
