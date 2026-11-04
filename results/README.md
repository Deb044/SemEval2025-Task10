# Experiment evidence

The complete typed catalog is [experiments.json](experiments.json), also displayed
in the website’s Experiments tab. It separates scores, format statistics, model-based
grounding diagnostics and timing units. Expand a result for seeds, language scores,
class-error diagnostics, source commits and input hashes.

## Current classification studies

All values below are native fixed-inventory macro F1. GNN entries average three
seeds; lexical and transformer-only comparisons use seed 42. The core architecture
is the cross-lingual transformer plus hierarchical label GCN, regardless of which
method has the highest score.

| Study | Model / control | Macro F1 | Seed SD |
|---|---|---:|---:|
| CARDS · extended hierarchy study | MiniLM + hierarchical GCN | 0.4682 | 0.0132 |
| CARDS · extended hierarchy study | Character TF-IDF + logistic regression | 0.4330 | — |
| CARDS · extended hierarchy study | LSA + logistic regression | 0.5510 | — |
| CARDS · extended hierarchy study | MiniLM + hierarchical GCN · random-labels | 0.4431 | 0.0198 |
| CARDS · extended hierarchy study | MiniLM + hierarchical GCN · self-loops | 0.4582 | 0.0211 |
| CARDS · extended hierarchy study | MiniLM + hierarchical GCN · shuffled-edges | 0.4313 | 0.0232 |
| CARDS · extended hierarchy study | MiniLM + logistic regression | 0.5274 | — |
| ProBel · extended bilingual study | MiniLM + hierarchical GCN | 0.0862 | 0.0092 |
| ProBel · extended bilingual study | XLM-R + hierarchical GCN | 0.0983 | 0.0013 |
| ProBel · extended bilingual study | Character TF-IDF + logistic regression | 0.0558 | — |
| ProBel · extended bilingual study | LSA + logistic regression | 0.0911 | — |
| ProBel · extended bilingual study | MiniLM + hierarchical GCN · random-labels | 0.0892 | 0.0103 |
| ProBel · extended bilingual study | MiniLM + hierarchical GCN · self-loops | 0.0995 | 0.0201 |
| ProBel · extended bilingual study | MiniLM + hierarchical GCN · shuffled-edges | 0.0817 | 0.0029 |
| ProBel · extended bilingual study | MiniLM + logistic regression | 0.0876 | — |
| ProBel · extended bilingual study | XLM-R + logistic regression | 0.0422 | — |
| CARDS → EMM source transfer | MiniLM + hierarchical GCN | 0.2701 | 0.0135 |
| CARDS → EMM source transfer | Character TF-IDF + logistic regression | 0.2421 | — |
| CARDS → EMM source transfer | LSA + logistic regression | 0.3593 | — |
| CARDS → EMM source transfer | MiniLM + hierarchical GCN · random-labels | 0.2817 | 0.0257 |
| CARDS → EMM source transfer | MiniLM + hierarchical GCN · self-loops | 0.2614 | 0.0062 |
| CARDS → EMM source transfer | MiniLM + hierarchical GCN · shuffled-edges | 0.2697 | 0.0418 |
| CARDS → EMM source transfer | MiniLM + logistic regression | 0.3451 | — |

CARDS tests 2,897 paragraphs across 18 labels. ProBel uses the fixed researcher
article-grouped news pilot (1,600 train / 400 development / 400 test; English and
Arabic; 23 techniques). EMM evaluates the fixed CARDS models on 1,118 held-out
climate snippets without training or tuning on EMM. Full graph, self-loop,
degree-preserving shuffled edges and random label vectors are compared. Paired
intervals identify their metric explicitly; they are not macro-F1 seed intervals.

## Longer-text explanations and grounding

The fixed 24-passage CARDS sample has at least eight sentences per passage. Semantic
and leading selection choose different evidence in 20 of 24 cases. The seed-42 GNN
predicts the supplied native claim label correctly in 9 of 24 cases. Compare extraction,
supplied-label generation, leading-evidence generation and predicted-label generation.
There are no human reference explanations for this sample.

Grounding assessment uses 30 blinded outputs from six source passages. A separate
Codex LLM judge applied the published 0/1/2 rubric without method identities or
classification scores. The exact runtime model identifier was not exposed. Its
frozen aggregate audit is ancillary evidence, not bitwise reproducible human gold.
A separately pinned DeBERTa NLI diagnostic checks source-window entailment.

The evaluators disagree on some outputs, including copied/extracted text. NLI can
miss discourse and attribution; the LLM can misjudge grounding or favor fluency.
Neither certifies factual truth or establishes human-validated explanation quality.
See [the judge protocol](../docs/llm-review-prompt.md) and [frozen audit](llm-audit.json).
Raw source texts, individual explanations and ratings remain local.

## Processing and inference measurements

The scaling study uses 40,749 unique public news records, with workloads of 1,000,
5,000, 10,000 and 25,000. JSON/Parquet conversion and queries run in isolated
subprocesses. First scans may include engine initialization and OS-cache effects;
Parquet is not consistently faster at these sizes. The 256 MB DuckDB setting is
an engine memory limit, not a total-process RAM bound.

MiniLM encoding is measured at 32 and 128 texts with model loading separated.
Cached vectors are checked for exact equality. Core MiniLM–GCN and XLM-R–GCN
inference is measured at 32, 128 and 400 held-out records with a verified complete
embedding cache. This includes cached-feature reading, scoring and decoding; it
is not raw-text transformer throughput. No distributed or larger-than-RAM result
is claimed.

## Reproduction and scope

Follow the [root setup and reproduction commands](../README.md). Configurations
and source/data hashes are retained; compatible checkpoints are validated before
reuse. [Error diagnostics](error-analysis.json) use fixed seed 42, with rare labels
defined as at most ten training positives. They are descriptive, not evidence of
a model improvement selected after test inspection.

Initial CARDS, ProBel, YASO, e-SNLI and transfer results remain in [history.json](history.json)
with original source provenance. A fresh reproduction reruns the current studies;
archived LLM assessments are visibly frozen, not silently treated as new judgments.
The official Task 10 corpus is unavailable. Public native claims and propaganda
techniques are not its narrative or entity-role annotations. Long-text grounding
is English-only; the shorter ProBel reference comparison covers English/Arabic
using publisher model-authored references.

## Apache Spark SQL input gate

The standard preparation/classification workflow now runs Spark SQL in `local[2]`
mode before model training. The measured audit covered **12,297 prepared records**
across all six CARDS/ProBel splits and found zero invalid records, within-split
duplicate groups or cross-split article-ID/normalized-text overlap groups.
Language and label counts, separate JVM startup/SQL timings, source hashes and
input checksums are in [spark-audit.json](spark-audit.json). This is a local
data-quality stage; no cluster speedup is claimed. Earlier model scores remain unchanged.

## Three generators and named LLM review

The website features five detailed studies; other results remain in its appendix.
The common 24-passage generator pilot uses pinned Q4_K_M models, identical
prompts/evidence, supplied labels and no repair retry. Qwen2.5-3B-Instruct reviews
the raw outputs on the same six passages per generator, blind to model identity.

| Generator | Valid raw outputs | Mean LLM rating /2 (n=6) | Median request seconds |
|---|---:|---:|---:|
| Qwen3-1.7B | 17/24 | 1.17 | 7.49 |
| SmolLM2-1.7B-Instruct | 18/24 | 1.08 | 8.33 |
| TinyLlama-1.1B-Chat-v1.0 | 0/24 | 1.29 | 9.72 |

TinyLlama received the highest mean rubric rating but exceeded the word limit
in every case. Qwen3 had the higher judged mean among models producing valid
outputs. These small English-pilot results require separate interpretation of
format, source faithfulness and latency. The judge is a single quantized model
sharing the Qwen family with one generator.

See [generator aggregates](generator-comparison.json), [named review](named-review.json),
[featured studies](featured-studies.json), and the [protocol](../docs/generator-comparison.md).
