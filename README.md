# Multilingual Narrative Understanding

CPU-first implementation of **SemEval-2025 Task 10**: entity framing, narrative
classification, and evidence-grounded explanations. Focus: subtasks 2 and 3.

**Status:** implementation in progress. No benchmark scores have been measured.
Restricted news data is not included. Synthetic checks are not research results.

## Setup

Use Python 3.11–3.13 in a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Transformer/GNN/BERTScore experiments additionally require
`python -m pip install -r requirements-models.txt`. The generator uses a local
GGUF model and `llama-server`, installed separately; no paid API is required.

## Data

Request authorized data and scorer access from the
[official task website](https://propaganda.math.unipd.it/semeval2025task10/).
Keep downloads under ignored `data/`. Do not publish the corpus, model APIs,
or private article excerpts. Free cloud use depends on the dataset conditions.

Gold labels are TSV: entity mention rows for ST1, article/narratives/subnarratives
for ST2, article/dominant narrative/dominant subnarrative/explanation for ST3.
Official predictions have 6+ columns, 3 columns, and 2 columns respectively.
Follow the exact label spellings in your corpus release. The taxonomy PDF and
website examples contain naming differences: never silently rename gold labels.

The core design compares lexical TF-IDF, latent semantic analysis, and frozen
multilingual transformer representations. A small label GCN tests hierarchy;
explanations compare TextRank, semantic extraction, and evidence-guided generation.

## Run

Import each task and split separately (example for narrative classification):

```powershell
python -m semeval import --task 2 --articles data/raw/train-articles-subtask-2 --annotations data/raw/train-labels.tsv --out data/st2/train.jsonl
python -m semeval audit data/st2/train.jsonl data/st2/dev.jsonl data/st2/test.jsonl --out artifacts/audit.json
```

ST1 imports require the correct `--offset-end inclusive` or `exclusive` for your
release. A mismatched mention stops the import. Gold imports require all documented
gold columns; unlabeled inference records use the same JSON fields without targets.

Create `data/taxonomy.json`: a dictionary keyed by **exact fine label**, each with
`parent` (exact broad label) and `definition` (annotation-guideline meaning).
Include global `Other` with parent `Other` and all narrative-specific Other labels.
This mapping must be checked against your authorized release. We do not infer
parents from colon-delimited strings or derive a taxonomy from held-out labels.

```powershell
python -m semeval train --train data/st2/train.jsonl --dev data/st2/dev.jsonl --method tfidf --taxonomy data/taxonomy.json --out artifacts/tfidf.joblib
python -m semeval predict --model artifacts/tfidf.joblib --data data/st2/test.jsonl --out artifacts/tfidf.jsonl
python -m semeval evaluate --gold data/st2/test.jsonl --predictions artifacts/tfidf.jsonl --out artifacts/tfidf.metrics.json
python -m semeval export --predictions artifacts/tfidf.jsonl --out artifacts/tfidf.tsv
```

Use `--method lsa`, `transformer`, or `graph` for the remaining methods. Graph
ablations: `--ablation self-loops` and `--ablation random-labels`. ST1 uses the same
commands without `--taxonomy` and supports the three non-graph representations.
Only load your own `.joblib` files; they use Python pickle serialization.

```powershell
python -m semeval explain --data data/st3/test.jsonl --taxonomy data/taxonomy.json --method textrank --out artifacts/textrank.jsonl
python -m semeval explain --data data/st3/test.jsonl --taxonomy data/taxonomy.json --method semantic --out artifacts/semantic.jsonl
python -m experiments.run experiments/example.json
```

See [the experiment protocol](experiments/README.md) for controlled comparisons,
three-seed GCN runs, data leakage checks, and manual review. All raw predictions,
checkpoints, and private data remain ignored. Exact-match and document-F1 local
metrics implement published definitions; parity with organizer scripts is pending.

## Local explanations

Install [llama.cpp](https://github.com/ggml-org/llama.cpp), obtain an authorized
Q4_K_M GGUF conversion of `Qwen/Qwen3-1.7B`, and record its source revision and
SHA256. Run `llama-server -m path/to/model.gguf --host 127.0.0.1 --port 8080 -c 4096`.
Keep the server local; the corpus agreement restricts public model APIs.

```powershell
python -m semeval explain --data data/st3/test.jsonl --taxonomy data/taxonomy.json --method generate --out artifacts/generated.jsonl
```

`--selection lead` runs the evidence ablation. `--predicted-labels
artifacts/st2-predictions.jsonl` runs the end-to-end setting; these predictions must
cover the ST3 article IDs. Outputs resume only when their input signature matches.
Quantized generation has not yet been validated with a downloaded model.

For ST3 evaluation, `--bert-model` and `--bert-layers` are explicit, mandatory
together. Match all settings to the organizer scorer before comparing official
scores. Without them, evaluation reports length/fallback statistics only.

## Checks and website

```powershell
python -m unittest discover -s tests -v
python -m http.server 8000 --directory website/dist
```

The checks use invented text, not SemEval data. The buildless website contains no
model endpoint and no private corpus content. Its aggregate results file starts
empty; measured findings must be reviewed before publication.

## References

- [Task definition, metrics, data conditions](https://propaganda.math.unipd.it/semeval2025task10/)
- [Organizer overview paper](https://aclanthology.org/2025.semeval-1.331/)
- [Narrative taxonomy](https://propaganda.math.unipd.it/semeval2025task10/NARRATIVE-TAXONOMIES.pdf)
- [Entity roles](https://propaganda.math.unipd.it/semeval2025task10/ENTITY-ROLE-TAXONOMY.pdf)

