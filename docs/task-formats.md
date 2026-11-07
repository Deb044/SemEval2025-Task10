# Optional task-format support

## Optional official-format data

The public workflows above require no Task 10 access. The importer below also
supports authorized data from the
[official task website](https://propaganda.math.unipd.it/semeval2025task10/) if it
becomes available; this is not a prerequisite for the public experiments.
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

For the final cross-task leakage check, edit `experiments/audit-example.json` and
run `python -m semeval audit --manifest experiments/audit-example.json --out artifacts/all-tasks-audit.json`.
Files within one partition may share articles across tasks. Files in different
partitions must not. Supply `--duplicate-groups data/duplicate-groups.json` for
known translations/near-duplicates, mapping article IDs to a shared group name.
These groups require verification; content hashing alone cannot discover them.

ST1 imports require the correct `--offset-end inclusive` or `exclusive` for your
release. A mismatched mention stops the import. Gold imports require all documented
gold columns. For unlabeled ST2, omit `--annotations` and add `--unlabeled`.
For unlabeled ST1, provide a TSV of ID/mention/start/end; for ST3, provide
ID/dominant narrative/dominant subnarrative. Add `--unlabeled` in both cases.

Create `data/taxonomy.json`: a dictionary keyed by **exact fine label**, each with
`parent` (exact broad label) and `definition` (annotation-guideline meaning).
Include global `Other` with parent `Other` and all narrative-specific Other labels.
This mapping must be checked against your authorized release. Parents are verified
against the coarse inventory rather than guessed by splitting fine-label strings.
The taxonomy is never derived from held-out annotations.

If the release includes coarse/fine class inventories, build the mapping with:

```powershell
python -m semeval taxonomy --coarse data/scorer/subtask2_narratives.txt --fine data/scorer/subtask2_subnarratives.txt --definitions data/definitions.json --out data/taxonomy.json
```

`definitions.json` maps every fine label to a verified paraphrase of its guideline
definition. Omitting `--definitions` produces an explicitly marked name-only
mapping, useful for setup but distinct from the planned name-plus-definition
experiment. The command stops if a parent cannot be established from the full
coarse inventory; it never invents a relationship.

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
python -m semeval explain --data data/st3/test.jsonl --taxonomy data/taxonomy.json --method generate --gguf path/to/model.gguf --out artifacts/generated.jsonl
```

`--selection lead` runs the evidence ablation. `--predicted-labels
artifacts/st2-predictions.jsonl`  runs the end-to-end setting; these predictions must
cover the ST3 article IDs. Outputs resume only when their input signature matches.
An invented English example has passed the local generation path; broader
multilingual quality and real-data performance remain unverified. Generated
claims still require source-grounding review; length checks do not prove accuracy.

For ST3 evaluation, `--bert-model` and `--bert-layers` are explicit, mandatory
together. Match all settings to the organizer scorer before comparing official
scores. Without them, evaluation reports length/fallback statistics only.

