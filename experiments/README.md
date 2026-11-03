# Experiment protocol

For the current public-only news experiments, use [PUBLIC_ALTERNATIVES.md](PUBLIC_ALTERNATIVES.md).

1. Obtain authorized data and scoring scripts. Import each language/split, then
   concatenate languages within the same task/split. Never combine different
   official partitions merely because they have different task numbers.
2. Audit all task splits together for article leakage. Manually inspect known
   translated versions and near-duplicates; content hashes cannot find those.
3. Pilot 100 article embeddings and 10 generated explanations across five
   languages. Record wall time and extrapolate before a full run.
4. Copy `example.json` and adjust local paths. Run
   `python -m experiments.run experiments/example.json` from the repository root.
5. The runner fits TF-IDF, LSA, frozen transformer heads, and (ST2) graph variants.
   Three graph seeds are fixed at 13/42/87. All preprocessing is fit on training
   only; regularization/threshold selection uses development only. GCN checkpoint
   selection uses development BCE; the final threshold uses development F1.
6. Run TextRank, semantic extraction, semantic generation, and lead-evidence
   generation with the same evaluation IDs. Gold-label and predicted-label runs
   are separate. ST3 labels must be actual dominant-label annotations, not the
   first entry in ST2's multi-label set.
7. If generation projects beyond eight hours, select at most 250 articles BEFORE
   examining outcomes, balanced by language/domain, seed 42. Evaluate every ST3
   variant on that identical subset; disclose missing strata and sample sizes.
8. Validate official TSVs with the organizer scorers. Local metrics remain marked
   unverified until scorer parity is checked. Specify the organizer's BERTScore
   model/layers/settings; never replace its encoder with MiniLM.
9. Publish aggregate results only, with split hashes, commit, configuration,
   sample counts, runtime, measured peak RAM, and limitations. Keep raw predictions
   private. The website must not label smoke tests or leaderboard numbers as ours.

Resume an interrupted classification run with
`python -m experiments.run experiments/example.json --resume`. A saved `run.json`
must match configuration, dataset/taxonomy hashes, and relevant source hashes.
Completed predictions and trained models are reused; an interrupted prediction
file is regenerated from its model. Original prediction timing is retained when
available. JSON, model checkpoints, and embedding-cache writes use replacement
files so an interruption does not overwrite the previous complete checkpoint.
Older runs without a manifest stay intact; start a new directory and reuse their
embedding cache rather than guessing their provenance.

After reviewing real-data identity, split integrity, and release permission, use
`python -m experiments.publish artifacts/st2-comparison/summary.json --reviewed`
to export aggregate metrics and an SVG comparison plot. Do not use this on toy
data. The exporter preserves local-scoring provenance and includes no article text.

Required ablations: graph with only self-loops; random label initialization;
semantic versus leading evidence under the same generator input budget. Preserve
architecture, examples, seeds, and training settings within each ablation.

The transformer/logistic-regression versus GCN comparison changes the prediction
head as well as the graph. It cannot isolate the graph's contribution. Use the
matched GCN versus self-loop-only ablation for that question. The runner records
paired bootstrap intervals for both graph ablations using per-article scores
averaged across the same three seeds; these intervals describe article-sampling
uncertainty, while the separate seed standard deviation describes training variation.

Optional free cloud compute can accelerate the exact same runs only if allowed by
the data agreement. Inference is the reproducibility target. No neural model
fine-tuning or paid service is necessary.

## Manual explanation review

Review up to 50 fixed examples, balanced by language and domain. For each record
check: source evidence support, attribution, label relevance, language, readability,
and <=80 words. Preserve unsupported allegations as attributed claims. Reviewers
must record which languages they can assess; machine translation is assistance,
not an unqualified human fluency judgment.

## ST3 runner

With a running local generator:

```powershell
python -m experiments.explanations --data data/st3/test.jsonl --taxonomy data/taxonomy.json --gguf path/to/model.gguf --out artifacts/st3-comparison
```

This selects at most 250 examples using seed 42 and round-robin language/domain
strata BEFORE generation, then runs all four gold-label variants on identical IDs.
Set `--limit` to the full corpus size when the timing pilot permits. Add
`--predicted-labels artifacts/st2-predictions.jsonl` for the end-to-end variant.
Generation resumes from matching signatures. Add `--bert-model` and `--bert-layers`
from the official scorer to compute BERTScore; otherwise no explanation-quality
score is claimed. The overview paper identifies the encoder as
`google-bert/bert-base-multilingual-cased`; verify layers and remaining settings
against the scorer before official comparisons.

## Current limitations

Authorized corpus and organizer scorers are not available in this checkout.
Public auxiliary experiments are complete; see [measured results](../results/README.md).
They do not establish official Task 10 performance or multilingual explanation
quality. Runtime records are local measurements with cache/process limitations.
