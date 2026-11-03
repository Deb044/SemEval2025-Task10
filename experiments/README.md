# Experiment protocol

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

Required ablations: graph with only self-loops; random label initialization;
semantic versus leading evidence under the same generator input budget. Preserve
architecture, examples, seeds, and training settings within each ablation.

Optional free cloud compute can accelerate the exact same runs only if allowed by
the data agreement. CPU inference is the reproducibility target. No neural model
fine-tuning or paid service is necessary.

## Manual explanation review

Review up to 50 fixed examples, balanced by language and domain. For each record
check: source evidence support, attribution, label relevance, language, readability,
and <=80 words. Preserve unsupported allegations as attributed claims. Reviewers
must record which languages they can assess; machine translation is assistance,
not an unqualified human fluency judgment.

## Current limitations

Authorized corpus and organizer scorers are not available in this checkout.
No official benchmark results, throughput estimates, peak-RAM measurements, or
scientific improvement claims have been established.
