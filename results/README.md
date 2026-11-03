# Measured auxiliary experiments

These are real local measurements on author-released English datasets.
**They are not SemEval-2025 Task 10 benchmark scores.** No official dataset or
organizer scoring scripts were available. Public alternatives are the selected
evaluation route; no official access is required to reproduce these experiments. Code for all three official task formats
is implemented; these experiments test related components with native labels.

| Dataset | Held-out examples | Methods | Primary metric | Best measured baseline |
|---|---:|---|---|---|
| Cleaned CARDS/4C | 2,897 paragraphs | TF-IDF, LSA, frozen MiniLM, label GCN and two ablations | Macro F1 over 18 native labels | LSA 0.551 |
| YASO clear-text subset | 305 targets in 95 sentences | TF-IDF, LSA, frozen MiniLM | Macro F1 over four native sentiments | LSA 0.439 |
| e-SNLI | 60 sentence pairs | Copied input, label template, local Qwen generation, premise withheld | Multi-reference BERTScore F1 | Copied input 0.806 |

## Bilingual news extension

[ProBel news protocol](../experiments/PUBLIC_ALTERNATIVES.md) adds native
multi-label persuasion-technique classification on English/Arabic news units.
The fixed researcher-defined source-article-grouped pilot uses 1,600 training,
400 development and 400 test examples (800/200/200 per language), a 23-technique,
six-parent hierarchy, and no tweets. It is not the publisher's benchmark split.

| ProBel news method | Macro F1 over all 23 techniques | Seed SD |
|---|---:|---:|
| Character TF-IDF + logistic regression | 0.0558 | — |
| LSA + logistic regression | 0.0911 | — |
| Frozen MiniLM + logistic regression | 0.0876 | — |
| Frozen MiniLM + full label GCN | 0.0862 | 0.0092 |
| GCN, self-loops only | 0.0995 | 0.0201 |
| GCN, random label-name vectors | 0.0892 | 0.0103 |

Graph rows average seeds 13/42/87; baselines use seed 42. The full graph does not
improve macro F1 here. Its paired samples-F1 difference from self-loops is −0.0011,
95% interval [−0.0100, 0.0080]. This interval is not a macro-F1 interval. Label
initialization uses publisher label names, rather than paraphrased definitions.

Scores are low, and should be reported as such. All 23 techniques occur in the
training pilot, 21 in development and 18 in testing; fixed-inventory macro F1 gives
zero to unsupported test classes. Rare-label support and source/language imbalance
limit this pilot. No claim of competitive full-dataset performance is made; no
model was retuned after inspecting these test outcomes. Per-language metrics and
native class supports are retained in diagnostics.

Source-language-only training gives the following fixed-inventory macro F1:

| Training language | Test language | MiniLM + logistic regression | MiniLM + GCN |
|---|---|---:|---:|
| English | English | 0.0337 | 0.0371 |
| English | Arabic | 0.0401 | 0.0187 |
| Arabic | English | 0.0343 | 0.0606 |
| Arabic | Arabic | 0.0808 | 0.0670 |

Each fitted transfer model uses source-language development data only and seed
42. Scores in different test languages are not controlled comparisons: their
sources and class distributions differ. The results demonstrate an executed
cross-lingual transfer protocol, not strong cross-lingual performance. These are
native persuasion techniques, not the Task 10 narrative or role labels.

## Bilingual propaganda explanations

The fixed 40-example ProBel news pilot supplies native technique labels and
withholds source spans and reference explanations from the prompts. BERTScore
compares our outputs with the publisher's **model-authored** references; those
references were human-evaluated by the publisher, not written by human annotators.
This does not constitute a human review of our generated outputs.

| Method | Overall BERTScore | English (20) | Arabic (20) |
|---|---:|---:|---:|
| TextRank extraction | 0.6836 | 0.6648 | 0.7024 |
| Semantic extraction | 0.6836 | 0.6648 | 0.7024 |
| Qwen with semantic evidence | 0.7098 | 0.6996 | 0.7201 |
| Qwen with leading evidence | 0.7098 | 0.6996 | 0.7201 |

All returned outputs meet the 80-word constraint; none used extraction fallback.
Both generator selectors choose exactly the same sentences on all 40 short news
units. The runner reuses the semantic generation when the leading-evidence prompt
is exactly equivalent (100% of this pilot). These are not 80 independent generator
outputs, and the equal scores provide no evidence about retrieval effectiveness
on longer articles. Ranking scores are excluded from all generator prompts.
Similarity scores do not establish grounding, language quality or dominant-narrative
explanation quality. Extraction selects the same text in both variants here too.

The classification source commit is `b86859b`, transfer is `adafc35`, and corrected
explanation generation is `21cf381`. Measured source and input hashes accompany
the exports. The initial explanation comparison is retained privately and excluded
from release because ranking metadata changed otherwise equivalent prompts.

## Findings and limits

CARDS baseline scores are TF-IDF 0.433, LSA 0.551, and frozen MiniLM 0.527.
Full label GCN averages 0.468 (sample seed SD 0.013), self-loops 0.458
(SD 0.021), and random label vectors 0.443 (SD 0.020), over seeds 13/42/87.
The graph does not improve on MiniLM/logistic regression in this experiment.
The matched graph versus self-loop accuracy difference is +0.0040, with paired
bootstrap 95% interval [−0.0018, +0.0099]. This interval concerns accuracy, not
macro F1, and includes zero. Definition vectors versus random vectors improve
accuracy by +0.0067 [0.0007, 0.0125]. These are exploratory comparisons; no
multiple-comparison correction or broad generalization claim is made.

YASO scores are TF-IDF 0.294, LSA 0.439, and MiniLM 0.408. Sentiment is not
entity framing. Its small, skewed clear-text subset has only six mixed-sentiment
test targets. The review-group split is researcher-defined, not the official
YASO evaluation protocol. Native per-class diagnostics accompany the averages.

e-SNLI BERTScore is 0.806 for copied input, 0.694 for the template, 0.733 for
full-context generation, and 0.712 with the premise withheld. Full context exceeds
the no-premise variant by 0.0210 [0.0142, 0.0282] in a paired bootstrap, yet copied
input scores higher than generation. Text overlap and similarity can reward
unexplanatory text. These scores establish neither factuality nor explanation
quality. The gold relation is supplied in each prompt; references are withheld.
All returned outputs satisfy the 80-word limit; full-context generation used one
template fallback out of 60. No human grounding audit is claimed.

The earlier CARDS/YASO/e-SNLI tasks are English-only; ProBel adds EN/AR auxiliary
classification and transfer. Multilingual Task 10 performance,
news-narrative explanation quality, near-duplicate/translation leakage beyond
exact matching, and organizer-scorer parity remain unverified. No model was
retuned after inspecting test results. EMM was prepared but not evaluated.

## Reproducibility and artifacts

[Public-data protocol](../experiments/PUBLIC_DATA.md) describes pinned author
releases and checksums, grouping/deduplication, fixed subsets, model revisions,
and native-label boundaries. CARDS uses 6,000 training and 1,000 development
paragraphs; YASO uses 834/310 targets in 264/90 sentences. Baselines use seed 42;
graph variants use 13/42/87. Configuration is generated by the data preparer.

- `measurements.json`: reviewed method/ablation and transfer aggregates, source
  and input hashes, local timing, seed SD, and paired intervals.
- `diagnostics.json`: native per-label/relation metrics and length/fallback rates.
- `*.svg`: separate plots for each dataset; values are not comparable across tasks.
- `website/dist/results.json`: the same measured aggregates used by the website.

The classification runs used source commit `71763c9`. Explanation checkpoints
were verified against executed-source signatures and committed in `ba2d37b`;
that source commit fixes the BERTScore API return handling. Subsequent code
hardening does not change these saved predictions or scores. Source hashes are
retained to identify the exact measured implementation. Windows line endings
can change byte hashes; regenerate into a fresh directory after source changes.

The generator is Qwen3-1.7B Q4_K_M (pinned revision in the protocol), llama.cpp
b11960, four threads, temperature zero, maximum 768 output tokens and
thinking disabled. The evaluator is multilingual BERT revision
`3f076fdb1ab68d5b2880cb87a0886f315b8146f8`, layer 9, BERTScore 0.3.13,
no IDF or baseline rescaling, and all available references. Bootstrap uses seed
42 and 2,000 resamples; differences are second method minus first. Classification
intervals average each article's accuracy over graph seeds before resampling.
This estimates article sampling uncertainty conditional on those trained models.

Timings include cache reuse where present and sequential shared-process effects;
they are not clean speed comparisons. Raw text, individual predictions, model
checkpoints and embeddings remain under ignored data/artifact/cache folders.
Only aggregate statistics are released.

To regenerate exports after reviewing real source/data identity:

```powershell
python -m experiments.publish artifacts/public-cards-v2/summary.json artifacts/public-yaso-v2/summary.json artifacts/public-esnli-v2/summary.json artifacts/public-probel/summary.json artifacts/public-probel-transfer/summary.json artifacts/public-probel-explanations-v2/summary.json --reviewed
```

Fresh runs may use the preparer's default output names instead of the local
`-v2` folders. Do not overwrite legacy runs or reinterpret smoke checks as results.
