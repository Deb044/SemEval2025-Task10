# Subtask 1: entity framing

This implementation predicts the role assigned to a supplied entity mention in
an article, using the public SemEval-2025 Task 10 hierarchy: three broad roles
(Protagonist, Antagonist, Innocent) and 22 fine roles. It does not convert YASO
sentiment labels into entity roles and does not substitute named-entity detection
for the task's supplied mention list.

## Exact input and offsets

Each mention is an exact occurrence with an article ID, language, text, surface
mention, start position and end position. Original source text is preserved.
The existing importer accepts release-specific inclusive/exclusive end offsets.
The new JSON/API entry also accepts explicitly named `codepoint` or `utf16`
units and converts browser positions to Unicode codepoints. A wrong mention,
duplicate occurrence, out-of-range position or split surrogate pair is rejected;
the code never silently selects a different occurrence of the name.

The UI lets users enter an exact name and select/add its occurrence. It supports
multiple supplied targets within one article. Export uses the original entity
spelling, one broad role and all selected fine roles as separate TSV fields.
There is no invented `Other` role. An abstained prototype result cannot be
exported as an official submission row.

Official taxonomy:
https://propaganda.math.unipd.it/semeval2025task10/ENTITY-ROLE-TAXONOMY.pdf
SHA256: `c55d1692f020cbd5f13135d6edfc76bbb90efe8fe6190a396051e4be47e21f59`.
`python -m experiments.entity_taxonomy` checks the pinned PDF and rebuilds the
identical backend/browser inventories. Definitions and NLI hypotheses are concise
project paraphrases, not copied organizer examples. Exact inventory is checked;
parity with an inaccessible organizer scorer is not claimed.

## Supervised transformer + role GNN

The dedicated model uses frozen multilingual MiniLM (or optional XLM-R) and
concatenates three views:

1. The target entity's name.
2. An entity-specific context with only the supplied occurrence marked using
   ordinary tokenizer text `[TARGET] ... [/TARGET]`. Nearby sentences retain
   attribution and negation where they fit within the bounded context.
3. The article representation, using the existing token-chunk pooling/cache.

A two-layer GCN contains 25 nodes: 22 fine roles and three parents. Label-name
and role-description embeddings initialize fine nodes. Parent-child edges and
self-loops use the official hierarchy. A learned entity projection scores both
fine and parent nodes. Joint training combines weighted fine-role binary
cross-entropy with parent cross-entropy. Development loss controls early
stopping; development fine-role exact match selects the decoding threshold.

The decoder selects one broad role and thresholds compatible fine roles using
parent-weighted scores. It retains at least the highest-scoring compatible fine
role for submission-format predictions. Forced nonempty output is not evidence
that an unframed/neutral target has an identifiable role. Scores are not
calibrated probabilities. A separate `unmarked` control uses only local context;
it removes both target-specific and global views, so it is not a marker-only
causal ablation.

## Runnable zero-shot baseline

The public browser mode uses pinned multilingual MiniLMv2 NLI, evaluating all
22 role hypotheses against the sentence containing the supplied occurrence.
Only that occurrence is normalized to `The target`; other names remain intact.
This reduces cross-entity contamination compared with scoring a whole article.
Source offsets and the unmodified displayed sentence remain available.

The strongest fine-role score selects a broad role, then compatible roles must
reach three-way entailment 0.25 and be within 85% of the strongest compatible
score. These are exploratory thresholds, not calibrated official settings.
Neutral mass is not discarded. The model can abstain. It can still confuse
attribution, negation and roles; short sentence context loses wider discourse.
Browser and local tokenizer/runtime differences can produce different scores.
No GNN is run in browser NLI mode. Local NLI uses the same pinned quantized
weights from the official-taxonomy prototype. Text is not uploaded in browser
mode. The local API is loopback-only.

## Training data and evaluation limits

The official annotated corpus remains unavailable. The architecture can train
on authorized official-format annotations when supplied. To exercise the real
training/inference path now, an optional small pilot uses public source excerpts
from CARDS (English), Slavic persuasion data (Bulgarian/Russian), H-Prop-News
(Hindi) and a Portugal-publisher PartisanLens subset. Native task labels are
not passed to the annotator or relabelled as roles.

Pinned local Qwen2.5-3B-Instruct Q4_K_M selects at most one clearly portrayed
named entity per excerpt and supplies roles plus an exact source quote. Candidates
must match the source exactly, with an unambiguous occurrence and a valid
single-parent role combination. Missing/ambiguous/truncated outputs are rejected;
neutral passages can abstain. Source splits are chosen before annotation, and
article/duplicate groups are checked across partitions. Up to 70 passages are
considered, with source excerpts limited explicitly to 1,500 characters.

These are **silver annotations from one model**, not independently verified gold.
Held-out pilot scores measure agreement with the same teacher. Fine-role coverage,
language counts, abstentions/rejections and teacher provenance are exported;
unknown/absent fine-role coverage is not hidden. This pilot does not establish
real entity-framing accuracy or multilingual generalization. Raw excerpts,
annotations, rationales, predictions and weights stay in ignored folders.

The dedicated evaluator reports fixed-inventory fine-role exact match, macro F1,
micro F1, sample F1, parent accuracy and per-language/per-label statistics. It
aligns by article/mention offsets, checks exact target identity and rejects
invalid parent/fine combinations. The official primary metric is fine-role
exact match. Local scorer parity remains unverified.

## Run

Use the root README environment and model setup. Start the website:

```powershell
python -m semeval.web
```

At http://127.0.0.1:8765/#analyze expand **Subtask 1**, enter an article, add exact
target occurrences and run the browser baseline. The first run downloads about
110 MB of NLI weights plus tokenizer/runtime resources. The invented example is
an interface smoke test, not a benchmark. Local modes become available when the
Python service and relevant checkpoint are present.

For the named-teacher pilot, first prepare the public sources as documented in
the root README and `docs/multilingual.md`, and install llama.cpp b11960. The runner downloads missing
pinned judge weights from `experiments/configs/generators.json`. Then run:

```powershell
python -m experiments.entity_framing --stage all
```

This owns a local teacher process on port 8084, writes silver JSONL under
`data/public/entity-silver-v1/`, trains the two controls, and writes checkpoints
under `artifacts/entity-silver-v1/`. The UI's local GNN option uses `full.joblib`.
Results are aggregate-only in `results/entity-framing.json`. Repeating the same
annotation inputs reuses saved outputs; changed pilots need fresh output folders.
The teacher runner currently reruns training when requested.

For your own authorized role annotations, import with the correct release
convention and train the dedicated model:

```powershell
python -m semeval import --task 1 --articles data/raw/train-articles --annotations data/raw/train.tsv --offset-end exclusive --out data/st1/train.jsonl
python -m semeval import --task 1 --articles data/raw/dev-articles --annotations data/raw/dev.tsv --offset-end exclusive --out data/st1/dev.jsonl
python -m semeval.entity_cli train --train data/st1/train.jsonl --dev data/st1/dev.jsonl --out artifacts/entity-role-gold/full.joblib
python -m semeval.entity_cli predict --model artifacts/entity-role-gold/full.joblib --data data/st1/test.jsonl --out artifacts/entity-role-gold/predictions.jsonl
python -m semeval.entity_cli evaluate --gold data/st1/test.jsonl --predictions artifacts/entity-role-gold/predictions.jsonl --out artifacts/entity-role-gold/metrics.json
python -m semeval.entity_cli export --predictions artifacts/entity-role-gold/predictions.jsonl --out artifacts/entity-role-gold/submission.tsv
```

Import the test partition separately before prediction. For unlabeled official
ST1 input, retain the supplied mention/offset columns and use the existing
importer's `--unlabeled` option. Only load trusted project checkpoints.

Checks:

```powershell
python -m unittest discover -s tests -p test_entity.py -v
node tests/browser-entity.mjs
```

Known baseline limitation: smoke tests exposed fine-role confusion on simple
multi-entity passages. The NLI mode is experimental and does not establish
improved entity-framing accuracy. The supervised GNN is the main implementation.
