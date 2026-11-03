# Public auxiliary experiments

These experiments address related components while the SemEval-2025 Task 10
corpus is unavailable. **They are not Task 10 benchmark results.** The native
annotations are retained; no sentiment, inference, or CARDS label is silently
converted into a SemEval role or narrative.

## Sources and limits

| Release | Experiment | Important limitation |
|---|---|---|
| [Cleaned CARDS/4C and EMM snippets](https://github.com/jpiskorski/CC-denial-resources) | Native hierarchical climate-claim classification | English paragraphs, one label each; different taxonomy from Task 10 |
| [YASO](https://github.com/IBM/yaso-tsa) | Native targeted sentiment classification | Clear-text subset only; sentiment is not entity framing |
| [e-SNLI](https://github.com/OanaMariaCamburu/e-SNLI) | Native inference explanations against human references | English premise/hypothesis pairs, not news narratives |

The task organizers explicitly list CARDS/4C and YASO as related resources. e-SNLI
is discussed in their annotation guidelines as related explanation work.

Download and prepare:

```powershell
python -m experiments.public_data
```

The downloader pins author-repository revisions and SHA256 checksums. Raw texts,
source metadata, and generated data stay in ignored `data/public/`; they are not
redistributed by this repository. The cleaned climate release has no separate
dataset license file in its repository: its authors publish it for research.
YASO specifies CDLA-Sharing-1.0 for its data (the code's Apache license is separate).
The SNLI base corpus specifies CC BY-SA 4.0; retain attribution to e-SNLI's authors
for their additional explanations. No source license is inferred from a paper's
license, and no raw corpus is uploaded to our website.

## Fixed classification protocol

```powershell
python -m experiments.run data/public/cards/experiment.json
python -m experiments.run data/public/yaso/experiment.json
```

CARDS preparation retains native codes such as `1_1` and `0_0`. Fine categories
are connected to the native six top-level classes; short descriptions paraphrase
Coan et al. (2021), Table 3. The provided test split is held out. Training/development
come from a seed-42 stratified split of the cleaned training file, then are capped
at 6,000/1,000 examples using stratified sampling, before observing model results.
Exact duplicate paragraphs are grouped, conflicting duplicate labels are removed,
and held-out duplicates are excluded from training. EMM snippets are a separate
out-of-domain evaluation source, not additional training data.

YASO preparation uses only supplied clear text with matching target offsets and
annotation confidence >=0.7. Hidden text is neither restored nor guessed. It keeps
the native positive/negative/mixed/none labels and creates a researcher-defined
60/20/20 review-group split using seed 42. This is not YASO's official evaluation
protocol, and review-domain coverage must be disclosed with results.

Native single-label classifiers use argmax, with development macro F1 for model
selection. The primary metric is macro F1 over the complete fixed label inventory;
accuracy, per-class F1, and counts are secondary. Graph variants run at seeds
13/42/87. Paired intervals use article-level accuracy, explicitly labelled as such;
macro-F1 seed means and variation are reported separately.

## Evidence requirements

Publish only aggregate scores and figures derived from saved predictions and
split hashes. Preserve dataset names and task limits in the website and report.
Report negative results. Do not treat author leaderboard scores, synthetic checks,
template explanations, or model-generated rationales as human gold annotations.
