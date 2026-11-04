# Controlled generator comparison and named review

Three Q4_K_M generators are compared: Qwen3-1.7B, SmolLM2-1.7B-Instruct and
TinyLlama-1.1B-Chat-v1.0. The immutable repository revisions and SHA256 hashes
are in `experiments/configs/generators.json`. Weights remain ignored locally.

```powershell
python -m experiments.reproduce --stage generators --server path/to/llama-server
python -m experiments.reproduce --stage export
```

The runner downloads missing pinned weights and starts/stops each loopback-only
llama.cpp b11960 process. Completed outputs resume only if source, input, case,
configuration and binary hashes agree. The main comparison uses the same fixed
24 English CARDS test passages selected before generation, supplied native claim
labels, and identical prompt text and selected evidence for every model. It is
separate from the earlier longer-evidence and predicted-label studies.

Evidence has a common budget of 800 Qwen-tokenizer tokens and six sentences,
including neighboring context. All servers use a 2,048-token context, four task
threads and one request slot; prompts are checked against a reserved context
budget. Generation uses temperature 0, seed 42, a 256-token output limit and no
retry. Native chat templates necessarily differ across models. We report raw
word compliance, successful complete generation rate, fallback rate and median
request latency. Model loading is excluded from that latency. Truncated or
invalid raw generations remain recorded; they do not receive quality credit
from an extractive fallback.

The named judge is **Qwen2.5-3B-Instruct**, Q4_K_M, from the official Qwen GGUF
release. The exact revision/checksum is pinned in the same manifest. It reviews
18 blinded raw outputs: the same first six fixed passages from each generator,
shuffled with seed 42. Its inputs contain the full source with sentence IDs,
the target description and candidate text, without generator identity. It rates
source support, attribution, target relevance and readability on 0/1/2 scales.
The full rubric, 8,192-token context, JSON output schema, temperature 0 and seed
42 are recorded. These are judgments from one quantized model, not human gold
or world-truth labels; the shared Qwen family may favor the Qwen generator.

The displayed mean averages the four rubric criteria and six reviewed outputs
per generator. Fully supported means the support rating equals 2. The judged
subset is smaller than the 24-output format/latency cohort. No universal model
ranking or multilingual generation claim follows from this small English pilot.

## Actual human review and pairing

Human reviews remain optional. To review this exact raw-output queue locally:

```powershell
python -m semeval.web --review-folder artifacts/generator-comparison
```

Submit ratings in the manual review workspace, then rerun
`python -m experiments.named_review` and the export command. Every human record
must have an explicit human/language confirmation and match the exact item hash
(source, target and raw explanation). Older unbound interface-check submissions
are preserved privately and are not counted in this comparison.

For each criterion and matched item, the combined score is:

`(LLM rating + mean of actual human ratings for that item) / 2`.

Combined means average only those matched items, with their coverage reported.
If no eligible human ratings exist, human and combined scores are null, never
invented, copied from the LLM, or described as a completed human study. The
featured plot labels LLM-only scores explicitly; partial paired results remain
separate in the aggregate evidence.

Only aggregate summaries, protocols, model IDs, checksums and plots are public.
Source articles, raw generations, judge rationales and reviewer aliases stay local.
