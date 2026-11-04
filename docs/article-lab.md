# Architecture and article lab

The Overview diagram explains the measured auxiliary-data architecture. Article
text and label descriptions share a frozen cross-lingual transformer. Label
vectors pass through a two-layer parent–child GCN; a learned projection compares
article vectors with graph-enhanced label vectors. Development data selects the
native decoder settings. Predicted labels guide evidence ranking, then local
Qwen generation produces at most 80 words. Spark SQL audits training inputs
before fitting; it is not rerun for every pasted article.

## Official-taxonomy prototype

The public Analyze tab accepts pasted text or a UTF-8 `.txt` file, up to 20,000
characters/200 text units. Browser mode performs actual inference on the user's
device; it does not upload the article. Model/runtime files (about 280 MB in
aggregate) are downloaded on first use and cached by the browser.

The label inventory is built from pages 1–2 of the organizer's public
[NARRATIVE-TAXONOMIES.pdf](https://propaganda.math.unipd.it/semeval2025task10/NARRATIVE-TAXONOMIES.pdf),
SHA256 `e9b26b6e4a58bb28900572ea140adf2da9e7c4858aa8a26f9039774812d8a26e`.
The task page supplies the global/parent-specific Other convention. There are
22 coarse categories (including global Other) and 96 fine labels. Full fine IDs
include domain, parent and subnarrative. Label spellings and Unicode punctuation
follow this public PDF profile. Exact parity with a private organizer scorer
release is not claimed; use its released inventory before official submission.

Rebuild this inventory with `python -m experiments.official_taxonomy`.
`semeval/taxonomies/official.json` and `website/dist/official-taxonomy.json` are
identical copies. The builder rejects a changed PDF checksum or unexpected counts.
Working NLI hypotheses are derived from the labels, not organizer annotations or
gold explanation text.

The official-vocabulary mode is **separate from the trained CARDS/ProBel GNNs**:

1. Preserve source text and split long sentences into units of at most 32 words.
2. Encode source units and label descriptions with multilingual MiniLM.
3. Retrieve up to eight candidate fine labels within the selected domain.
4. Use multilingual MiniLMv2 NLI on each candidate's nearby source context.
5. Keep candidates with semantic similarity >=0.30, entailment-vs-contradiction
   agreement >=0.60, three-way entailment >=0.10, and a combined score
   (0.8 × semantic similarity + 0.2 × agreement) within 0.05 of the
   best surviving candidate. Show the strongest candidate by default; an explicit
   option includes up to three. All cutoffs are exploratory, and the full ranking
   remains inspectable.
6. Suppress a parent's Other when a concrete child survives, and validate every
   result against the pinned hierarchy. Return global Other when none survive.
7. Highlight supporting source units and provide an extract of at most 80 words.
   This browser extract is not LLM generation. Low-scoring candidates remain
   inspectable in the ranking; unmeasured NLI values remain null.

Entailment-vs-contradiction scores ignore neutral mass, so they are reported
alongside a minimum three-way entailment criterion. They are model agreement
proxies, not calibrated probabilities of truth. The prototype can abstain,
miss narratives or select incorrect candidates. Structural vocabulary validation
does not establish classification accuracy. No official Task 10 evaluation data
or official-label GNN checkpoint is available, and no CARDS/ProBel score has been
relabeled as official performance. Invented UI examples are smoke tests only.

## Model provenance

- Browser encoder: `Xenova/paraphrase-multilingual-MiniLM-L12-v2`, revision
  `2c4055b12046f11709e9df2c122e59ffbdc2f900`, quantized ONNX.
- Claim checker: `onnx-community/multilingual-MiniLMv2-L6-mnli-xnli-ONNX`, revision
  `ca5daf3d11b6c4b3143b1f4602a2edfb64c3ad7e`, ONNX checksum
  `55614f3c7da74184742eaa0006b978744437aa91de9ba4913db42f94d7844a8f`.
- Browser runtime: `@xenova/transformers@2.17.2`, using one WASM thread in a worker.
- Local retrieval uses the project's pinned PyTorch MiniLM encoder; local claim
  checking uses the same quantized ONNX NLI weights with ONNX Runtime 1.23.2.
  Runtime/tokenizer differences mean browser/local scores need not be identical.

## Full local workspace

After normal README setup, start the existing Qwen generator in one terminal:

```powershell
work/llama-b11960/llama-server.exe -m work/models/Qwen3-1.7B-Q4_K_M.gguf --host 127.0.0.1 --port 8080 -c 4096 -t 4 -ngl 0 -np 1 --jinja
```

In another terminal:

```powershell
python -m semeval.web
```

Open `http://127.0.0.1:8765/#analyze`, select Local workspace mode, and optionally
request Qwen generation. That paragraph explains the highest-ranked candidate;
invalid generation uses a marked extractive fallback. The API returns label and
evidence results even when optional generation is unavailable. It does not save
pasted article text. Model caches may store hashed embeddings, as elsewhere in
the project.

The Advanced section also runs the actual trained MiniLM/XLM-R + GNN checkpoints
on articles, with native category names and a reported execution trace. This
keeps the measured assignment architecture testable without silently pretending
its heads were trained on the official narrative inventory.

Checks: `python -m unittest discover -s tests -v` and
`node tests/browser-taxonomy.mjs`. Browser smoke tests exercise invented climate,
war and unrelated-topic examples; they do not create benchmark scores.
