# Project website

Buildless HTML, CSS and JavaScript with three tabs: Overview, Experiments, Analyze.
The central architecture is the cross-lingual transformer plus hierarchical label
GCN. All current and historical measured studies are in one experiment catalog;
filters and expandable details retain metric units, seeds, source/data hashes,
per-language scores, error diagnostics and interpretation limits.

```powershell
python -m experiments.catalog
python -m semeval.web
```

Open http://127.0.0.1:8765. The local service loads only registered project GNN
checkpoints, keeps pasted text local, and can generate explanations through the
loopback generator. Optional manual review submissions stay under ignored
artifacts. The main grounding study uses independent LLM/NLI model-based review,
not human validation.

The hosted version is static. It displays aggregate experiments and provides a
link to the local workspace; it does not host model endpoints or raw corpus text.
`dist/experiments.json` is generated from reviewed aggregate evidence by
`experiments.catalog`. Historical aggregates are retained in `results/history.json`.
Archived LLM assessments are labelled frozen, and never presented as a new review
of regenerated outputs. The judge's exact runtime identifier was unavailable;
use the pinned NLI diagnostic for a separately repeatable automatic measure.

Preserve the identity and private audience in `.openai/hosting.json` when using
the Sites publishing workflow. No installation or build step is needed for the
static assets. Serve `dist/` with any static HTTP server for presentation-only use.
