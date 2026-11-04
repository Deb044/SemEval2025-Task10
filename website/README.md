# Project website

Buildless HTML/CSS/JavaScript in `dist/` presents the transformer–GNN pipeline,
five detailed studies with scientific SVG plots, and a compact appendix containing
all remaining aggregate results. The hosted site is static; analysis and manual
review use the loopback-only Python workspace.

```powershell
python -m experiments.reproduce --stage export
python -m semeval.web
```

Export reads completed, provenance-bearing experiments, builds the full catalog,
and generates `studies.json` plus five plots with `experiments.website_studies`.
The same study payload is saved as `results/featured-studies.json`. Plot rebuilding
uses committed aggregates and does not require publishing article text. The new
generator comparison and named review must be completed before creating that
payload; missing scores are never synthesized.

The five studies cover native hierarchical classification, bilingual encoders,
matched graph controls, three generators with Qwen2.5-3B-Instruct review, and the
Spark/DuckDB/embedding-cache data pipeline. A judge score is an explicitly named
model-based rating. Human and LLM scores are combined only for actual matching
human submissions, with paired coverage stated. Historical exploratory audits
retain their original limitations in the appendix.

The site manifest in `.openai/hosting.json` selects the existing Sites project.
Source publication and access settings are managed through Sites. Raw data,
generations, review rationales, weights and reviewer identities stay ignored.
