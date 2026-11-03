# Multilingual Narrative Understanding

CPU-first implementation of **SemEval-2025 Task 10**: entity framing, narrative
classification, and evidence-grounded explanations. Focus: subtasks 2 and 3.

**Status:** implementation in progress. No benchmark scores have been measured.
Restricted news data is not included. Synthetic checks are not research results.

## Setup

Use Python 3.11–3.13 in a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Transformer/GNN/BERTScore experiments additionally require
`python -m pip install -r requirements-models.txt`. The generator uses a local
GGUF model and `llama-server`, installed separately; no paid API is required.

## Data

Request authorized data and scorer access from the
[official task website](https://propaganda.math.unipd.it/semeval2025task10/).
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

Further commands will be documented as the corresponding components are verified.

## References

- [Task definition, metrics, data conditions](https://propaganda.math.unipd.it/semeval2025task10/)
- [Organizer overview paper](https://aclanthology.org/2025.semeval-1.331/)
- [Narrative taxonomy](https://propaganda.math.unipd.it/semeval2025task10/NARRATIVE-TAXONOMIES.pdf)
- [Entity roles](https://propaganda.math.unipd.it/semeval2025task10/ENTITY-ROLE-TAXONOMY.pdf)

