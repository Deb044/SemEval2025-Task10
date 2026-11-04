# Multilingual narrative and propaganda analysis

A cross-lingual transformer encodes news text and label descriptions. A trainable
**hierarchical label-graph GNN** connects parent and child categories and scores
fine-grained labels. Evidence selection and local generation produce explanations
of up to 80 words. The project follows SemEval-2025 Task 10, emphasizing narrative
classification and grounded explanations.

Public CARDS/4C and ProBel data support the main studies; native dataset labels
remain distinct from official SemEval labels. MiniLM is the primary encoder,
with frozen XLM-R as an encoder comparison. Graph controls cover self-loops,
shuffled hierarchy edges, and random label vectors. TF-IDF and LSA are baselines.
[Project website](https://semeval-narrative-lab-deb044.solidflask.chatgpt.site) ·
[All experiments](results/experiments.json). Results retain their provenance and limits.

## Setup

Use Python 3.11–3.13 and Java 17, 21 or 25 from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-models.txt -r requirements-spark.txt
python -m experiments.download_models
```

The model command downloads pinned resources into ignored local folders. Install
[llama.cpp b11960](https://github.com/ggml-org/llama.cpp/releases/tag/b11960) for
generation; pass its `llama-server` executable with `--server` when it is not on
PATH. Model weights, texts, individual predictions and review submissions stay local.
Spark uses `local[2]` to check invalid records, duplicate groups and split overlap,
and compute language/label counts before training. It preserves the model inputs.
DuckDB remains responsible for the separate Parquet scalability experiments.
See [Spark setup and checks](docs/spark.md) for the SQL stage and Windows notes.

## Reproduce

```powershell
python -m experiments.reproduce --stage all --server path/to/llama-server
```

This prepares public data, runs an Apache Spark SQL input audit, then runs fixed
classification matrices and EMM transfer,
compares longer-text explanations, measures data/embedding scalability, and exports
the experiment catalog. It starts and stops its own local generator. Use
`--stage prepare`, `spark`, `classification`, `source-transfer`, `explanations`,
`generators`, `scalability`, `grounding`, or `export` to run a single stage. Compatible classification
checkpoints resume; completed other stages are reused only after source and input
hash validation. Changed experiments need fresh output directories.

```powershell
python -m unittest discover -s tests -v
python -m semeval.web
```

Open **http://127.0.0.1:8765** for the pipeline, every experiment, local GNN analysis,
and the blinded explanation review queue. Start the documented local generator
when requesting interactive explanations. The hosted website presents aggregates;
text analysis and review run locally.

Historical studies are retained with their original source commits. Current
reproduction is defined by [fixed configurations](experiments/configs),
[public-data protocols](experiments/PUBLIC_ALTERNATIVES.md), and
[optional official-format commands](docs/task-formats.md). The generator comparison tests Qwen3, SmolLM2 and TinyLlama on identical inputs.
A pinned Qwen2.5-3B-Instruct judge reviews blinded raw outputs; NLI remains a
separate diagnostic. The reported comparison uses LLM review.
See [the generation/review protocol](docs/generator-comparison.md).
