# Spark SQL data-quality gate

`python -m experiments.reproduce --stage spark` audits all six prepared CARDS
and ProBel classification splits with Apache Spark 4.2.0. The same stage runs
before classification in the standard reproduction command. It does not alter
input records, labels, splits or model checkpoints.

Install `requirements-spark.txt` and a supported Java 17, 21 or 25 runtime.
`JAVA_HOME` should point to that Java installation. The default Spark master is
`local[2]`, using two task threads on one machine. No cluster performance is claimed.

Spark reads the JSONL files with an explicit schema, tags dataset/split boundaries,
and executes SQL that rejects malformed or missing fields, empty splits, repeated
article IDs, normalized exact-text duplicate groups, and cross-split overlap within
each dataset. SQL also aggregates record/character/label counts by language and
label frequencies. Only these aggregates and provenance are exported. Translation
pairs and semantic near duplicates require additional checks.

The prepared inputs remain JSONL. Spark performs ingestion, SQL aggregation and
validation; DuckDB performs the separate Parquet scalability study. Spark's native
Parquet writer is not required, avoiding extra Windows Hadoop native components.
The successful stage is a gate in the reproduction workflow; invoking an individual
model module directly does not run it automatically.

To run the focused integration tests in PowerShell:

```powershell
$env:RUN_SPARK_TESTS = '1'
python -m unittest discover -s tests -p test_spark_audit.py -v
```

On Windows, Spark launcher scripts require a temporary directory without spaces.
If needed, set `TEMP` and `TMP` to an existing writable path without spaces before
launching Python. This does not require changing global Windows settings.

The local summary is `artifacts/spark-audit/summary.json`. Export it with
`python -m experiments.catalog`; the aggregate copy is `results/spark-audit.json`
and the website adds a typed timing entry with checks and counts in its details.
JVM startup and SQL audit time are recorded separately. One local run does not
establish a speedup over Python or distributed scaling.

Reference: [Apache Spark documentation](https://spark.apache.org/docs/4.2.0/).
