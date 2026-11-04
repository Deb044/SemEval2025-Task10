"""Audit model inputs with Apache Spark SQL before training.

Only aggregate counts are exported. Source text and the prepared JSONL inputs
remain unchanged. Spark runs locally by default; no cluster speedup is claimed.
"""
import argparse
import json
import os
import time
from pathlib import Path

from experiments.common import provenance
from semeval.data import write_json

INPUTS = {f"{dataset}/{split}": Path(f"data/public/{dataset}/{split}.jsonl")
          for dataset in ("cards", "probel") for split in ("train", "dev", "test")}
SCHEMA = "article_id STRING, language STRING, text STRING, task INT, labels ARRAY<STRING>, _corrupt_record STRING"


def audit(spark, inputs):
    """Run SQL checks over explicit dataset/split boundaries, without repairing rows."""
    from pyspark.sql import functions as F
    frames = []
    for key, path in inputs.items():
        dataset, split = key.split("/")
        frame = spark.read.schema(SCHEMA).option("mode", "PERMISSIVE").json(str(Path(path).resolve()))
        frames.append(frame.withColumn("dataset", F.lit(dataset)).withColumn("split", F.lit(split)))
    if not frames:
        raise ValueError("Provide at least one dataset split")
    rows = frames[0]
    for frame in frames[1:]:
        rows = rows.unionByName(frame)
    # Python str.split treats Unicode whitespace and U+001C..U+001F as separators.
    normalized = F.trim(F.regexp_replace(F.col("text"), r"(?U)[\s\x{001C}-\x{001F}]+", " "))
    rows = rows.withColumn("text_hash", F.sha2(normalized, 256)).cache()
    rows.createOrReplaceTempView("news_inputs")
    try:
        counts = [row.asDict() for row in spark.sql("""
            SELECT dataset, split, language, count(*) AS records,
                   sum(length(text)) AS characters,
                   sum(size(labels)) AS label_assignments
            FROM news_inputs GROUP BY dataset, split, language
            ORDER BY dataset, split, language
        """).collect()]
        invalid = spark.sql("""
            SELECT count(*) AS n FROM news_inputs
            WHERE _corrupt_record IS NOT NULL OR article_id IS NULL OR trim(article_id) = ''
               OR text IS NULL OR text_hash = sha2('', 256) OR language IS NULL
               OR language NOT IN ('en', 'ar') OR task IS NULL OR task <> 2
               OR labels IS NULL OR exists(labels, x -> x IS NULL OR trim(x) = '')
        """).first().n
        empty = [key for key in inputs if not any(row['dataset'] + '/' + row['split'] == key for row in counts)]
        overlaps = {}
        duplicates = {}
        for column in ("article_id", "text_hash"):
            overlaps[column] = spark.sql(f"""
                SELECT count(*) AS n FROM (
                    SELECT dataset, {column} FROM news_inputs
                    GROUP BY dataset, {column} HAVING count(DISTINCT split) > 1
                )
            """).first().n
            duplicates[column] = spark.sql(f"""
                SELECT count(*) AS n FROM (
                    SELECT dataset, split, {column} FROM news_inputs
                    GROUP BY dataset, split, {column} HAVING count(*) > 1
                )
            """).first().n
        if invalid or empty or any(overlaps.values()) or any(duplicates.values()):
            raise ValueError(f"Spark input audit failed: invalid={invalid}, empty={empty}, overlaps={overlaps}, duplicate_groups={duplicates}")
        labels = [row.asDict() for row in spark.sql("""
            SELECT dataset, split, language, label, count(*) AS records
            FROM news_inputs LATERAL VIEW explode(labels) expanded AS label
            GROUP BY dataset, split, language, label ORDER BY dataset, split, language, label
        """).collect()]
        return {"records": sum(row['records'] for row in counts), "counts": counts,
                "label_counts": labels, "invalid_records": invalid,
                "cross_split_overlap_groups": overlaps, "within_split_duplicate_groups": duplicates,
                "empty_splits": empty, "passed": True}
    finally:
        rows.unpersist()
        spark.catalog.dropTempView("news_inputs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='artifacts/spark-audit/summary.json')
    args = parser.parse_args()
    import pyspark
    from pyspark.sql import SparkSession
    # Avoid Windows launcher ambiguity when the virtualenv path contains spaces.
    os.environ.setdefault('SPARK_HOME', str(Path(pyspark.__file__).parent))
    os.environ.setdefault('SPARK_LOCAL_IP', '127.0.0.1')
    started = time.perf_counter()
    spark = (SparkSession.builder.master('local[2]').appName('NarrativeLab-InputAudit')
             .config('spark.ui.enabled', 'false').config('spark.driver.bindAddress', '127.0.0.1')
             .config('spark.sql.shuffle.partitions', '2').getOrCreate())
    startup = time.perf_counter() - started
    spark.sparkContext.setLogLevel('ERROR')
    try:
        started = time.perf_counter()
        result = audit(spark, INPUTS)
        result.update({"tool": "Apache Spark SQL", "spark_version": spark.version,
                       "master": spark.sparkContext.master, "startup_seconds": startup,
                       "audit_seconds": time.perf_counter() - started,
                       "limitations": ["Local execution with two task threads; no distributed performance claim.",
                           "Exact normalized-text and article-ID overlap checks within each dataset; translation and semantic near-duplicate detection are not established.",
                           "Inputs and labels are preserved; this stage validates rather than silently removes records.",
                           "Timings are one local execution and are not a speedup comparison."],
                       **provenance(['experiments/spark_audit.py'], INPUTS)})
        write_json(args.out, result)
        print(f"Spark SQL audit passed: {result['records']} records; aggregates saved to {args.out}")
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
