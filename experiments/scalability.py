"""Measure real news ingestion/query scaling and uncached/cached embeddings.

Each measurement uses an isolated subprocess. First scans are not claimed to be
cold OS-cache scans. No replicated or synthetic records are counted as new data.
"""
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

from experiments.common import provenance
from semeval.data import fingerprint, write_json
from semeval.resources import peak_rss_mb


def worker(operation, source, destination):
    if operation == "embeddings":
        import numpy as np
        import torch
        from semeval.features import FrozenEncoder
        torch.set_num_threads(4)
        rows = [json.loads(line) for line in Path(source).read_text(encoding="utf-8").splitlines()]
        encoder = FrozenEncoder(cache=destination)
        started = time.perf_counter()
        _ = encoder.model
        load_seconds = time.perf_counter() - started
        started = time.perf_counter()
        first = encoder.encode([row["text"] for row in rows])
        uncached = time.perf_counter() - started
        started = time.perf_counter()
        second = encoder.encode([row["text"] for row in rows])
        cached = time.perf_counter() - started
        if not np.array_equal(first, second):
            raise ValueError("Embedding cache changed the measured vectors")
        return {"operation": operation, "count": len(rows), "model_load_seconds": load_seconds,
            "uncached_seconds": uncached, "cached_seconds": cached,
            "uncached_records_per_second": len(rows) / uncached, "cached_records_per_second": len(rows) / cached,
            "vectors_identical": True, "peak_rss_mb": peak_rss_mb()}
    import duckdb
    connection = duckdb.connect()
    connection.execute("SET threads=4")
    connection.execute("SET memory_limit='256MB'")
    if operation == "convert":
        started = time.perf_counter()
        connection.read_json(source, format="newline_delimited").write_parquet(destination, compression="zstd")
        return {"operation": operation, "seconds": time.perf_counter() - started,
            "json_bytes": Path(source).stat().st_size, "parquet_bytes": Path(destination).stat().st_size,
            "peak_rss_mb": peak_rss_mb()}
    function = "read_parquet" if operation == "parquet" else "read_json_auto"
    query = f"SELECT language, count(*) AS n, sum(CAST(propagandistic AS INTEGER)) AS positive FROM {function}(?) GROUP BY language ORDER BY language"
    times = []
    results = None
    for _ in range(2):
        started = time.perf_counter()
        results = connection.execute(query, [source]).fetchall()
        times.append(time.perf_counter() - started)
    return {"operation": operation, "first_scan_seconds": times[0], "repeat_scan_seconds": times[1],
        "count": sum(row[1] for row in results), "counts_by_language": results, "peak_rss_mb": peak_rss_mb()}


def isolated(operation, source, destination):
    command = [sys.executable, "-m", "experiments.scalability", "--worker", operation,
               "--source", str(source), "--destination", str(destination)]
    return json.loads(subprocess.check_output(command, text=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/extensions-scalability")
    parser.add_argument("--worker", choices=("convert", "json", "parquet", "embeddings"))
    parser.add_argument("--source")
    parser.add_argument("--destination")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker, args.source, args.destination)))
        return
    from experiments.public_probel import CHECKSUMS, ROOT, download
    download()
    output = Path(args.out)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh benchmark output directory to preserve timings and cold embedding caches")
    output.mkdir(parents=True)
    inputs = {name: ROOT / "raw" / name for name in CHECKSUMS if name.endswith(".jsonl")}
    meta = provenance(["experiments/scalability.py", "semeval/features.py", "semeval/resources.py"], inputs)
    unique = {}
    for source in inputs.values():
        with open(source, encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["source_type"] != "news":
                    continue
                unique.setdefault(fingerprint(row["text"]), {"language": row["language"], "text": row["text"], "propagandistic": row["binary"]})
    pool = [unique[key] for key in sorted(unique)]
    random.Random(42).shuffle(pool)
    records = []
    for count in (1000, 5000, 10000, 25000):
        if count > len(pool):
            continue
        source, parquet = output / f"news-{count}.jsonl", output / f"news-{count}.parquet"
        source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pool[:count]), encoding="utf-8")
        conversion = isolated("convert", source, parquet)
        baseline = isolated("json", source, output)
        columnar = isolated("parquet", parquet, output)
        if baseline["counts_by_language"] != columnar["counts_by_language"] or baseline["count"] != count:
            raise ValueError("JSON and Parquet queries disagree")
        records.append({"count": count, "conversion": conversion, "json": baseline, "parquet": columnar})
        print(f"Measured {count} unique news records", flush=True)
    embedding_records = []
    for count in (32, 128):
        source = output / f"embedding-input-{count}.jsonl"
        source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pool[:count]), encoding="utf-8")
        embedding_records.append(isolated("embeddings", source, output / f"embedding-cache-{count}"))
        print(f"Measured {count} uncached and cached embeddings", flush=True)
    import duckdb
    write_json(output / "summary.json", {**meta, "experiment_id": "data-and-embedding-scalability", "dataset": "ProBel news unique-text pool",
        "unique_source_records": len(pool), "measurements": records, "embeddings": embedding_records,
        "duckdb_version": duckdb.__version__, "threads": 4, "duckdb_memory_limit": "256MB",
        "limitations": "Isolated worker peak RSS includes imports and runtime overhead. First scans may benefit from OS caches. DuckDB memory limit is not a total-process RAM bound. No distributed-processing or larger-than-RAM demonstration is claimed."})


if __name__ == "__main__":
    main()
