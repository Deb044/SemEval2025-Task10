"""Run `python -m semeval --help`. Heavy dependencies load only when needed."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from .data import audit, build_taxonomy, check_splits, import_official, load_taxonomy, read_jsonl, write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("import", help="Convert a release gold TSV and article folder to JSONL")
    command.add_argument("--task", type=int, choices=(1, 2, 3), required=True)
    command.add_argument("--articles", required=True)
    command.add_argument("--annotations")
    command.add_argument("--unlabeled", action="store_true", help="Import ST1 mentions, ST2 articles, or ST3 dominant labels without targets")
    command.add_argument("--offset-end", choices=("inclusive", "exclusive"), default="exclusive")
    command.add_argument("--out", required=True)
    command = commands.add_parser("taxonomy", help="Build a parent mapping from exact release label inventories")
    command.add_argument("--coarse", required=True)
    command.add_argument("--fine", required=True)
    command.add_argument("--definitions", help="JSON of guideline paraphrases keyed by every fine label")
    command.add_argument("--out", required=True)
    command = commands.add_parser("audit")
    command.add_argument("paths", nargs="*")
    command.add_argument("--manifest", help="JSON mapping partition names to lists of task JSONL paths")
    command.add_argument("--duplicate-groups", help="JSON mapping article IDs to known duplicate/translation group IDs")
    command.add_argument("--out", required=True)
    command = commands.add_parser("train")
    command.add_argument("--train", required=True)
    command.add_argument("--dev", required=True)
    command.add_argument("--method", choices=("tfidf", "lsa", "transformer", "graph"), required=True)
    command.add_argument("--taxonomy")
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--native", action="store_true", help="Preserve auxiliary dataset labels")
    command.add_argument("--native-decision", choices=("single", "multilabel"), default="single")
    command.add_argument("--ablation", choices=("none", "self-loops", "random-labels", "shuffled-edges"), default="none")
    command.add_argument("--encoder", choices=("minilm", "xlmr"), default="minilm")
    command.add_argument("--out", required=True)
    command = commands.add_parser("predict")
    command.add_argument("--model", required=True)
    command.add_argument("--data", required=True)
    command.add_argument("--out", required=True)
    command = commands.add_parser("explain")
    command.add_argument("--data", required=True)
    command.add_argument("--taxonomy", required=True)
    command.add_argument("--method", choices=("textrank", "semantic", "generate"), required=True)
    command.add_argument("--selection", choices=("semantic", "lead"), default="semantic")
    command.add_argument("--endpoint", default="http://127.0.0.1:8080/v1")
    command.add_argument("--gguf", help="Required for generation: local model file, hashed for provenance")
    command.add_argument("--predicted-labels")
    command.add_argument("--out", required=True)
    command = commands.add_parser("evaluate")
    command.add_argument("--gold", required=True)
    command.add_argument("--predictions", required=True)
    command.add_argument("--bert-model")
    command.add_argument("--bert-layers", type=int)
    command.add_argument("--out", required=True)
    command = commands.add_parser("export")
    command.add_argument("--predictions", required=True)
    command.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "import":
        write_jsonl(args.out, import_official(args.articles, args.annotations, args.task, args.offset_end, args.unlabeled))
    elif args.command == "taxonomy":
        write_json(args.out, build_taxonomy(args.coarse, args.fine, args.definitions))
    elif args.command == "audit":
        if bool(args.paths) == bool(args.manifest):
            raise ValueError("Supply either partition paths or --manifest, not both")
        partitions = json.loads(Path(args.manifest).read_text(encoding="utf-8")) if args.manifest else {path: [path] for path in args.paths}
        if not isinstance(partitions, dict) or not partitions or any(not isinstance(paths, list) or not paths for paths in partitions.values()):
            raise ValueError("Audit manifest must map partition names to nonempty lists of paths")
        datasets = {name: [row for path in paths for row in read_jsonl(path)] for name, paths in partitions.items()}
        groups = json.loads(Path(args.duplicate_groups).read_text(encoding="utf-8")) if args.duplicate_groups else None
        check_splits(*datasets.values(), duplicate_groups=groups)
        write_json(args.out, {name: audit(rows) for name, rows in datasets.items()})
    elif args.command == "train":
        import joblib
        from .classify import train
        model = train(read_jsonl(args.train), read_jsonl(args.dev), args.method, args.taxonomy, args.seed, args.ablation,
                      native=args.native, native_decision=args.native_decision, encoder=args.encoder)
        model["training_articles"] = sorted({r["article_id"] for r in read_jsonl(args.train)})
        model["data_sha256"] = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                for path in (args.train, args.dev, args.taxonomy) if path}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.out)
        write_json(args.out + ".json", {key: model[key] for key in ("task", "method", "seed", "ablation", "threshold", "regularization", "development_metric", "development_score", "fit_seconds", "data_sha256")})
    elif args.command == "predict":
        import joblib
        from .classify import predict
        # Only load your own artifacts: joblib/pickle files can execute code.
        write_jsonl(args.out, predict(joblib.load(args.model), read_jsonl(args.data)))
    elif args.command == "explain":
        from .explain import explain_row
        rows, taxonomy = read_jsonl(args.data), load_taxonomy(args.taxonomy)
        labels = {r["article_id"]: r for r in read_jsonl(args.predicted_labels)} if args.predicted_labels else None
        result = []
        model_hash = None
        if args.method == "generate":
            if not args.gguf:
                raise ValueError("Generation requires --gguf to record the actual local model file hash")
            with open(args.gguf, "rb") as stream:
                model_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        # Resume only matching-input runs; never trust a partially unrelated file.
        signature = hashlib.sha256(json.dumps({"data": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
            "taxonomy": taxonomy, "method": args.method, "selection": args.selection,
            "labels": labels, "endpoint": args.endpoint, "gguf_sha256": model_hash,
            "implementation": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                               for name in ("explain.py", "features.py")}}, sort_keys=True).encode()).hexdigest()
        output = Path(args.out)
        if output.exists():
            result = read_jsonl(output)
            if any(r.get("run_signature") != signature for r in result):
                raise ValueError("Existing output belongs to a different explanation run")
        done = {r["article_id"] for r in result}
        for row in rows:
            if row["article_id"] in done:
                continue
            started = time.perf_counter()
            value = explain_row(row, args.method, taxonomy, args.endpoint, args.selection,
                                labels[row["article_id"]] if labels is not None else None)
            value.update(run_signature=signature, seconds=time.perf_counter() - started, gguf_sha256=model_hash)
            result.append(value)
            write_jsonl(output, result)
    elif args.command == "evaluate":
        from .evaluate import evaluate
        report = evaluate(read_jsonl(args.gold), read_jsonl(args.predictions), args.bert_model, args.bert_layers)
        report["input_sha256"] = {key: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                  for key, path in (("gold", args.gold), ("predictions", args.predictions))}
        try:
            report["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            report["commit"] = None
        write_json(args.out, report)
    elif args.command == "export":
        from .evaluate import export_tsv
        export_tsv(read_jsonl(args.predictions), args.out)


if __name__ == "__main__":
    main()
