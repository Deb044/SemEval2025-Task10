"""Reproduce the current fixed studies; historical results retain original commits.

An existing completed stage is skipped only after its input and source hashes
match. Classification has its own strict per-model checkpoint validation.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from experiments.common import sha256


def run(*arguments):
    subprocess.run([sys.executable, "-m", *arguments], check=True)


def verified(path, inputs):
    if not path.exists():
        return False
    summary = json.loads(path.read_text(encoding="utf-8"))
    for source, expected in summary.get("source_sha256", {}).items():
        if not Path(source).exists() or sha256(source) != expected:
            raise ValueError(f"Source changed since {path}; preserve it and choose a fresh output directory")
    if not summary.get("source_sha256"):
        raise ValueError("Completed stage lacks source provenance")
    for name, source in inputs.items():
        if summary.get("input_sha256", {}).get(name) != sha256(source):
            raise ValueError(f"Input changed since {path}")
    return True


def ready():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=1) as response:
            return response.status == 200
    except OSError:
        return False


@contextmanager
def generator(args):
    if ready():
        if not args.use_running_server:
            raise ValueError("Port 8080 already has a generator. Use --use-running-server only for the documented model/settings")
        yield
        return
    binary = args.server or shutil.which("llama-server")
    if not binary and Path("work/llama-b11960/llama-server.exe").exists():
        binary = "work/llama-b11960/llama-server.exe"
    if not binary:
        raise ValueError("Install llama.cpp b11960 and pass --server path/to/llama-server")
    log_path = Path("artifacts/generator.log")
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.Popen([binary, "-m", args.gguf, "--host", "127.0.0.1", "--port", "8080", "-c", "4096", "-t", "4", "-ngl", "0"],
            stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            deadline = time.monotonic() + 120
            while not ready():
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("Generator failed to become ready; inspect artifacts/generator.log")
                time.sleep(.5)
            yield
        finally:
            process.terminate()
            process.wait(timeout=20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "prepare", "classification", "source-transfer", "explanations", "grounding", "scalability", "export"), default="all")
    parser.add_argument("--gguf", default="work/models/Qwen3-1.7B-Q4_K_M.gguf")
    parser.add_argument("--server")
    parser.add_argument("--use-running-server", action="store_true")
    args = parser.parse_args()
    selected = lambda name: args.stage in ("all", name)
    if selected("prepare"):
        run("experiments.public_data")
        run("experiments.public_probel")
    if selected("classification"):
        for name in ("cards", "probel"):
            run("experiments.run", "experiments/configs/" + name + ".json", "--resume")
    if selected("source-transfer"):
        inputs = {name: Path("data/public/cards") / (name + ".jsonl") for name in ("train", "dev", "test", "ood")}
        if not verified(Path("artifacts/extensions-emm/summary.json"), inputs):
            run("experiments.ood")
    if selected("explanations"):
        inputs = {"evaluation": "data/public/cards/test.jsonl", "taxonomy": "data/public/cards/taxonomy.json",
            "classifier": "artifacts/public-cards-v2/graph-none-42.joblib", "generator": args.gguf}
        # Prefer a reproduced current model in fresh checkouts; historical local
        # runs retain their explicitly recorded classifier in their summary.
        if not Path(inputs["classifier"]).exists():
            inputs["classifier"] = "artifacts/extensions-cards/graph-none-42.joblib"
        explanation_summary = Path("artifacts/extensions-long-explanations/summary.json")
        complete = verified(explanation_summary, inputs)
        if complete and json.loads(explanation_summary.read_text())["statistics"]["source_passages"] != 24:
            raise ValueError("Existing explanation run uses a different subset limit; preserve it and choose fresh output")
        if not complete:
            with generator(args):
                run("experiments.long_explanations", "--gguf", args.gguf, "--model", str(inputs["classifier"]))
    if selected("scalability"):
        from experiments.public_probel import CHECKSUMS, ROOT
        inputs = {name: ROOT / "raw" / name for name in CHECKSUMS if name.endswith(".jsonl")}
        if not verified(Path("artifacts/extensions-scalability/summary.json"), inputs):
            run("experiments.scalability")
        from experiments.throughput import MODELS
        if not verified(Path("artifacts/extensions-throughput/summary.json"), {"evaluation": "data/public/probel/test.jsonl", **MODELS}):
            run("experiments.throughput")
    if selected("grounding"):
        # The local diagnostic is repeatable. A separate capable LLM can repeat
        # the ancillary blinded audit using docs/llm-review-prompt.md.
        run("experiments.grounding")
    if selected("export") or args.stage == "all":
        if all(Path(path).exists() for path in ("artifacts/extensions-cards/summary.json", "artifacts/extensions-probel/summary.json", "artifacts/extensions-emm/summary.json")):
            run("experiments.errors")
        run("experiments.catalog")


if __name__ == "__main__":
    main()
