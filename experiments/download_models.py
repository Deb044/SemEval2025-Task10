"""Download pinned model resources into ignored project-local caches."""
import argparse
from pathlib import Path

from experiments.common import sha256

GGUF_REPO = "unsloth/Qwen3-1.7B-GGUF"
GGUF_REVISION = "d7f544eead698dbd1f15126ef60b45a1e1933222"
GGUF_FILE = "Qwen3-1.7B-Q4_K_M.gguf"
GGUF_SHA256 = "b139949c5bd74937ad8ed8c8cf3d9ffb1e99c866c823204dc42c0d91fa181897"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification-only", action="store_true")
    args = parser.parse_args()
    from semeval.features import get_encoder
    for name in ("minilm", "xlmr"):
        _ = get_encoder(name).model
        print(name + " ready", flush=True)
    if args.classification_only:
        return
    from huggingface_hub import hf_hub_download
    path = Path("work/models") / GGUF_FILE
    if not path.exists():
        hf_hub_download(GGUF_REPO, GGUF_FILE, revision=GGUF_REVISION, local_dir="work/models")
    if sha256(path) != GGUF_SHA256:
        raise ValueError("Generator checksum mismatch; do not use these weights")
    from semeval.explain import generator_tokenizer
    _ = generator_tokenizer()
    from experiments.public_explanations import download_evaluator
    download_evaluator()
    from experiments.grounding import download as download_nli
    download_nli()
    print("Generator, tokenizer and evaluator ready", flush=True)


if __name__ == "__main__":
    main()
