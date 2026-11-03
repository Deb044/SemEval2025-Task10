"""CPU-first experiments for SemEval-2025 Task 10."""
import os
from pathlib import Path

# This project uses PyTorch only. Avoid importing an unrelated system Keras stack.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("HF_HOME", str(Path.cwd() / ".cache" / "huggingface"))

