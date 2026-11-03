"""Small provenance helpers shared by extension experiments."""
import hashlib
import importlib.metadata
import platform
import subprocess
from pathlib import Path


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def provenance(sources, inputs):
    return {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": {str(path): sha256(path) for path in sources},
            "input_sha256": {name: sha256(path) for name, path in inputs.items()},
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                "packages": {name: importlib.metadata.version(name) for name in
                    ("numpy", "scikit-learn", "torch", "transformers", "sentence-transformers")}}}
