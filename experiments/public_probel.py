"""Pinned public ProBel news-only pilot; native multilabels, article-group splits.

Run from the repository root. Raw texts and model-authored references stay ignored.
The custom split is not the publisher benchmark and never a SemEval evaluation.
"""
import argparse
import hashlib
import json
import random
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit, train_test_split

from semeval.data import audit, check_splits, fingerprint, write_json, write_jsonl

REVISION = "076ac4ebcf5c1d87e6679d3e21205159f00ae7db"
ROOT = Path("data/public/probel")
CHECKSUMS = {
    "README.md": "0d9366dd0721a16fbf8a9401317e63abcf9004cd5e06ccba6cc2979e8ffaf5ad",
    "LICENSE": "e6053d704272991fef4a69ac537793dfe81f8c8f15a25acc72e8e94b07567a24",
    "label_taxonomy.json": "0c4d5a8c6843bf7b8cb6ed0cfe54e7ce1847e3eb624c4ac4a44ac36ba4f6d9aa",
    "arabic/train.jsonl": "e59c8fdc94244a7530ce4ce26a2c0ef5e0d62d96f7675f0ca3601040b9f68212",
    "arabic/dev.jsonl": "fafefd7650fd1fab0b25d005589b977738ad98e191ad5c1559773b08d8986cd2",
    "arabic/test.jsonl": "d7f6ff8cd1f7c4b0ec3b62e9d7040df0a277409fbdb5a714dfc143dbc2360a7d",
    "english/train.jsonl": "1de727e23e2c47188dbe918a30841e0533c2c11fa123fc27f9b08ea289232224",
    "english/dev.jsonl": "2d0464f5db8c49607facf9002873886ccae30f31d6e01cb1a087fceddeca9e66",
    "english/test.jsonl": "cccbb741ddbda206784183897fce32c19df42a0b928414673f24895bb7573797",
}
DATASET = "ProBel news grouped pilot"


def download():
    for name, expected in CHECKSUMS.items():
        path = ROOT / "raw" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            urllib.request.urlretrieve(f"https://huggingface.co/datasets/QCRI/ProBel/resolve/{REVISION}/{name}", path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source checksum mismatch: {name}")


def convert(raw, language, mapping):
    labels = sorted(set(raw["techniques"]))
    if set(labels) - set(mapping) or bool(labels) != raw["binary"]:
        raise ValueError("Unknown or inconsistent native technique labels")
    if sorted(set(mapping[label] for label in labels)) != sorted(raw["coarse_categories"]):
        raise ValueError("Native coarse/fine labels disagree")
    if not raw["metadata"].get("page_id"):
        raise ValueError("News source article ID missing")
    for span in raw["spans"]:
        if raw["text"][span["start"]:span["end"]] != span["text"]:
            raise ValueError("Source span offset mismatch")
    return {"task": 2, "article_id": language.upper() + "_probel_" + raw["id"] + ".txt",
        "language": language, "text": raw["text"], "labels": labels,
        "narratives": sorted(set(mapping[label] for label in labels)),
        "source_group": language + ":" + raw["metadata"]["page_id"],
        "dataset": DATASET, "label_schema": "native", "decision": "native-multilabel",
        "reference_explanation": raw["explanation"],
        "reference_provenance": "OpenAI o1 generated; publisher human-evaluated, not human-authored",
        "source_type": "news", "source_spans": raw["spans"]}


def group_split(rows):
    """Join source articles connected by duplicate content before any sampling."""
    parents = {}
    def find(key):
        parents.setdefault(key, key)
        if parents[key] != key:
            parents[key] = find(parents[key])
        return parents[key]
    for row in rows:
        a, b = find(row["source_group"]), find("text:" + fingerprint(row["text"]))
        parents[b] = a
    groups = [find(row["source_group"]) for row in rows]
    indices = list(range(len(rows)))
    train, held = next(GroupShuffleSplit(n_splits=1, test_size=.3, random_state=42).split(indices, groups=groups))
    dev_index, test_index = next(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=42).split(held, groups=[groups[i] for i in held]))
    return {"train": [rows[i] for i in train], "dev": [rows[held[i]] for i in dev_index],
            "test": [rows[held[i]] for i in test_index]}


def capped(rows, per_language):
    selected = []
    for language in ("en", "ar"):
        pool = sorted((row for row in rows if row["language"] == language), key=lambda row: row["article_id"])
        if len(pool) > per_language:
            pool, _ = train_test_split(pool, train_size=per_language, random_state=42,
                                       stratify=[bool(row["labels"]) for row in pool])
        selected.extend(pool)
    return sorted(selected, key=lambda row: row["article_id"])


def prepare(train_cap=800, evaluation_cap=200):
    mapping = json.loads((ROOT / "raw/label_taxonomy.json").read_text())["fine_to_coarse"]
    taxonomy = {label: {"parent": parent, "definition": label.replace("_", " ").replace("-", " / "),
                        "definition_source": "publisher_label_name_only"} for label, parent in mapping.items()}
    rows, rejected, sources = [], Counter(), {}
    for language, code in (("english", "en"), ("arabic", "ar")):
        groups_by_split = {}
        for split in ("train", "dev", "test"):
            records = [json.loads(line) for line in (ROOT / "raw" / language / (split + ".jsonl")).read_text(encoding="utf-8").splitlines() if line.strip()]
            news = [record for record in records if record["source_type"] == "news"]
            groups_by_split[split] = {record["metadata"].get("page_id") for record in news}
            for raw in news:
                try:
                    rows.append(convert(raw, code, mapping))
                except ValueError as error:
                    rejected[str(error)] += 1
        sources[language] = {a + "_" + b: len(groups_by_split[a] & groups_by_split[b])
                             for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))}
    duplicate_groups = defaultdict(list)
    for row in rows:
        duplicate_groups[fingerprint(row["text"])].append(row)
    cleaned = []
    for group in duplicate_groups.values():
        if len({tuple(row["labels"]) for row in group}) != 1:
            rejected["conflicting_duplicate_text"] += len(group)
        else:
            cleaned.extend(group)  # Keep articles connected until group assignment.
    splits = group_split(sorted(cleaned, key=lambda row: row["article_id"]))
    # Deduplicate within each already-disjoint partition, then cap without inspecting scores.
    for split, partition in splits.items():
        unique = {}
        for row in partition:
            unique.setdefault(fingerprint(row["text"]), row)
        splits[split] = capped(list(unique.values()), train_cap if split == "train" else evaluation_cap)
    groups = {row["article_id"]: row["source_group"] for partition in splits.values() for row in partition}
    check_splits(*splits.values(), duplicate_groups=groups)
    write_json(ROOT / "taxonomy.json", taxonomy)
    write_json(ROOT / "duplicate-groups.json", groups)
    for split, partition in splits.items():
        write_jsonl(ROOT / (split + ".jsonl"), partition)
    # Fixed bilingual explanation pilot, balanced by language and native binary label.
    explanations = []
    rng = random.Random(42)
    for language in ("en", "ar"):
        for propagandistic in (False, True):
            pool = [row for row in splits["test"] if row["language"] == language and bool(row["labels"]) == propagandistic and row["reference_explanation"].strip()]
            rng.shuffle(pool)
            explanations.extend(dict(row, task=3, explanation=row["reference_explanation"]) for row in pool[:10])
    write_jsonl(ROOT / "explanations.jsonl", explanations)
    write_json(ROOT / "experiment.json", {"dataset": DATASET, "label_schema": "native", "native_decision": "multilabel",
        **{split: str(ROOT / (split + ".jsonl")) for split in splits}, "taxonomy": str(ROOT / "taxonomy.json"),
        "output": "artifacts/public-probel"})
    audit_result = {split: audit(partition) for split, partition in splits.items()}
    audit_result.update(source_revision=REVISION, source_sha256=CHECKSUMS, rejected=dict(rejected),
        original_source_article_overlap=sources, explanation_examples=len(explanations),
        protocol="News only. Original splits pooled and regrouped by article plus duplicate text; 70/15/15 seed42, then bilingual caps preserving binary proportions. Not publisher benchmark.",
        references="Model-authored OpenAI o1 explanations, publisher human-evaluated; not human gold.")
    write_json(ROOT / "audit.json", audit_result)
    print(json.dumps({"splits": {key: len(value) for key, value in splits.items()}, "explanations": len(explanations), "rejected": dict(rejected)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-per-language", type=int, default=800)
    parser.add_argument("--evaluation-per-language", type=int, default=200)
    args = parser.parse_args()
    if min(args.train_per_language, args.evaluation_per_language) < 2:
        raise ValueError("Pilot caps must be at least two")
    download()
    prepare(args.train_per_language, args.evaluation_per_language)


if __name__ == "__main__":
    main()
