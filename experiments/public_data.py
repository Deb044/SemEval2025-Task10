"""Download author releases and prepare explicitly auxiliary native-label tasks.

No raw text is committed or relabelled as SemEval annotations. All source bytes
are pinned by revision and SHA256. Run from the repository root.
"""
import argparse
import csv
import hashlib
import json
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit, train_test_split

from semeval.data import audit, check_splits, fingerprint, write_json, write_jsonl

ROOT = Path("data/public")
SOURCES = {
    "climate-corpora.zip": ("https://raw.githubusercontent.com/jpiskorski/CC-denial-resources/aa13290bfbf2655b95b1e63d330b36bf8af68215/CLIMATE_CHANGE_DENIAL_CORPORA.zip", "694d6d1cd843aec4995056fd04a25d6cd121a5db9755a53d1b7cbca5829d6f1b"),
    "yaso_hidden.json": ("https://raw.githubusercontent.com/IBM/yaso-tsa/7c83da107496392d1c3ebc887dbf072588f0441a/yaso_tsa/data/yaso_hidden.json", "90c186e7975d596014c253e81e89afbba9b97232fdd23fcc81d0e76b1e6c6c12"),
    "esnli_test.csv": ("https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/7b585a3f077fdea899780eb0473940522ae44a2e/dataset/esnli_test.csv", "9269aa8075dbfed0fa8a5012eb2ac9c18a92be139671f05df897f69092592927"),
}
CARDS_PARENTS = {
    "0": "CARDS 0: No contrarian claim", "1": "CARDS 1: Warming is not occurring",
    "2": "CARDS 2: Humans are not causing warming", "3": "CARDS 3: Impacts are not harmful",
    "4": "CARDS 4: Solutions will not work", "5": "CARDS 5: Science or movement is unreliable",
}
# Paraphrases of the native categories in Coan et al. (2021), Table 3.
CARDS_DEFINITIONS = {
    "0_0": "No annotated contrarian claim.", "1_1": "Claims that ice is not melting.",
    "1_2": "Claims that global cooling is approaching.", "1_3": "Uses cold weather to challenge warming.",
    "1_4": "Claims recent temperatures have not risen.", "1_6": "Minimizes sea-level rise.",
    "1_7": "Denies links between climate change and extreme weather.",
    "2_1": "Attributes warming to natural variation.", "2_3": "Rejects carbon dioxide greenhouse warming.",
    "3_1": "Claims low climate sensitivity or cooling feedbacks.", "3_2": "Claims ecosystems benefit or remain unharmed.",
    "3_3": "Portrays carbon dioxide as beneficial.", "4_1": "Claims climate policies cause harm.",
    "4_2": "Claims climate policies are ineffective.", "4_4": "Claims clean energy technologies fail.",
    "4_5": "Argues that conventional energy remains necessary.",
    "5_1": "Challenges climate research methods or reliability.", "5_2": "Discredits climate advocates or their motives.",
}


def download():
    raw = ROOT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in SOURCES.items():
        path = raw / name
        if not path.exists():
            urllib.request.urlretrieve(url, path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source checksum mismatch: {name}; do not use modified or partial data")


def unique_records(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[fingerprint(row["text"])].append(row)
    # Remove ambiguous duplicate annotations before any model sees the data.
    return [group[0] for group in groups.values() if len({tuple(r["labels"]) for r in group}) == 1]


def cap(rows, maximum):
    if len(rows) <= maximum:
        return rows
    selected, _ = train_test_split(rows, train_size=maximum, stratify=[r["labels"][0] for r in rows], random_state=42)
    return sorted(selected, key=lambda r: r["article_id"])


def save_partition(folder, name, rows):
    write_jsonl(folder / (name + ".jsonl"), rows)


def prepare_cards(max_train=6000, max_dev=1000):
    folder = ROOT / "cards"
    names = {"training": "4C_training_CLEANED.txt", "test": "4C_test_CLEANED.txt", "ood": "EMM_CLIMATE_CHANGE_CORPUS_version_1.0.txt"}
    partitions = {}
    with zipfile.ZipFile(ROOT / "raw/climate-corpora.zip") as archive:
        for split, name in names.items():
            rows = []
            for index, line in enumerate(archive.read(name).decode("utf-8-sig").splitlines()):
                if not line.strip():
                    continue
                text, label = line.rsplit("\t", 1)
                if label not in CARDS_DEFINITIONS:
                    raise ValueError(f"Unknown native CARDS class: {label}")
                rows.append({"task": 2, "dataset": "EMM climate snippets" if split == "ood" else "Cleaned CARDS/4C", "label_schema": "native",
                    "article_id": f"EN_{'EMM' if split == 'ood' else '4C'}_{split}_{index}.txt", "language": "en", "text": text,
                    "labels": [label], "narratives": [CARDS_PARENTS[label.split("_")[0]]]})
            partitions[split] = unique_records(rows)
    test_hashes = {fingerprint(row["text"]) for row in partitions["test"]}
    ood = [r for r in partitions["ood"] if fingerprint(r["text"]) not in test_hashes]
    heldout_hashes = test_hashes | {fingerprint(r["text"]) for r in ood}
    training = [r for r in partitions["training"] if fingerprint(r["text"]) not in heldout_hashes]
    train, dev = train_test_split(training, test_size=.1, stratify=[r["labels"][0] for r in training], random_state=42)
    train, dev = cap(train, max_train), cap(dev, max_dev)
    check_splits(train, dev, partitions["test"], ood)
    for name, rows in {"train": train, "dev": dev, "test": partitions["test"], "ood": ood}.items():
        save_partition(folder, name, rows)
    taxonomy = {label: {"parent": CARDS_PARENTS[label.split("_")[0]], "definition": description,
                        "definition_source": "Coan2021_Table3_paraphrase"} for label, description in CARDS_DEFINITIONS.items()}
    write_json(folder / "taxonomy.json", taxonomy)
    write_json(folder / "audit.json", {name: audit(rows) for name, rows in {"train": train, "dev": dev, "test": partitions["test"], "ood": ood}.items()})
    write_json(folder / "experiment.json", {"dataset": "Cleaned CARDS/4C", "label_schema": "native",
        **{name: str(folder / (name + ".jsonl")) for name in ("train", "dev", "test")},
        "taxonomy": str(folder / "taxonomy.json"), "output": "artifacts/public-cards"})


def prepare_yaso():
    rows = json.loads((ROOT / "raw/yaso_hidden.json").read_text(encoding="utf-8"))
    examples, group_by_id = [], {}
    for index, row in enumerate(rows):
        text = row["text"]
        if not isinstance(text, str):
            continue  # Do not restore or guess protected/missing source text.
        article_id = f"EN_YASO_{index}.txt"
        for target in row["targets"]:
            start, end = target["location"]["begin"], target["location"]["end"]
            if target["confidence"] < .7 or text[start:end] != target["text"]:
                continue
            examples.append({"task": 1, "dataset": "YASO clear-text subset", "label_schema": "native",
                "article_id": article_id, "language": "en", "text": text, "mention": target["text"],
                "start": start, "end": end, "end_exclusive": end, "labels": [target["sentiment"]]})
            group_by_id[article_id] = fingerprint(row.get("review_text") or text)
    groups = [group_by_id[r["article_id"]] for r in examples]
    train_ids, holdout_ids = next(GroupShuffleSplit(n_splits=1, test_size=.4, random_state=42).split(examples, groups=groups))
    train = [examples[i] for i in train_ids]
    heldout = [examples[i] for i in holdout_ids]
    dev_ids, test_ids = next(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=42).split(heldout, groups=[group_by_id[r["article_id"]] for r in heldout]))
    dev, test = [heldout[i] for i in dev_ids], [heldout[i] for i in test_ids]
    check_splits(train, dev, test, duplicate_groups=group_by_id)
    folder = ROOT / "yaso"
    for name, records in {"train": train, "dev": dev, "test": test}.items():
        save_partition(folder, name, records)
    write_json(folder / "taxonomy.json", {label: {"parent": "Target sentiment", "definition": label + " sentiment toward the supplied target"}
                                         for label in ("positive", "negative", "mixed", "none")})
    write_json(folder / "audit.json", {name: audit(records) for name, records in {"train": train, "dev": dev, "test": test}.items()})
    write_json(folder / "experiment.json", {"dataset": "YASO clear-text subset", "label_schema": "native",
        **{name: str(folder / (name + ".jsonl")) for name in ("train", "dev", "test")},
        "taxonomy": str(folder / "taxonomy.json"), "output": "artifacts/public-yaso"})


def prepare_esnli(limit=60):
    with open(ROOT / "raw/esnli_test.csv", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows = cap([{"labels": [row["gold_label"]], "article_id": row["pairID"], "source": row} for row in rows], limit)
    result = []
    for wrapper in rows:
        row = wrapper["source"]
        result.append({"dataset": "e-SNLI", "source_split": "test", "pair_id": row["pairID"],
                       "premise": row["Sentence1"], "hypothesis": row["Sentence2"], "label": row["gold_label"],
                       "references": [row[f"Explanation_{i}"] for i in (1, 2, 3) if row[f"Explanation_{i}"].strip()]})
    write_jsonl(ROOT / "esnli/evaluation.jsonl", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-train", type=int, default=6000)
    parser.add_argument("--max-dev", type=int, default=1000)
    parser.add_argument("--explanations", type=int, default=60)
    args = parser.parse_args()
    download()
    prepare_cards(args.max_train, args.max_dev)
    prepare_yaso()
    prepare_esnli(args.explanations)
    write_json(ROOT / "sources.json", {name: {"url": url, "sha256": checksum} for name, (url, checksum) in SOURCES.items()})
    print("Prepared native-label public datasets under ignored data/public/. No SemEval annotations were synthesized.")


if __name__ == "__main__":
    main()
