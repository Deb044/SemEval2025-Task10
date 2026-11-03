"""Explicit official-format import, validation, and leakage checks.

Internal records are JSONL. Original article text is never normalized: offsets
refer to that exact Unicode string. The user must choose the offset convention
documented by their corpus release, rather than silently repairing mismatches.
"""
import hashlib
import json
import time
from pathlib import Path

LANGUAGES = {"EN": "en", "HI": "hi", "BG": "bg", "PT": "pt", "RU": "ru"}
ROLE_GROUPS = {
    "Protagonist": ["Guardian", "Martyr", "Peacemaker", "Rebel", "Underdog", "Virtuous"],
    "Antagonist": ["Instigator", "Conspirator", "Tyrant", "Foreign Adversary", "Traitor",
                   "Spy", "Saboteur", "Corrupt", "Incompetent", "Terrorist", "Deceiver", "Bigot"],
    "Innocent": ["Forgotten", "Exploited", "Victim", "Scapegoat"],
}
ROLE_PARENT = {role: parent for parent, roles in ROLE_GROUPS.items() for role in roles}


def replace_file(temporary, path):
    """Allow short-lived Windows/OneDrive locks without losing the old file."""
    for attempt in range(7):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 6:
                raise
            time.sleep(.05 * 2 ** attempt)


def read_jsonl(path):
    with open(path, encoding="utf-8-sig") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    return rows


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    replace_file(temporary, path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    replace_file(temporary, path)


def fingerprint(text):
    """Whitespace-insensitive exact duplicate hash, NOT a near-duplicate detector."""
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def identity(row):
    if row["task"] == 1:
        return (row["article_id"], row["start"], row["end"])
    return (row["article_id"],)


def validate(rows):
    if not rows:
        raise ValueError("Dataset contains no examples")
    seen = set()
    articles = {}
    for row in rows:
        key = (row["task"], *identity(row))
        if key in seen:
            raise ValueError(f"Duplicate example: {key}")
        seen.add(key)
        allowed_languages = set(LANGUAGES.values()) | ({"ar"} if row.get("label_schema") == "native" else set())
        if row["language"] not in allowed_languages or not row["text"].strip():
            raise ValueError(f"Invalid language or empty article: {key}")
        article = (row["language"], row["text"])
        if row["article_id"] in articles and articles[row["article_id"]] != article:
            raise ValueError(f"Conflicting text or language for article: {row['article_id']}")
        articles[row["article_id"]] = article
        if row["task"] == 1:
            if not 0 <= row["start"] < row["end_exclusive"] <= len(row["text"]):
                raise ValueError(f"Entity offsets out of range: {key}")
            if row["text"][row["start"]:row["end_exclusive"]] != row["mention"]:
                raise ValueError(f"Entity offset mismatch: {key}")
            if row.get("label_schema") != "native":
                for role in row.get("labels", []):
                    if role not in ROLE_PARENT or ROLE_PARENT[role] != row.get("main_role"):
                        raise ValueError(f"Unknown or incompatible entity role: {role}")


def check_splits(*splits, duplicate_groups=None):
    """Reject ID, content and researcher-identified translation-group overlap.

    Each argument is a partition, which may contain records from multiple tasks.
    Duplicate groups map article IDs to shared, manually verified group IDs.
    """
    duplicate_groups = duplicate_groups or {}
    owners = {}
    for index, rows in enumerate(splits):
        validate(rows)
        for row in rows:
            keys = ["id:" + row["article_id"], "text:" + fingerprint(row["text"])]
            if row["article_id"] in duplicate_groups:
                keys.append("group:" + str(duplicate_groups[row["article_id"]]))
            for key in keys:
                if key in owners and owners[key] != index:
                    raise ValueError(f"Article leakage between splits: {row['article_id']}")
                owners[key] = index


def import_official(articles, annotations, task, offset_end="exclusive", unlabeled=False):
    files = {}
    for path in Path(articles).rglob("*.txt"):
        if path.name in files:
            raise ValueError(f"Ambiguous article basename: {path.name}")
        files[path.name] = path
    rows = []
    if annotations is None:
        if task != 2 or not unlabeled:
            raise ValueError("Only unlabeled ST2 can be imported without an annotation file")
        for article_id, path in sorted(files.items()):
            with open(path, encoding="utf-8", newline="") as article:
                rows.append({"task": 2, "article_id": article_id, "language": LANGUAGES.get(article_id[:2]), "text": article.read()})
        validate(rows)
        return rows
    with open(annotations, encoding="utf-8-sig", newline="") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            fields = line.rstrip("\r\n").split("\t")
            article_id = fields[0]
            if article_id not in files:
                raise ValueError(f"Line {number}: missing article {article_id}")
            # newline='' avoids Windows CRLF conversion, preserving corpus offsets.
            with open(files[article_id], encoding="utf-8", newline="") as article:
                text = article.read()
            row = {"task": task, "article_id": article_id, "language": LANGUAGES.get(article_id[:2]), "text": text}
            if task == 1:
                if len(fields) < (4 if unlabeled else 6):
                    raise ValueError(f"Line {number}: missing entity columns")
                row.update(mention=fields[1], start=int(fields[2]), end=int(fields[3]),
                           end_exclusive=int(fields[3]) + (offset_end == "inclusive"))
                if not unlabeled:
                    row.update(main_role=fields[4], labels=fields[5:])
            elif task == 2:
                if len(fields) != 3:
                    raise ValueError(f"Line {number}: expected three columns")
                row.update(narratives=fields[1].split(";"), labels=fields[2].split(";"))
            elif task == 3:
                if len(fields) != (3 if unlabeled else 4):
                    raise ValueError(f"Line {number}: expected dominant labels and {'no reference' if unlabeled else 'reference explanation'}")
                row.update(dominant_narrative=fields[1], dominant_subnarrative=fields[2])
                if not unlabeled:
                    row["explanation"] = fields[3]
            rows.append(row)
    validate(rows)
    return rows


def load_taxonomy(path, require_other=True):
    """Use exact release label strings; do not guess parents from colon splitting."""
    taxonomy = json.loads(Path(path).read_text(encoding="utf-8"))
    if not taxonomy or not all(isinstance(v.get("parent"), str) and isinstance(v.get("definition"), str)
                               for v in taxonomy.values()):
        raise ValueError("Taxonomy must map fine labels to parent and definition")
    if require_other and ("Other" not in taxonomy or taxonomy["Other"]["parent"] != "Other"):
        raise ValueError("Include the global Other label with parent Other")
    return taxonomy


def build_taxonomy(coarse_path, fine_path, definitions_path=None):
    """Build a checked mapping from the release's explicit class inventories.

    Match full coarse-label prefixes rather than blindly splitting on colons.
    Definitions are researcher-provided paraphrases of the public guidelines.
    Without them the output is explicitly a label-name-only representation.
    """
    def labels(path):
        values = [line.strip() for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if not values or len(set(values)) != len(values):
            raise ValueError(f"Empty or duplicate class inventory: {path}")
        return values
    coarse, fine = labels(coarse_path), labels(fine_path)
    definitions = json.loads(Path(definitions_path).read_text(encoding="utf-8")) if definitions_path else {}
    if definitions_path and set(definitions) != set(fine):
        raise ValueError("Definition keys must exactly match the fine-label inventory")
    taxonomy = {}
    for label in fine:
        parents = [parent for parent in coarse if label == "Other" == parent or label.startswith(parent + ": ")]
        if len(parents) != 1:
            raise ValueError(f"Cannot establish one explicit parent for {label!r}; create a manually verified mapping")
        definition = definitions.get(label, label)
        if not isinstance(definition, str) or not definition.strip():
            raise ValueError(f"Missing or empty definition for {label}")
        taxonomy[label] = {"parent": parents[0], "definition": definition,
                           "definition_source": "guideline_paraphrase" if definitions_path else "label_text_only"}
    if taxonomy.get("Other", {}).get("parent") != "Other":
        raise ValueError("Release inventories must include the global Other class")
    return taxonomy


def audit(rows):
    from collections import Counter
    validate(rows)
    articles = {r["article_id"]: r for r in rows}
    def domain(row):
        labels = row.get("labels", []) + [row.get("dominant_narrative", "")]
        domains = sorted({label.split(":", 1)[0] for label in labels if label.startswith(("CC:", "URW:"))})
        return "+".join(domains) or "not supplied"
    return {"examples": len(rows), "articles": len(articles),
            "examples_by_task": dict(Counter(r["task"] for r in rows)),
            "languages": dict(Counter(r["language"] for r in articles.values())),
            "domains_from_annotations_only": dict(Counter(domain(r) for r in articles.values())),
            "labels": dict(Counter(label for r in rows for label in r.get("labels", []))),
            "article_characters": {"min": min(len(r["text"]) for r in rows),
                                   "max": max(len(r["text"]) for r in rows)},
            "duplicate_content_articles": len(articles) - len({fingerprint(r["text"]) for r in articles.values()}),
            "near_duplicates": "Not automatically verified; inspect before final evaluation."}

