"""Explicit official-format import, validation, and leakage checks.

Internal records are JSONL. Original article text is never normalized: offsets
refer to that exact Unicode string. The user must choose the offset convention
documented by their corpus release, rather than silently repairing mismatches.
"""
import hashlib
import json
from pathlib import Path

LANGUAGES = {"EN": "en", "HI": "hi", "BG": "bg", "PT": "pt", "RU": "ru"}
ROLE_GROUPS = {
    "Protagonist": ["Guardian", "Martyr", "Peacemaker", "Rebel", "Underdog", "Virtuous"],
    "Antagonist": ["Instigator", "Conspirator", "Tyrant", "Foreign Adversary", "Traitor",
                   "Spy", "Saboteur", "Corrupt", "Incompetent", "Terrorist", "Deceiver", "Bigot"],
    "Innocent": ["Forgotten", "Exploited", "Victim", "Scapegoat"],
}
ROLE_PARENT = {role: parent for parent, roles in ROLE_GROUPS.items() for role in roles}


def read_jsonl(path):
    with open(path, encoding="utf-8-sig") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    return rows


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def fingerprint(text):
    """Whitespace-insensitive exact duplicate hash, NOT a near-duplicate detector."""
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def identity(row):
    if row["task"] == 1:
        return (row["article_id"], row["start"], row["end"])
    return (row["article_id"],)


def validate(rows):
    seen = set()
    for row in rows:
        key = (row["task"], *identity(row))
        if key in seen:
            raise ValueError(f"Duplicate example: {key}")
        seen.add(key)
        if row["language"] not in LANGUAGES.values() or not row["text"].strip():
            raise ValueError(f"Invalid language or empty article: {key}")
        if row["task"] == 1:
            if row["text"][row["start"]:row["end_exclusive"]] != row["mention"]:
                raise ValueError(f"Entity offset mismatch: {key}")
            for role in row.get("labels", []):
                if ROLE_PARENT.get(role) != row.get("main_role"):
                    raise ValueError(f"Unknown or incompatible entity role: {role}")


def check_splits(*splits):
    """Reject article-ID/content overlap, even across different subtasks."""
    owners = {}
    for index, rows in enumerate(splits):
        validate(rows)
        for row in rows:
            for key in ("id:" + row["article_id"], "text:" + fingerprint(row["text"])):
                if key in owners and owners[key] != index:
                    raise ValueError(f"Article leakage between splits: {row['article_id']}")
                owners[key] = index


def import_official(articles, annotations, task, offset_end="exclusive"):
    files = {}
    for path in Path(articles).rglob("*.txt"):
        if path.name in files:
            raise ValueError(f"Ambiguous article basename: {path.name}")
        files[path.name] = path
    rows = []
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
                if len(fields) < 6:
                    raise ValueError(f"Line {number}: expected entity gold columns")
                row.update(mention=fields[1], start=int(fields[2]), end=int(fields[3]),
                           end_exclusive=int(fields[3]) + (offset_end == "inclusive"),
                           main_role=fields[4], labels=fields[5:])
            elif task == 2:
                if len(fields) != 3:
                    raise ValueError(f"Line {number}: expected three columns")
                row.update(narratives=fields[1].split(";"), labels=fields[2].split(";"))
            elif task == 3:
                if len(fields) != 4:
                    raise ValueError(f"Line {number}: expected four gold columns")
                row.update(dominant_narrative=fields[1], dominant_subnarrative=fields[2], explanation=fields[3])
            rows.append(row)
    validate(rows)
    return rows


def load_taxonomy(path):
    """Use exact release label strings; do not guess parents from colon splitting."""
    taxonomy = json.loads(Path(path).read_text(encoding="utf-8"))
    if not taxonomy or not all(isinstance(v.get("parent"), str) and isinstance(v.get("definition"), str)
                               for v in taxonomy.values()):
        raise ValueError("Taxonomy must map fine labels to parent and definition")
    return taxonomy


def audit(rows):
    from collections import Counter
    validate(rows)
    articles = {r["article_id"]: r for r in rows}
    return {"examples": len(rows), "articles": len(articles),
            "languages": dict(Counter(r["language"] for r in articles.values())),
            "labels": dict(Counter(label for r in rows for label in r.get("labels", []))),
            "article_characters": {"min": min(len(r["text"]) for r in rows),
                                   "max": max(len(r["text"]) for r in rows)},
            "duplicate_content_articles": len(articles) - len({fingerprint(r["text"]) for r in articles.values()}),
            "near_duplicates": "Not automatically verified; inspect before final evaluation."}

