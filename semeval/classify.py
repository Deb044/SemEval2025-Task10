"""Comparable lexical, latent-semantic, transformer and label-graph experiments."""
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

from .data import ROLE_PARENT, check_splits, load_taxonomy
from .features import Features, example_text
from .metrics import classification_scores


class BinaryHeads:
    """One binary model per label, including constant targets absent in training."""
    def fit(self, x, y, regularization):
        self.heads = []
        for column in y.T:
            if np.unique(column).size == 1:
                self.heads.append(float(column[0]))
            else:
                head = LogisticRegression(C=regularization, max_iter=1000, solver="liblinear", random_state=42)
                head.fit(x, column)
                self.heads.append(head)
        return self

    def probabilities(self, x):
        return np.column_stack([np.full(x.shape[0], h) if isinstance(h, float) else h.predict_proba(x)[:, 1]
                                for h in self.heads])


def decode(probabilities, labels, threshold, task, decision="multilabel"):
    decoded = []
    for scores in probabilities:
        if decision == "single":
            decoded.append([labels[int(scores.argmax())]])
            continue
        chosen = [label for label, score in zip(labels, scores) if score >= threshold]
        if decision == "native-multilabel":
            decoded.append(chosen)  # Empty set is the native no-technique annotation.
            continue
        if task == 1:
            # A mention has exactly one broad role; maximize its best fine-role score.
            best = labels[int(scores.argmax())]
            chosen = [label for label in chosen if ROLE_PARENT[label] == ROLE_PARENT[best]] or [best]
        else:
            # Global Other cannot coexist with a specific label in our decoder.
            chosen = [label for label in chosen if label != "Other"] or ["Other"]
        decoded.append(chosen)
    return decoded


def targets(rows, labels):
    unknown = {label for r in rows for label in r["labels"]} - set(labels)
    if unknown:
        raise ValueError(f"Labels missing from training taxonomy: {sorted(unknown)}")
    return np.asarray([[label in row["labels"] for label in labels] for row in rows], dtype=np.float32)


def train(train_rows, dev_rows, method, taxonomy_path=None, seed=42, ablation="none", native=False,
          native_decision="single", encoder="minilm"):
    check_splits(train_rows, dev_rows)
    task = train_rows[0]["task"]
    if task not in (1, 2) or any(r["task"] != task for r in train_rows + dev_rows):
        raise ValueError("Classification inputs must contain one task: 1 or 2")
    if task == 2 and not taxonomy_path:
        raise ValueError("ST2 requires an exact-release taxonomy mapping")
    if native and any(row.get("label_schema") != "native" for row in train_rows + dev_rows):
        raise ValueError("Native experiments require explicitly identified native-schema records")
    if not native and any(row.get("label_schema") == "native" for row in train_rows + dev_rows):
        raise ValueError("Use native=True for auxiliary labels; never treat them as SemEval roles")
    if native_decision not in ("single", "multilabel") or (not native and native_decision != "single"):
        raise ValueError("Native decision must be single or multilabel and requires native=True")
    taxonomy = load_taxonomy(taxonomy_path, require_other=not native) if task == 2 or native else None
    labels = sorted(taxonomy if taxonomy else ROLE_PARENT)
    y, yd = targets(train_rows, labels), targets(dev_rows, labels)
    started = time.perf_counter()
    features = Features("transformer" if method == "graph" else method, seed, encoder)
    x = features.fit_transform([example_text(r) for r in train_rows])
    xd = features.transform([example_text(r) for r in dev_rows])
    metric = "macro_f1" if native else ("exact_match" if task == 1 else "samples_f1")
    decision = ("single" if native_decision == "single" else "native-multilabel") if native else "multilabel"
    best = None
    if method == "graph":
        if task != 2:
            raise ValueError("Label graph is implemented for ST2")
        from .graph import fit_graph
        head, probabilities = fit_graph(x, y, xd, yd, labels, taxonomy, seed, ablation, features.encoder)
        candidates = [(head, probabilities, None)]
    else:
        candidates = []
        for regularization in (.1, 1., 10.):
            head = BinaryHeads().fit(x, y, regularization)
            candidates.append((head, head.probabilities(xd), regularization))
    for head, probabilities, regularization in candidates:
        for threshold in ([.5] if decision == "single" else np.arange(.1, 1., .1)):
            predictions = decode(probabilities, labels, threshold, task, decision)
            score = classification_scores(dev_rows, predictions, classes=labels)[metric]
            if best is None or score > best[0]:
                best = (score, head, float(threshold), regularization)
    return {"task": task, "method": method, "encoder": encoder, "features": features, "head": best[1], "labels": labels,
            "threshold": best[2], "regularization": best[3], "taxonomy": taxonomy, "seed": seed,
            "ablation": ablation, "development_metric": metric, "development_score": best[0],
            "native": native, "decision": decision,
            "fit_seconds": time.perf_counter() - started}


def predict(model, rows):
    if any(r["task"] != model["task"] for r in rows):
        raise ValueError("Prediction task differs from trained task")
    x = model["features"].transform([example_text(r) for r in rows])
    if model["method"] == "graph":
        from .graph import graph_probabilities
        probabilities = graph_probabilities(model["head"], x)
    else:
        probabilities = model["head"].probabilities(x)
    labels = decode(probabilities, model["labels"], model["threshold"], model["task"], model.get("decision", "multilabel"))
    result = []
    for row, chosen, scores in zip(rows, labels, probabilities, strict=True):
        record = {key: row[key] for key in ("task", "article_id", "language")}
        record.update(labels=chosen, scores=dict(zip(model["labels"], scores.astype(float).tolist())))
        if "dataset" in row:
            record["dataset"] = row["dataset"]
        if model.get("native"):
            record["label_schema"] = "native"
            record["decision"] = model.get("decision", "single")
        if model["task"] == 1:
            record.update({key: row[key] for key in ("mention", "start", "end")})
            if not model.get("native"):
                record["main_role"] = ROLE_PARENT[chosen[0]]
        else:
            record["narratives"] = sorted({model["taxonomy"][label]["parent"] for label in chosen})
        result.append(record)
    return result

