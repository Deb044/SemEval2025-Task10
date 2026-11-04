"""Serve the project UI with loopback-only analysis and blinded review APIs.

Run: python -m semeval.web. Models, pasted text and review records stay local.
Only registered, locally trained model paths can be selected by the browser.
"""
import argparse
import json
import threading
import urllib.request
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .data import read_jsonl
from .review import save_review

ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    "cards-gcn": {"label": "Climate claims · MiniLM + hierarchical GCN", "language": ["en"],
        "paths": ["artifacts/extensions-cards/graph-none-42.joblib", "artifacts/public-cards-v2/graph-none-42.joblib"]},
    "probel-gcn": {"label": "Propaganda techniques · MiniLM + hierarchical GCN", "language": ["en", "ar"],
        "paths": ["artifacts/extensions-probel/graph-none-42.joblib", "artifacts/public-probel/graph-none-42.joblib"]},
    "probel-xlmr-gcn": {"label": "Propaganda techniques · XLM-R + hierarchical GCN", "language": ["en", "ar"],
        "paths": ["artifacts/extensions-probel/xlmr-graph-none-42.joblib"]},
}


def model_path(key):
    if key not in MODELS:
        raise ValueError("Select a registered project model")
    return next((ROOT / path for path in MODELS[key]["paths"] if (ROOT / path).is_file()), None)


@lru_cache(maxsize=3)
def load_model(path):
    import joblib
    return joblib.load(path)


def analyze(payload):
    from .classify import predict
    from .explain import explain_row
    from .propaganda import explain_propaganda
    key = payload.get("model", "cards-gcn")
    if key == "semeval-official":
        from .official import analyze as official_analyze
        return official_analyze(payload)
    path = model_path(key)
    if path is None:
        raise ValueError("Reproduce this model before selecting it")
    text = str(payload.get("text", "")).strip()
    language = payload.get("language", "en")
    if not text or len(text) > 50000:
        raise ValueError("Enter between 1 and 50,000 characters")
    if language not in MODELS[key]["language"]:
        raise ValueError("Select a supported evaluation language for this model")
    model = load_model(str(path))
    row = {"task": 2, "article_id": "local-input", "language": language, "text": text, "label_schema": "native"}
    prediction = predict(model, [row])[0]
    result = {"labels": prediction["labels"], "parents": prediction["narratives"],
        "label_details": [{"code": label, "description": model["taxonomy"][label]["definition"]} for label in prediction["labels"]],
        "scores": [{"label": label, "value": score} for label, score in sorted(prediction["scores"].items(), key=lambda pair: -pair[1])],
        "model": MODELS[key]["label"], "language": language, "explanation": None,
        "note": "Native dataset labels. Scores are model outputs, not calibrated truth probabilities."}
    result["trace"] = [
        {"stage": "Input", "state": "complete", "detail": f"{len(text)} characters; native {language} model input."},
        {"stage": "Frozen encoder", "state": "complete", "detail": model.get("encoder", "minilm") + " embeddings from the reproduced model."},
        {"stage": "Hierarchical label GNN", "state": "complete", "detail": f"{model['head'].initial.shape[0]} label/parent nodes; two graph-convolution layers."},
        {"stage": "Decoder", "state": "complete", "detail": f"{len(prediction['labels'])} native labels selected; parent categories derived from the model taxonomy."}]
    if payload.get("generate"):
        try:
            if key == "cards-gcn":
                explanation = explain_row(row, "generate", model["taxonomy"], "http://127.0.0.1:8080/v1", predicted=prediction, encoder=model["features"].encoder)
            else:
                explanation = explain_propaganda(dict(row, labels=prediction["labels"]), "generate-semantic", model["taxonomy"], "http://127.0.0.1:8080/v1", encoder=model["features"].encoder)
            from .explain import sentences
            lookup = {unit["id"]: unit["text"] for unit in sentences(text)}
            result["explanation"] = {"text": explanation["explanation"], "fallback": explanation["fallback"],
                "review_required": True, "evidence": [{"id": key, "text": lookup[key]} for key in explanation["evidence_ids"]]}
        except (OSError, TimeoutError):
            result["explanation_error"] = "The local generator is unavailable. Classification results are shown."
    result["trace"].append({"stage": "Evidence and explanation", "state": "complete" if result["explanation"] else "not_used",
        "detail": "Source evidence and the local generator were used; inspect whether a fallback was required." if result["explanation"] else "Generation was not requested or was unavailable."})
    return result


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "website/dist"), **kwargs)

    def local_request(self):
        host = self.headers.get("Host", "")
        if urlparse("http://" + host).hostname not in ("127.0.0.1", "localhost", "::1"):
            return False
        origin = self.headers.get("Origin")
        return not origin or urlparse(origin).hostname in ("127.0.0.1", "localhost", "::1")

    def respond(self, code, value):
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self.local_request():
            self.respond(403, {"error": "Local requests only"})
            return
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            try:
                with urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=1) as response:
                    generator = response.status == 200
            except OSError:
                generator = False
            self.respond(200, {"local": True, "generator": generator,
                "models": [{"id": key, "label": value["label"], "languages": value["language"], "available": model_path(key) is not None} for key, value in MODELS.items()],
                "official_taxonomy": True, "review_available": self.server.queue.is_file()})
        elif parsed.path == "/api/reviews":
            if not self.server.queue.is_file():
                self.respond(200, {"items": [], "message": "Generate the fixed review queue first"})
                return
            reviewer = parse_qs(parsed.query).get("reviewer", [""])[0]
            ratings = read_jsonl(self.server.ratings) if self.server.ratings.exists() else []
            completed = {row["review_id"] for row in ratings if row["reviewer"] == reviewer}
            queue = read_jsonl(self.server.queue)
            # Do not expose method names or original article IDs to the reviewer.
            self.respond(200, {"items": [{key: row[key] for key in ("review_id", "language", "text", "target_description", "explanation", "evidence_ids")} for row in queue if row["review_id"] not in completed],
                               "completed": len(completed), "total": len(queue)})
        elif parsed.path.startswith("/api/"):
            self.respond(404, {"error": "Unknown API route"})
        else:
            super().do_GET()

    def do_POST(self):
        if not self.local_request():
            self.respond(403, {"error": "Local requests only"})
            return
        if self.headers.get_content_type() != "application/json":
            self.respond(415, {"error": "Expected JSON"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 300000:
                raise ValueError("Invalid request size")
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object")
            with self.server.lock:
                if self.path == "/api/analyze":
                    result = analyze(payload)
                elif self.path == "/api/review":
                    result = save_review(payload, self.server.queue, self.server.ratings)
                else:
                    self.respond(404, {"error": "Unknown API route"})
                    return
            self.respond(200, result)
        except (ValueError, KeyError) as error:
            self.respond(400, {"error": str(error)})
        except Exception:
            self.respond(500, {"error": "Analysis failed. Inspect the local service output and model setup."})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--review-folder", default="artifacts/extensions-long-explanations")
    args = parser.parse_args()
    folder = Path(args.review_folder)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.lock = threading.Lock()
    server.queue, server.ratings = folder / "review-queue.jsonl", folder / "human-ratings.jsonl"
    print(f"Open http://127.0.0.1:{args.port} - local analysis and review", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
