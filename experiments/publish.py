"""Export aggregate results and an SVG plot after manual provenance review.

Never publish articles, reference explanations, or predictions. This command does
not certify that a dataset is official; the researcher must verify that separately.
"""
import argparse
import html
import json
from pathlib import Path

from semeval.data import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary")
    parser.add_argument("--reviewed", action="store_true", help="Confirm data identity, split integrity, and aggregate release permission were reviewed")
    args = parser.parse_args()
    if not args.reviewed:
        parser.error("Review the source run and dataset identity before using --reviewed. Synthetic runs must never be published as benchmark results.")
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    if not summary.get("commit") or not summary.get("input_sha256"):
        raise ValueError("Source summary lacks commit or dataset hashes")
    runs = []
    for run in summary["runs"]:
        metric = run["primary_metric"]
        value = run["metrics"].get(metric)
        if value is None:
            continue
        runs.append({"dataset": "SemEval-2025 Task 10", "measured": True,
                     "task": run["task"], "name": run["name"], "metric": metric, "value": value,
                     "count": run["metrics"]["count"], "split": "held-out evaluation (see source manifest)",
                     "commit": summary["commit"], "scoring_provenance": run["scoring_provenance"],
                     "by_language": run["metrics"].get("by_language"), "seed": run.get("seed"),
                     "fit_seconds": run.get("fit_seconds"), "predict_seconds": run.get("predict_seconds"),
                     "process_peak_rss_mb": run.get("process_peak_rss_mb")})
    if not runs:
        raise ValueError("No measured primary metrics; nothing to publish")
    payload = {"status": "measured_local_metrics", "runs": runs,
               "input_sha256": summary["input_sha256"], "memory_note": summary.get("memory_note"),
               "organizer_scorer_verified": summary.get("organizer_scorer_verified", False)}
    write_json("results/measurements.json", payload)
    write_json("website/dist/results.json", payload)
    height = 100 + 40 * len(runs)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 {height}" role="img">',
             '<rect width="100%" height="100%" fill="#f5f4ed"/>',
             '<g font-family="Arial,sans-serif" fill="#172e32">',
             '<text x="25" y="30" font-size="18">Measured primary metrics · local scoring</text>',
             '<text x="25" y="52" font-size="11">Verify scorer provenance and split metadata before comparing to official leaderboard scores.</text>']
    for index, run in enumerate(runs):
        y = 85 + 40 * index
        parts += [f'<text x="25" y="{y + 14}" font-size="11">{html.escape(run["name"])}</text>',
                  f'<rect x="290" y="{y}" width="500" height="21" fill="#e1e7dc"/>',
                  f'<rect x="290" y="{y}" width="{500 * run["value"]:.2f}" height="21" fill="#275b53"/>',
                  f'<text x="810" y="{y + 15}" font-size="12">{run["value"]:.3f}</text>']
    parts += ['</g></svg>']
    Path("results/comparison.svg").write_text("\n".join(parts), encoding="utf-8")
    print("Exported reviewed aggregate metrics. Update website verification notes before deployment.")


if __name__ == "__main__":
    main()
