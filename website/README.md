# Project website

Buildless HTML/CSS/JavaScript. Serve `dist/` with any static web host.

```powershell
python -m http.server 8000 --directory website/dist
```

Open http://localhost:8000. Features: responsive project explanation, interactive
label-family diagram, method comparisons, expandable ablations, and filtered
aggregate findings. All article examples are invented and labelled illustrative.
No corpus text, live model endpoint, or official benchmark claims are published.

`dist/results.json` is intentionally empty until authorized experiments are run.
To add a reviewed measurement, use a `runs` record with: `dataset` exactly
`SemEval-2025 Task 10`, `measured: true`, `task` 1/2/3, `name`, `metric`, numeric
`value` 0–1, positive integer `count`, `split`, `commit`, and
`scoring_provenance`. Add language, runtime, seed, and limitations where available.
Scoring provenance must distinguish local metrics from organizer-verified scores.
Update the evidence-board status and verification notes alongside real results.

Never populate the board with synthetic test outcomes or other teams' scores.
