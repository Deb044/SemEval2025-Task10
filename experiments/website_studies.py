"""Build five source-grounded study narratives and standalone scientific SVG plots."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('website/dist')
PLOTS = ROOT / 'plots'
COLORS = ['#728391', '#4b6073', '#27485c', '#167181']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.labelcolor': '#314759', 'text.color': '#172b3a',
    'xtick.color': '#526473', 'ytick.color': '#526473', 'svg.fonttype': 'path',
    'axes.edgecolor': '#d6dfe4', 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    'svg.hashsalt': 'narrative-lab-five-studies-v1'})


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(fig, name):
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / name, bbox_inches='tight', metadata={'Date': None})
    plt.close(fig)
    return 'plots/' + name


def bars(axis, labels, values, deviations=None, maximum=None):
    positions = np.arange(len(labels))
    axis.barh(positions, values, color=[COLORS[min(i, 3)] for i in positions], height=.58,
              xerr=deviations, error_kw={'ecolor': '#172b3a', 'capsize': 3, 'linewidth': 1})
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, maximum or max(values) * 1.25)
    axis.grid(axis='x', alpha=.15)
    axis.set_axisbelow(True)
    for index, value in enumerate(values):
        axis.text(value + axis.get_xlim()[1] * .025, index, f'{value:.4f}', va='center', fontsize=9)


def main():
    entries = load('results/experiments.json')['experiments']
    def rows(study):
        return [row for row in entries if row['study'] == study]
    def lookup(study, name):
        return next(row for row in rows(study) if row['name'] == name)
    cards = 'CARDS · extended hierarchy study'
    probel = 'ProBel · extended bilingual study'
    studies, featured = [], set()
    names = ['Character TF-IDF + logistic regression', 'LSA + logistic regression',
             'MiniLM + logistic regression', 'MiniLM + hierarchical GCN']
    labels = ['TF-IDF + LR', 'LSA + LR', 'MiniLM + LR', 'MiniLM + GCN']
    selected = [lookup(cards, name) for name in names]
    featured.update(row['id'] for row in selected)
    fig, ax = plt.subplots(figsize=(10, 3.2), layout='constrained')
    bars(ax, labels, [r['value'] for r in selected], [r.get('seed_std') or 0 for r in selected], .65)
    ax.set_xlabel('Fixed-inventory macro F1 · higher is better')
    studies.append({'id': 'cards', 'title': 'Can the core model classify hierarchical climate claims?',
        'aim': 'Compare the requested multilingual-transformer + hierarchical label GNN against lexical, latent-semantic and transformer-only classifiers.',
        'protocol': 'CARDS/4C: 6,000 training, 1,000 development and 2,897 held-out English records; 18 native claim labels. The frozen MiniLM encoder feeds a trainable two-layer label GCN. GCN results average seeds 13, 42 and 87; comparison heads use seed 42. Hyperparameters are selected on development data.',
        'finding': f"The core GCN scored {selected[3]['value']:.4f} macro F1, while LSA + logistic regression scored {selected[1]['value']:.4f}. The graph model did not beat the strongest baseline on this split. This establishes a measured comparison rather than an assumed benefit from greater model complexity.",
        'plot': save(fig, 'cards-classification.svg'), 'plot_alt': 'Macro F1 for four CARDS classifiers, with GCN seed standard deviation.',
        'caption': 'Bars start at zero. Error bars show training-seed standard deviation for the three-seed GCN; other heads were run once.',
        'headers': ['Model', 'Macro F1', 'Seed SD'], 'rows': [[label, f"{row['value']:.4f}", f"{row['seed_std']:.4f}" if row.get('seed_std') is not None else 'One seed'] for label, row in zip(labels, selected)],
        'limits': 'Native climate-claim classification on English passages. These are not official SemEval narrative scores or a multilingual evaluation.', 'source': 'results/experiments.json'})
    pnames = ['MiniLM + logistic regression', 'MiniLM + hierarchical GCN', 'XLM-R + logistic regression', 'XLM-R + hierarchical GCN']
    plabels = ['MiniLM\n+ LR', 'MiniLM\n+ GCN', 'XLM-R\n+ LR', 'XLM-R\n+ GCN']
    selected = [lookup(probel, name) for name in pnames]
    featured.update(row['id'] for row in selected)
    fig, ax = plt.subplots(figsize=(10, 3.8), layout='constrained')
    x = np.arange(4)
    for offset, (key, color, label) in enumerate([('overall', '#27485c', 'Pooled test set'), ('en', '#91a0aa', 'English'), ('ar', '#167181', 'Arabic')]):
        values = [row['value'] if key == 'overall' else row['by_language'][key]['value'] for row in selected]
        ax.bar(x + (offset - 1) * .23, values, .23, label=label, color=color)
    ax.set_xticks(x, plabels)
    ax.set_ylabel('Fixed-inventory macro F1')
    ax.set_ylim(0, .15)
    ax.legend(frameon=False, ncol=3, loc='upper center')
    ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    studies.append({'id': 'bilingual', 'title': 'Does the encoder change bilingual propaganda classification?',
        'aim': 'Compare frozen multilingual MiniLM and XLM-R under linear and hierarchical graph heads, and inspect English/Arabic differences.',
        'protocol': 'ProBel news pilot: article-grouped splits with 1,600 training, 400 development and 400 held-out records. The test set has 200 English and 200 Arabic records. Predict 23 native persuasion techniques, grouped into six parent categories. Graph heads use the same three seeds.',
        'finding': f"XLM-R + GCN reached {selected[3]['value']:.4f} pooled macro F1, compared with {selected[1]['value']:.4f} for MiniLM + GCN. XLM-R + linear heads scored {selected[2]['value']:.4f}. The encoder comparison therefore depends on the prediction head; the result does not show that XLM-R is always better.",
        'plot': save(fig, 'bilingual-encoders.svg'), 'plot_alt': 'Overall, English and Arabic macro F1 for MiniLM and XLM-R with linear and GCN heads.',
        'caption': 'The pooled score is computed across the 400 test records; it is not the average of the two language scores. Bars show means where three graph seeds were run.',
        'headers': ['Encoder + head', 'Pooled F1', 'English F1', 'Arabic F1'],
        'rows': [[r['name'], f"{r['value']:.4f}", f"{r['by_language']['en']['value']:.4f}", f"{r['by_language']['ar']['value']:.4f}"] for r in selected],
        'limits': 'This is a small researcher-defined grouped pilot. Scores are low, rare labels are challenging, and persuasion techniques describe rhetoric rather than narrative topics.', 'source': 'results/probel-protocol.json'})
    suffixes = ['', ' · self-loops', ' · random-labels', ' · shuffled-edges']
    control_labels = ['Full hierarchy', 'Self-loops only', 'Random label vectors', 'Shuffled parent edges']
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.7), layout='constrained')
    table = []
    for ax, study, title, limit in zip(axes, [cards, probel], ['CARDS', 'ProBel'], [.65, .15]):
        group = [lookup(study, 'MiniLM + hierarchical GCN' + suffix) for suffix in suffixes]
        featured.update(row['id'] for row in group)
        bars(ax, control_labels, [r['value'] for r in group], [r['seed_std'] for r in group], limit)
        ax.set_title(title); ax.set_xlabel('Macro F1')
        for label, row in zip(control_labels, group): table.append([title, label, f"{row['value']:.4f} ± {row['seed_std']:.4f}"])
    studies.append({'id': 'graph', 'title': 'Does the label hierarchy itself help?',
        'aim': 'Isolate graph structure and label semantics while keeping the encoder, examples, graph-head architecture, training settings and seeds fixed.',
        'protocol': 'Compare the full MiniLM–GCN with self-loops only, random initial label vectors, and shuffled parent–child edges that preserve node degrees. Each control uses seeds 13, 42 and 87 on both CARDS and ProBel. Self-loops remove message passing between labels; shuffled edges test the correctness of the taxonomy.',
        'finding': 'CARDS favors the full hierarchy over these graph controls. ProBel favors self-loops over the full hierarchy. The evidence therefore does not support a consistent hierarchy benefit across datasets. The linear-head comparison in the other studies changes more than graph structure and cannot isolate this effect.',
        'plot': save(fig, 'graph-ablations.svg'), 'plot_alt': 'Full graph, self-loop, random-label and shuffled-edge macro F1 with seed variation on CARDS and ProBel.',
        'caption': 'Error bars are standard deviations over three training seeds. Panels use different, explicitly labeled axis ranges. Paired bootstrap diagnostics remain in the aggregate catalog.',
        'headers': ['Dataset', 'Graph condition', 'Macro F1 ± SD'], 'rows': table,
        'limits': 'These controls test this architecture and split protocol. A higher mean on one dataset does not establish a general causal improvement.', 'source': 'results/experiments.json'})
    generated, reviewed = load('results/generator-comparison.json'), load('results/named-review.json')
    models = ['qwen3', 'smollm2', 'tinyllama']
    g = {r['model']: r for r in generated['runs']}
    j = {r['model']: r for r in reviewed['runs']}
    display = ['Qwen3 1.7B', 'SmolLM2 1.7B', 'TinyLlama 1.1B']
    if any(g[key]['count'] != 24 or j[key]['reviewed_outputs'] != 6 for key in models):
        raise ValueError('Featured generator study requires 24 generations and six reviewed outputs per model')
    valid_gap = abs(round(g['smollm2']['valid_generation_rate'] * 24) - round(g['qwen3']['valid_generation_rate'] * 24))
    output_word = 'output' if valid_gap == 1 else 'outputs'
    tiny_note = 'TinyLlama exceeded the word limit in every case. ' if g['tinyllama']['raw_length_compliance'] == 0 else ''
    featured.update('generator-' + key for key in models)
    featured.update('named-judge-' + key for key in models)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), layout='constrained')
    for ax, values, title, maximum in [(axes[0], [j[k]['llm_overall_mean'] for k in models], 'Named-judge mean · 0–2', 2),
        (axes[1], [g[k]['valid_generation_rate'] for k in models], 'Valid raw generation · fraction', 1),
        (axes[2], [g[k]['median_seconds'] for k in models], 'Median generation · seconds', None)]:
        ax.bar(np.arange(3), values, color=[COLORS[3], COLORS[1], COLORS[0]], width=.6)
        ax.set_xticks(np.arange(3), display, rotation=25, ha='right', fontsize=8)
        ax.set_title(title, fontsize=10, pad=18); ax.set_ylim(0, maximum * 1.12 if maximum else max(values) * 1.2)
        if maximum: ax.set_yticks(np.linspace(0, maximum, 5))
        ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
        for i, value in enumerate(values): ax.annotate(f'{value:.2f}', (i, value), xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9)
    best = max(j[k]['llm_overall_mean'] for k in models)
    leaders = [generated['models'][k]['name'] for k in models if abs(j[k]['llm_overall_mean'] - best) < 1e-9]
    viable = [key for key in models if g[key]['valid_generation_rate'] > 0]
    preferred = max(viable, key=lambda key: (j[key]['llm_overall_mean'], -g[key]['median_seconds'])) if viable else None
    choice_note = (f"Among models that produced valid outputs, {g[preferred]['name']} had the highest mean judge rating. The small sample supports a tentative choice, not a definitive ranking. " if preferred else '')
    human_count = reviewed['human_paired_items']
    human_note = (f"There are {human_count} source-bound human/LLM paired items; paired means are reported with their own coverage in the appendix."
                  if human_count else 'Review scores are from the named Qwen2.5-3B-Instruct LLM judge. The plot averages its four rubric criteria over the same six source passages per generator.')
    studies.append({'id': 'generation', 'title': 'Which small generator produces the most useful explanation?',
        'aim': 'Compare Qwen3-1.7B with two additional generators, SmolLM2-1.7B-Instruct and TinyLlama-1.1B-Chat-v1.0, under a common explanation prompt and evidence budget.',
        'protocol': 'All three models generate once for the same 24 held-out English CARDS passages using supplied native labels, identical semantically selected evidence (up to 800 tokens/six sentences), a 2,048-token context, 256 output tokens, temperature 0, seed 42 and Q4_K_M weights. The separately pinned Qwen2.5-3B-Instruct judge reviews a blinded 18-output subset: the same six passages from every generator. It rates source support, attribution, target relevance and readability from 0 to 2. Quality is scored on raw outputs before fallback.',
        'finding': f"Highest mean judge rating in this pilot: {', '.join(leaders)} ({best:.2f}/2). SmolLM2 and Qwen3 differed by {valid_gap} valid raw {output_word} out of 24. " + tiny_note + "Judge quality scores therefore need to be read alongside format compliance and latency. " + choice_note + human_note,
        'plot': save(fig, 'generator-comparison.svg'), 'plot_alt': 'Three generators compared by named-judge rating, valid-generation rate and median generation latency.',
        'caption': 'Judge ratings average four rubric criteria across six outputs per generator. Format and latency cover 24 outputs per generator. Quality bars score raw outputs and do not include extractive fallback credit. Review coverage is stated separately.',
        'headers': ['Generator', 'Judge mean /2 (n=6)', 'Fully supported', 'Valid output (n=24)', 'Fallback', 'Median seconds'],
        'rows': [[g[k]['name'], f"{j[k]['llm_overall_mean']:.2f}", f"{j[k]['llm_fully_supported_rate']:.0%}", f"{round(g[k]['valid_generation_rate'] * g[k]['count'])}/{g[k]['count']} ({g[k]['valid_generation_rate']:.0%})", f"{g[k]['fallback_rate']:.0%}", f"{g[k]['median_seconds']:.2f}"] for k in models],
        'review_headers': ['Generator', 'Source support /2', 'Attribution /2', 'Relevance /2', 'Readability /2'],
        'review_rows': [[g[k]['name'], *[f"{j[k]['llm_mean_ratings'][criterion]:.2f}" for criterion in ('supported', 'attribution', 'relevance', 'readability')]] for k in models],
        'limits': 'Small English pilot with one quantized judge. The judge shares the Qwen family with one generator, so family preference is possible. This is not evidence of multilingual generation quality or a universal model ranking.',
        'source': 'results/named-review.json'})
    query_rows = rows('Data processing scalability')
    sizes = sorted({row['n'] for row in query_rows})
    scale = {'measurements': [{'count': n,
        'json': next(r['details'] for r in query_rows if r['n'] == n and r['name'] == 'JSON aggregation'),
        'parquet': next(r['details'] for r in query_rows if r['n'] == n and r['name'] == 'Parquet aggregation')}
        for n in sizes], 'embeddings': sorted([r['details'] for r in rows('Embedding cache scalability')
            if r['name'] == 'MiniLM · cached'], key=lambda r: r['count'])}
    spark = load('results/spark-audit.json')
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.7), layout='constrained')
    sizes = [r['count'] for r in scale['measurements']]
    for key, color, label in [('json', COLORS[1], 'JSON'), ('parquet', COLORS[3], 'Parquet')]:
        axes[0].plot(sizes, [r[key]['first_scan_seconds'] for r in scale['measurements']], marker='o', color=color, label=label)
    axes[0].set_xlabel('Unique news texts'); axes[0].set_ylabel('First query · seconds')
    axes[0].set_ylim(0, 1.8); axes[0].legend(frameon=False); axes[0].grid(alpha=.15)
    for offset, (mode, color) in enumerate([('uncached', COLORS[1]), ('cached', COLORS[3])]):
        axes[1].bar(np.arange(len(scale['embeddings'])) + offset * .32, [r[mode + '_seconds'] for r in scale['embeddings']], .32, label=mode.title(), color=color)
    axes[1].set_xticks(np.arange(len(scale['embeddings'])) + .16, [str(r['count']) for r in scale['embeddings']])
    axes[1].set_xlabel('Texts encoded'); axes[1].set_ylabel('Embedding time · seconds'); axes[1].legend(frameon=False); axes[1].grid(axis='y', alpha=.15); axes[1].set_axisbelow(True)
    selected_scale = [r for r in entries if r['study'] in ('Data processing scalability', 'Embedding cache scalability', 'Apache Spark SQL input audit') and r['name'] != 'JSON → Parquet']
    featured.update(r['id'] for r in selected_scale)
    cache128 = next(r for r in scale['embeddings'] if r['count'] == 128)
    table = [['Spark SQL input gate', str(spark['records']), f"{spark['audit_seconds']:.3f} s audit", 'Zero invalid/duplicate/overlap groups']]
    table += [[f"{r['count']} texts: JSON / Parquet", str(r['count']), f"{r['json']['first_scan_seconds']:.3f} / {r['parquet']['first_scan_seconds']:.3f} s", 'Same aggregate query'] for r in scale['measurements']]
    table += [[f"Embedding cache ({r['count']} texts)", str(r['count']), f"{r['uncached_seconds']:.3f} / {r['cached_seconds']:.3f} s", 'Uncached / cached; vectors identical'] for r in scale['embeddings']]
    studies.append({'id': 'systems', 'title': 'What does the data pipeline contribute?',
        'aim': 'Measure the practical data-engineering components: Spark SQL validation, columnar querying and reuse of expensive transformer embeddings.',
        'protocol': 'Spark SQL local[2] checks all six prepared classification splits before training. Separate isolated DuckDB workers compare JSON and Parquet aggregation on 1,000, 5,000, 10,000 and 25,000 real unique news texts from a 40,749-text pool. Embedding trials compare fresh cache and disk-cache reuse for 32 and 128 texts, with model loading timed separately.',
        'finding': f"Spark validated {spark['records']:,} records with zero invalid records, duplicate groups or checked split overlaps. Parquet was not consistently faster for these small first-scan queries. Reusing embeddings reduced the 128-text encoding stage from {cache128['uncached_seconds']:.2f} to {cache128['cached_seconds']:.2f} seconds while retaining identical vectors.",
        'plot': save(fig, 'data-pipeline.svg'), 'plot_alt': 'JSON versus Parquet first-query timings at four data sizes, and cached versus uncached embedding timings.',
        'caption': 'Each point is a local measurement, not a multi-run estimate. First scan includes process/engine effects and may benefit from OS caching. Embedding times exclude loading the model weights.',
        'headers': ['Operation', 'Records', 'Measured time', 'Interpretation'], 'rows': table,
        'limits': 'Spark runs on one machine. These workloads do not demonstrate distributed execution, out-of-core processing or a general Parquet speed advantage.', 'source': 'results/spark-audit.json'})
    import xml.etree.ElementTree as ET
    for study in studies:
        box = ET.parse(ROOT / study['plot']).getroot().attrib['viewBox'].split()
        study['plot_width'], study['plot_height'] = float(box[2]), float(box[3])
    payload = {'studies': studies, 'featured_result_ids': sorted(featured),
        'appendix_count': sum(r['id'] not in featured for r in entries),
        'review_note': human_note, 'judge_model': reviewed['judge']['name']}
    (ROOT / 'studies.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    Path('results/featured-studies.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('Built five studies and five SVG plots')


if __name__ == '__main__':
    main()
