const families = {
  climate: { parent: 'Criticism of climate policies', children: ['Ineffective policies', 'Economic impact', 'Profit motives'] },
  media: { parent: 'Distrust towards Media', children: ['Western media as propaganda', 'Ukrainian media distrust'] }
};
document.querySelectorAll('[data-family]').forEach(button => {
  button.addEventListener('click', () => {
    const family = families[button.dataset.family];
    document.querySelectorAll('[data-family]').forEach(item => {
      const active = item === button;
      item.classList.toggle('active', active);
      item.setAttribute('aria-pressed', String(active));
    });
    document.getElementById('parent-label').textContent = family.parent;
    const children = document.getElementById('child-labels');
    children.replaceChildren(...family.children.map((text, index) => {
      const node = document.createElement('div');
      node.className = `graph-node${index === 1 ? ' selected' : ''}`;
      node.textContent = text;
      return node;
    }));
    children.style.gridTemplateColumns = `repeat(${family.children.length},1fr)`;
    document.querySelector('.branches').style.visibility = family.children.length === 3 ? 'visible' : 'hidden';
  });
});

let runs = [];
const methodNames = {
  tfidf: 'Character TF-IDF + logistic regression', lsa: 'LSA + logistic regression',
  transformer: 'Frozen MiniLM + logistic regression', graph: 'Frozen MiniLM + label GCN',
  'graph / self-loops': 'Self-loop GCN (ablation)',
  'graph / random-labels': 'Random label vectors (ablation)',
  'copy-input': 'Copy input (control)', 'label-template': 'Label template (control)',
  'textrank': 'TextRank extraction', 'semantic': 'Semantic extraction',
  'generate-semantic': 'Qwen: semantic evidence', 'generate-lead': 'Qwen: leading evidence',
  'generate-full': 'Qwen: full context', 'generate-no-premise': 'Qwen: premise withheld'
};
const emptyState = document.getElementById('results').cloneNode(true);
function renderResults() {
  const task = document.getElementById('task-filter').value;
  const dataset = document.getElementById('dataset-filter').value;
  const visible = runs.filter(run => (task === 'all' || String(run.task) === task) && (dataset === 'all' || run.dataset === dataset));
  const root = document.getElementById('results');
  document.getElementById('result-count').textContent = visible.length ? `${visible.length} measured variants` : 'No measured runs';
  root.replaceChildren();
  if (!visible.length) {
    root.append(...Array.from(emptyState.childNodes).map(node => node.cloneNode(true)));
    if (runs.length) {
      root.querySelector('h3').textContent = dataset === 'SemEval-2025 Task 10' ? 'No official SemEval scores.' : 'No runs match these filters.';
      root.querySelector('p').textContent = 'Public auxiliary experiments retain their native datasets and labels. They do not establish SemEval benchmark performance.';
    }
    return;
  }
  visible.forEach(run => {
    const row = document.createElement('article');
    row.className = 'result-row';
    const details = document.createElement('div');
    const title = document.createElement('strong');
    title.textContent = methodNames[run.name] || run.name;
    const metadata = document.createElement('p');
    metadata.textContent = `${run.dataset} · ${run.label_schema === 'native' ? 'Auxiliary, native labels' : 'Task 10'} · ${run.metric} · n=${run.count}${Number.isFinite(run.seed_std) ? ' · ' + run.seeds.length + ' seeds · SD ' + run.seed_std.toFixed(3) : ''}`;
    const provenance = document.createElement('p');
    provenance.textContent = run.scoring_provenance.replace('native-multilabel', 'multi-label');
    const languages = document.createElement('p');
    languages.textContent = Object.entries(run.by_language || {}).map(([language, score]) =>
      `${({ar:'Arabic',en:'English'})[language] || language}: ${score.value.toFixed(3)} (n=${score.count})`).join(' · ');
    const progress = document.createElement('progress');
    progress.max = 1; progress.value = run.value;
    progress.setAttribute('aria-label', `${run.metric}: ${run.value}`);
    const value = document.createElement('span');
    value.className = 'result-value'; value.textContent = run.value.toFixed(3);
    details.append(title, metadata, provenance, languages, progress);
    row.append(details, value); root.append(row);
  });
}
document.getElementById('task-filter').addEventListener('change', renderResults);
document.getElementById('dataset-filter').addEventListener('change', renderResults);
fetch('results.json', { cache: 'no-store' }).then(response => {
  if (!response.ok) throw new Error('Results could not be loaded');
  return response.json();
}).then(data => {
  if (!Array.isArray(data.runs)) throw new Error('Invalid results file');
  runs = data.runs.filter(run => typeof run.dataset === 'string' && run.dataset !== 'unspecified' && run.measured === true &&
    Number.isFinite(run.value) && run.value >= 0 && run.value <= 1 &&
    Number.isInteger(run.count) && run.count > 0 && run.name && run.metric &&
    run.split && run.commit && run.scoring_provenance && [1,2,3].includes(run.task));
  if (runs.length) {
    const status = document.querySelector('.status-pill');
    status.replaceChildren(document.createTextNode(runs.every(run => run.label_schema === 'native') ? 'MEASURED AUXILIARY EXPERIMENTS' : 'MEASURED RUNS · SEE SCORING PROVENANCE'));
    const filter = document.getElementById('dataset-filter');
    [...new Set(runs.map(run => run.dataset))].sort().filter(name => name !== 'SemEval-2025 Task 10').forEach(name => {
      const option = document.createElement('option'); option.value = name; option.textContent = name; filter.append(option);
    });
    if (runs.some(run => run.dataset === 'ProBel news grouped pilot')) filter.value = 'ProBel news grouped pilot';
  }
  renderResults();
}).catch(() => {
  document.getElementById('result-count').textContent = 'Results unavailable';
  const state = document.querySelector('#results .empty-results');
  state.querySelector('h3').textContent = 'Results file could not be loaded.';
  state.querySelector('p').textContent = 'The page cannot verify experiment availability. Open through the hosted website or a local HTTP server.';
  state.querySelector('strong').textContent = 'No scores are inferred from a loading error.';
});
