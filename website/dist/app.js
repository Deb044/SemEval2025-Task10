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
const emptyState = document.getElementById('results').cloneNode(true);
function renderResults() {
  const task = document.getElementById('task-filter').value;
  const visible = runs.filter(run => task === 'all' || String(run.task) === task);
  const root = document.getElementById('results');
  document.getElementById('result-count').textContent = visible.length ? `${visible.length} measured runs` : 'No measured runs';
  root.replaceChildren();
  if (!visible.length) {
    root.append(...Array.from(emptyState.childNodes).map(node => node.cloneNode(true)));
    return;
  }
  visible.forEach(run => {
    const row = document.createElement('article');
    row.className = 'result-row';
    const details = document.createElement('div');
    const title = document.createElement('strong');
    title.textContent = run.name;
    const metadata = document.createElement('p');
    metadata.textContent = `ST${run.task} · ${run.metric} · n=${run.count} · ${run.split} · ${run.scoring_provenance}`;
    const progress = document.createElement('progress');
    progress.max = 1; progress.value = run.value;
    progress.setAttribute('aria-label', `${run.metric}: ${run.value}`);
    const value = document.createElement('span');
    value.className = 'result-value'; value.textContent = run.value.toFixed(3);
    details.append(title, metadata, progress);
    row.append(details, value); root.append(row);
  });
}
document.getElementById('task-filter').addEventListener('change', renderResults);
fetch('results.json').then(response => {
  if (!response.ok) throw new Error('Results could not be loaded');
  return response.json();
}).then(data => {
  if (!Array.isArray(data.runs)) throw new Error('Invalid results file');
  runs = data.runs.filter(run => run.dataset === 'SemEval-2025 Task 10' && run.measured === true &&
    Number.isFinite(run.value) && run.value >= 0 && run.value <= 1 &&
    Number.isInteger(run.count) && run.count > 0 && run.name && run.metric &&
    run.split && run.commit && run.scoring_provenance && [1,2,3].includes(run.task));
  if (runs.length) {
    const status = document.querySelector('.status-pill');
    status.replaceChildren(document.createTextNode(data.organizer_scorer_verified ? 'MEASURED RUNS · ORGANIZER SCORER VERIFIED' : 'MEASURED RUNS · LOCAL SCORING'));
  }
  renderResults();
}).catch(() => {
  document.getElementById('result-count').textContent = 'Results unavailable';
  const state = document.querySelector('#results .empty-results');
  state.querySelector('h3').textContent = 'Results file could not be loaded.';
  state.querySelector('p').textContent = 'The page cannot verify experiment availability. Open through the hosted website or a local HTTP server.';
  state.querySelector('strong').textContent = 'No scores are inferred from a loading error.';
});
