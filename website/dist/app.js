/* Project UI: presentation, aggregate evidence, and a separate local workspace. */
const byId = id => document.getElementById(id);
function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

const tabs = [...document.querySelectorAll('[data-tab]')];
function activateTab(name, focus = false) {
  if (!tabs.some(tab => tab.dataset.tab === name)) name = 'overview';
  tabs.forEach(tab => {
    const active = tab.dataset.tab === name;
    tab.setAttribute('aria-selected', String(active));
    tab.tabIndex = active ? 0 : -1;
    byId(tab.dataset.tab).hidden = !active;
    if (active && focus) tab.focus();
  });
  history.replaceState(null, '', '#' + name);
  window.scrollTo(0, 0);
}
tabs.forEach((tab, index) => {
  tab.addEventListener('click', () => activateTab(tab.dataset.tab));
  tab.addEventListener('keydown', event => {
    if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
      event.preventDefault();
      const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 :
        (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
      activateTab(tabs[next].dataset.tab, true);
    }
  });
});
window.addEventListener('hashchange', () => activateTab(location.hash.slice(1)));
activateTab(location.hash.slice(1));

let experiments = [];
let featuredResultIds = new Set();
function resultText(row) {
  if (Number.isFinite(row.value)) return row.value.toFixed(row.unit === 'seconds' ? 3 : 4);
  return row.status === 'awaiting_human_review' ? 'Review pending' : 'Not measured';
}
function renderExperiments() {
  const study = byId('study-filter').value;
  const kind = byId('kind-filter').value;
  const query = byId('experiment-search').value.toLowerCase().trim();
  const rows = experiments.filter(row => !featuredResultIds.has(row.id) && (study === 'all' || row.study === study) &&
    (kind === 'all' || row.kind === kind) && (!query || [row.name, row.dataset, row.study, row.metric].join(' ').toLowerCase().includes(query)));
  byId('experiment-count').textContent = `${rows.length} result entries`;
  const root = byId('experiment-rows');
  root.replaceChildren();
  rows.forEach(row => {
    const tr = element('tr');
    const studyCell = element('td');
    studyCell.append(element('strong', row.study), element('p', row.dataset));
    const method = element('td');
    method.append(element('strong', row.name));
    if (row.role) method.append(element('br'), element('span', row.role, 'badge'));
    const result = element('td');
    result.append(element('span', resultText(row), 'result'), element('p', row.metric || 'Quality not scored'));
    if (row.unit) result.append(element('p', row.unit));
    if (Number.isFinite(row.seed_std)) result.append(element('p', `Seed SD ${row.seed_std.toFixed(4)}`));
    const evaluation = element('td');
    evaluation.append(element('span', row.n ? `n = ${row.n}` : 'See protocol'));
    const languages = Object.entries(row.by_language || {}).map(([code, data]) => `${code.toUpperCase()} ${data.value.toFixed(3)}`).join(' · ');
    if (languages) evaluation.append(element('p', languages));
    if (row.seeds?.length) evaluation.append(element('p', `${row.seeds.length} seed${row.seeds.length > 1 ? 's' : ''}`));
    if (row.format) evaluation.append(element('p', row.format));
    const detailsCell = element('td');
    const details = element('details');
    details.append(element('summary', 'Inspect'), element('p', Array.isArray(row.notes) ? row.notes.join(' ') : (row.notes || '')));
    if (row.provenance?.commit) {
      const link = element('a', 'Source commit ↗');
      link.href = `https://github.com/Deb044/SemEval2025-Task10/commit/${encodeURIComponent(row.provenance.commit)}`;
      details.append(link);
    }
    details.append(element('pre', JSON.stringify(row.details || row.provenance || {}, null, 2)));
    detailsCell.append(details);
    tr.append(studyCell, method, result, evaluation, detailsCell);
    root.append(tr);
  });
  byId('experiment-message').textContent = rows.length ? '' : 'No results match these filters.';
}
['study-filter', 'kind-filter'].forEach(id => byId(id).addEventListener('change', renderExperiments));
byId('experiment-search').addEventListener('input', renderExperiments);
function studyTable(headers, rows, caption) {
  const wrap = element('div', undefined, 'table-wrap'); const table = element('table', undefined, 'study-table');
  const head = element('thead'); const header = element('tr');
  headers.forEach(text => { const cell = element('th', text); cell.scope = 'col'; header.append(cell); }); head.append(header);
  const body = element('tbody'); rows.forEach(values => { const row = element('tr'); values.forEach(value => row.append(element('td', String(value)))); body.append(row); });
  table.append(element('caption', caption, 'sr-only'), head, body); wrap.append(table); return wrap;
}
function renderStudies(data) {
  featuredResultIds = new Set(data.featured_result_ids || []);
  const root = byId('featured-studies'); root.replaceChildren();
  const navigation = byId('study-navigation'); navigation.replaceChildren();
  data.studies.forEach((study, index) => {
    const article = element('article', undefined, 'study-card'); article.id = 'study-' + study.id;
    const jump = element('button', `${String(index + 1).padStart(2, '0')} ${['Classification', 'Bilingual encoders', 'Graph controls', 'Generators & judge', 'Data pipeline'][index]}`);
    jump.type = 'button'; jump.addEventListener('click', () => article.scrollIntoView({behavior: 'smooth', block: 'start'})); navigation.append(jump);
    article.append(element('p', `STUDY ${String(index + 1).padStart(2, '0')}`, 'eyebrow'), element('h3', study.title));
    const context = element('div', undefined, 'study-context');
    [['Aim', study.aim], ['How we tested it', study.protocol]].forEach(([title, text]) => {
      const block = element('div'); block.append(element('h4', title), element('p', text)); context.append(block);
    });
    article.append(context);
    const finding = element('div', undefined, 'study-finding'); finding.append(element('h4', 'Results'), element('p', study.finding)); article.append(finding);
    const figure = element('figure'); const img = element('img');
    if (!/^plots\/[a-z-]+\.svg$/.test(study.plot)) throw new Error('Invalid study plot path');
    img.src = study.plot; img.alt = study.plot_alt; img.loading = 'lazy'; img.decoding = 'async';
    img.width = Math.round(study.plot_width); img.height = Math.round(study.plot_height);
    img.style.aspectRatio = `${study.plot_width} / ${study.plot_height}`;
    figure.append(img, element('figcaption', study.caption)); article.append(figure);
    article.append(studyTable(study.headers, study.rows, 'Measured results'));
    if (study.review_rows) {
      const breakdown = element('div', undefined, 'rubric-breakdown');
      breakdown.append(element('h4', 'LLM rubric breakdown'), studyTable(study.review_headers, study.review_rows, 'Mean rubric ratings across six outputs per generator'));
      article.append(breakdown);
    }
    article.append(element('p', study.limits, 'study-limit'));
    const source = element('a', 'Aggregate evidence ↗', 'small'); source.href = 'https://github.com/Deb044/SemEval2025-Task10/blob/main/' + study.source; article.append(source);
    root.append(article);
  });
  const remaining = experiments.filter(row => !featuredResultIds.has(row.id));
  byId('appendix-summary').textContent = `Browse ${remaining.length} additional result entries`;
  const groups = new Map(); remaining.forEach(row => groups.set(row.study, (groups.get(row.study) || 0) + 1));
  const overview = byId('appendix-overview'); overview.replaceChildren();
  groups.forEach((count, study) => {
    const button = element('button', `${study} · ${count}`); button.type = 'button';
    button.addEventListener('click', () => { byId('appendix-catalog').open = true; byId('study-filter').value = study; byId('kind-filter').value = 'all'; byId('experiment-search').value = ''; renderExperiments(); });
    overview.append(button);
    const option = element('option', study); option.value = study; byId('study-filter').append(option);
  });
}
Promise.all(['experiments.json', 'studies.json'].map(async path => {
  const response = await fetch(path, {cache: 'no-store'});
  if (!response.ok) throw new Error('Study evidence unavailable');
  return response.json();
})).then(([catalog, studies]) => {
  experiments = catalog.experiments || [];
  if (studies.studies?.length !== 5) throw new Error('Expected five complete featured studies');
  renderStudies(studies); renderExperiments();
}).catch(error => {
  byId('featured-studies').replaceChildren(element('p', error.message + '. The repository contains the aggregate evidence.'));
  byId('experiment-count').textContent = 'Results unavailable';
});

let localModels = [];
let reviewItems = [];
async function api(path, payload) {
  const response = await fetch(path, payload === undefined ? { cache: 'no-store' } : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Local request failed');
  return data;
}
function updateLanguages() {
  const languages = localModels.find(model => model.id === byId('analysis-model').value)?.languages || [];
  [...byId('analysis-language').options].forEach(option => option.disabled = !languages.includes(option.value));
  if (!languages.includes(byId('analysis-language').value)) byId('analysis-language').value = languages[0] || 'en';
}
byId('analysis-model').addEventListener('change', updateLanguages);
api('/api/status').then(data => {
  if (!data.local) throw new Error('Local workspace not detected');
  localModels = data.models;
  localModels.forEach(model => {
    const option = element('option', model.label + (model.available ? '' : ' — reproduce first'));
    option.value = model.id; option.disabled = !model.available; byId('analysis-model').append(option);
  });
  const first = localModels.find(model => model.available);
  if (first) byId('analysis-model').value = first.id;
  updateLanguages();
  byId('analyze-button').disabled = !first;
  byId('generate-explanation').disabled = !data.generator;
  byId('load-reviews').disabled = !data.review_available;
  byId('local-instructions').hidden = true;
  byId('service-status').textContent = 'Local workspace connected. ' + (data.generator ? 'Explanation generator ready.' : 'Start the local generator to enable explanations.');
}).catch(() => {
  byId('service-status').textContent = 'Analysis and review are available in the local workspace.';
});
byId('analysis-form').addEventListener('submit', async event => {
  event.preventDefault();
  const button = byId('analyze-button'); button.disabled = true;
  const root = byId('analysis-output'); root.replaceChildren(element('p', 'Analyzing text…'));
  try {
    const data = await api('/api/analyze', {model: byId('analysis-model').value, language: byId('analysis-language').value,
      text: byId('analysis-text').value, generate: byId('generate-explanation').checked});
    const panel = element('article', undefined, 'panel');
    panel.append(element('h3', 'Predicted labels'), element('p', data.model), element('p', data.note, 'small'));
    const labels = element('ul', undefined, 'metric-list');
    const descriptions = data.label_details?.map(label => `${label.code} — ${label.description}`) || data.labels;
    (descriptions.length ? descriptions : ['No technique selected']).forEach(label => labels.append(element('li', label)));
    panel.append(labels, element('p', data.parents.join(' · '), 'small'));
    if (data.explanation) {
      panel.append(element('h3', data.explanation.fallback ? 'Extracted fallback' : 'Generated explanation'), element('p', data.explanation.text));
      panel.append(element('h3', 'Selected evidence'));
      const evidence = element('ol', undefined, 'evidence');
      data.explanation.evidence.forEach(unit => evidence.append(element('li', `[${unit.id}] ${unit.text}`)));
      panel.append(evidence, element('p', 'Review the explanation against the source before relying on it.', 'small'));
    }
    if (data.explanation_error) panel.append(element('p', data.explanation_error));
    const scores = element('details'); scores.append(element('summary', 'Inspect label scores'));
    scores.append(element('pre', data.scores.map(score => `${score.label}: ${score.value.toFixed(4)}`).join('\n')));
    panel.append(scores);
    if (data.trace) {
      const flow = element('details'); flow.append(element('summary', 'Inspect the trained pipeline trace'));
      data.trace.forEach(stage => flow.append(element('p', `${stage.stage} (${stage.state}): ${stage.detail}`, 'small')));
      panel.append(flow);
    }
    root.replaceChildren(panel);
  } catch (error) { root.replaceChildren(element('p', error.message)); }
  finally { button.disabled = false; }
});

const criteria = {
  supported: ['Source support', '0: unsupported · 1: partly supported · 2: fully supported'],
  attribution: ['Attribution', '0: lost/distorted · 1: partly preserved · 2: preserved'],
  relevance: ['Target relevance', '0: unrelated · 1: partly relevant · 2: relevant'],
  readability: ['Readability', '0: unclear · 1: partly clear · 2: clear']
};
Object.entries(criteria).forEach(([key, [title, rubric]]) => {
  const label = element('label', title); const select = element('select'); select.id = 'rating-' + key; select.required = true;
  const blank = element('option', 'Choose a score'); blank.value = ''; select.append(blank);
  [0, 1, 2].forEach(value => { const option = element('option', String(value)); option.value = value; select.append(option); });
  label.append(select, element('span', rubric, 'small')); byId('rating-fields').append(label);
});
async function loadReviews() {
  const reviewer = byId('reviewer').value.trim();
  if (!reviewer) { byId('review-status').textContent = 'Enter your reviewer alias first.'; return; }
  const data = await api('/api/reviews?reviewer=' + encodeURIComponent(reviewer));
  reviewItems = data.items;
  byId('review-status').textContent = `${data.completed || 0} of ${data.total || 0} items reviewed under this alias.`;
  byId('review-form').hidden = !reviewItems.length;
  if (!reviewItems.length) return;
  const row = reviewItems[0];
  byId('review-form').reset();
  byId('review-source').textContent = row.text;
  byId('review-explanation').textContent = row.explanation;
  byId('review-target').textContent = `Target: ${row.target_description} · Language: ${row.language.toUpperCase()}`;
}
byId('load-reviews').addEventListener('click', () => loadReviews().catch(error => byId('review-status').textContent = error.message));
byId('review-form').addEventListener('submit', async event => {
  event.preventDefault();
  const payload = {review_id: reviewItems[0].review_id, reviewer: byId('reviewer').value.trim(),
    human_reviewer: byId('review-human').checked, language_confident: byId('review-human').checked,
    comments: byId('review-comments').value};
  Object.keys(criteria).forEach(key => payload[key] = Number(byId('rating-' + key).value));
  try { await api('/api/review', payload); await loadReviews(); }
  catch (error) { byId('review-status').textContent = error.message; }
});
