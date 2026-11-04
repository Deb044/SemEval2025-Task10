"""Official-vocabulary zero-shot matching, separate from auxiliary trained GNNs."""
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

TAXONOMY = Path(__file__).parent / 'taxonomies/official.json'
LANGUAGES = ('en', 'bg', 'hi', 'pt', 'ru')
CUTOFF = .40  # Exploratory semantic-similarity cutoff, not a calibrated probability.


@lru_cache(maxsize=1)
def taxonomy():
    data = json.loads(TAXONOMY.read_text(encoding='utf-8'))
    ids = {row['id'] for row in data['labels']}
    parents = {row['id'] for row in data['parents']}
    if len(ids) != 96 or len(parents) != 22 or any(row['parent'] not in parents for row in data['labels']):
        raise ValueError('Invalid official taxonomy inventory')
    return data


def units(text):
    result = []
    for sentence in re.finditer(r'[^.!?।\r\n]+(?:[.!?।]+|$)', text):
        matches = list(re.finditer(r'\S+', sentence.group()))
        for offset in range(0, len(matches), 32):
            group = matches[offset:offset + 32]
            if not group:
                continue
            start, end = sentence.start() + group[0].start(), sentence.start() + group[-1].end()
            result.append({'id': len(result), 'text': text[start:end],
                'start': len(text[:start].encode('utf-16-le')) // 2, 'end': len(text[:end].encode('utf-16-le')) // 2})
    if not result:
        raise ValueError('Enter article text containing words')
    if len(result) > 200:
        raise ValueError('Article is too long for this interactive prototype')
    return result


def query(row):
    prefix = 'Climate change. ' if row['domain'] == 'CC' else 'Ukraine-Russia war. '
    parent = row['parent'].split(': ', 1)[-1]
    return prefix + parent + ('' if row['is_other'] else '. ' + row['definition'])


def select_matches(data, vectors, article_vectors, domain='CC', cutoff=CUTOFF):
    pooled = article_vectors.mean(axis=0)
    pooled /= max(float(np.linalg.norm(pooled)), 1e-12)
    similarity = article_vectors @ vectors.T
    values = .65 * similarity.max(axis=0) + .35 * (vectors @ pooled)
    scored = []
    for index, row in enumerate(data['labels']):
        if row['id'] == 'Other' or (domain != 'all' and row['domain'] != domain):
            continue
        score = float(values[index]) * (.9 if row['is_other'] else 1)
        scored.append({'label': row['id'], 'parent': row['parent'], 'score': score,
            'evidence_id': int(similarity[:, index].argmax()), 'is_other': row['is_other']})
    concrete = {row['parent'] for row in scored if not row['is_other'] and row['score'] >= cutoff}
    order = sorted([row for row in scored if not row['is_other'] or row['parent'] not in concrete],
                   key=lambda row: (-row['score'], row['label']))
    candidates = [row for row in order if row['score'] >= cutoff][:3]
    if not candidates:
        candidates = [{'label': 'Other', 'parent': 'Other', 'score': None, 'evidence_id': None}]
    return candidates, sorted(scored, key=lambda row: (-row['score'], row['label']))[:8]


def extract_evidence(chunks, matches):
    selected = set()
    for match in matches:
        if match['evidence_id'] is not None:
            center = match['evidence_id']
            for index in [center, center - 1, center + 1]:
                if 0 <= index < len(chunks):
                    selected.add(index)
    evidence = [chunks[i] for i in sorted(selected)]
    preview, count = [], 0
    for row in evidence:
        size = len(row['text'].split())
        if count + size <= 80:
            preview.append(row['text']); count += size
    return evidence, ' '.join(preview)


def analyze(payload):
    from .features import get_encoder
    text = str(payload.get('text', '')).strip()
    language, domain = payload.get('language', 'en'), payload.get('domain', 'CC')
    if not 0 < len(text) <= 20000 or language not in LANGUAGES or domain not in ('CC', 'URW', 'all'):
        raise ValueError('Use 1–20,000 characters, a supported language and a taxonomy domain')
    max_labels = payload.get('max_labels', 1)
    if type(max_labels) is not int or max_labels not in (1, 3):
        raise ValueError('Choose one or up to three candidate labels')
    data, chunks = taxonomy(), units(text)
    encoder = get_encoder('minilm')
    label_vectors = encoder.encode([query(row) for row in data['labels']])
    article_vectors = encoder.encode([row['text'] for row in chunks])
    matches, ranking = select_matches(data, label_vectors, article_vectors, domain)
    from .official_nli import rerank
    matches, ranking = rerank(data, chunks, ranking)
    matches = matches[:max_labels]
    evidence, preview = extract_evidence(chunks, matches)
    allowed = {row['id'] for row in data['labels']}
    if any(row['label'] not in allowed for row in matches):
        raise ValueError('Output escaped the official label inventory')
    trace = [{'stage': 'Input', 'state': 'complete', 'detail': f'{len(text)} characters; {len(chunks)} text units.'},
        {'stage': 'Multilingual encoder', 'state': 'complete', 'detail': f'{article_vectors.shape[1]}-dimensional normalized embeddings for text and label names.'},
        {'stage': 'Candidate ranking', 'state': 'complete', 'detail': 'Cosine-based retrieval of up to eight official-label candidates.'},
        {'stage': 'Entailment check', 'state': 'complete', 'detail': 'Pinned multilingual MiniLMv2 NLI checks claims against nearby source context; semantic similarity has 80% weight in the combined rank; agreement ≥0.60 and three-way entailment ≥0.10 are exploratory cutoffs.'},
        {'stage': 'Hierarchy validation', 'state': 'complete', 'detail': 'Every label and parent is checked against the pinned 96-label public organizer taxonomy.'},
        {'stage': 'Label GNN', 'state': 'not_used', 'detail': 'The trained GNN checkpoints use auxiliary labels; they are not an official-taxonomy classifier.'},
        {'stage': 'Evidence', 'state': 'complete', 'detail': f'{len(evidence)} source units selected with neighboring context.'}]
    result = {'matches': matches, 'ranking': ranking, 'evidence': evidence, 'preview': preview, 'trace': trace,
        'units': chunks, 'taxonomy_version': data['version'], 'source': data['source'], 'engine': 'Local MiniLM retrieval + multilingual NLI · zero-shot',
        'note': 'Official-vocabulary candidates, not validated official-task predictions. Similarity is not a truth probability.', 'generated_explanation': None}
    if payload.get('generate') and matches[0]['label'] != 'Other':
        from .explain import explain_row
        mapping = {row['id']: {'parent': row['parent'], 'definition': row['definition']} for row in data['labels']}
        prediction = {'labels': [row['label'] for row in matches], 'scores': {row['label']: row['selection_score'] for row in matches}}
        try:
            value = explain_row({'article_id': 'local-official-input', 'language': language, 'text': text},
                'generate', mapping, 'http://127.0.0.1:8080/v1', predicted=prediction, encoder=encoder)
            result['generated_explanation'] = {'text': value['explanation'], 'fallback': value['fallback'], 'target': value['dominant_subnarrative']}
            result['trace'].append({'stage': 'Explanation', 'state': 'complete', 'detail': 'Qwen requested; invalid output uses an explicitly marked extractive fallback.'})
        except (OSError, TimeoutError, ValueError):
            result['explanation_error'] = 'The local generator did not return a usable explanation. Candidate labels and evidence are preserved.'
            result['trace'].append({'stage': 'Explanation', 'state': 'unavailable', 'detail': 'Local Qwen generation was unavailable or invalid; inspect the source extract.'})
    else:
        result['trace'].append({'stage': 'Explanation', 'state': 'extractive', 'detail': 'Source-evidence extract of at most 80 words; not LLM generation.'})
    return result
