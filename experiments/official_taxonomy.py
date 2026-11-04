"""Build a pinned public-guideline taxonomy; no private SemEval articles are used."""
import hashlib
import json
import urllib.request
from pathlib import Path

URL = 'https://propaganda.math.unipd.it/semeval2025task10/NARRATIVE-TAXONOMIES.pdf'
PDF_SHA256 = 'e9b26b6e4a58bb28900572ea140adf2da9e7c4858aa8a26f9039774812d8a26e'


def main():
    from pypdf import PdfReader
    path = Path('data/public/official/NARRATIVE-TAXONOMIES.pdf')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(urllib.request.urlopen(URL, timeout=45).read())
    if hashlib.sha256(path.read_bytes()).hexdigest() != PDF_SHA256:
        raise ValueError('Organizer PDF changed; verify the inventory before updating its checksum')
    parents, labels = [], []
    for page, domain in zip(PdfReader(path).pages[:2], ['URW', 'CC']):
        parent = None
        for raw in page.extract_text().splitlines():
            line = raw.strip()
            if line.startswith('Figure '):
                break
            if not line or line == 'Other':
                continue
            if '\ufffd' in line:
                raise ValueError('Ambiguous PDF character; inspect the source rather than guessing')
            if line.startswith('- '):
                name = line[2:]
                if parent is None:
                    raise ValueError('Subnarrative without a parent')
                labels.append({'id': parent + ': ' + name, 'parent': parent, 'name': name,
                    'domain': domain, 'definition': name, 'is_other': False})
            else:
                parent = domain + ': ' + line
                parents.append({'id': parent, 'name': line, 'domain': domain})
                # The task page specifies a parent-specific Other for each narrative.
                labels.append({'id': parent + ': Other', 'parent': parent, 'name': 'Other',
                    'domain': domain, 'definition': line + ': no more specific subnarrative.', 'is_other': True})
    parents.append({'id': 'Other', 'name': 'Other', 'domain': 'ALL'})
    labels.append({'id': 'Other', 'parent': 'Other', 'name': 'Other', 'domain': 'ALL',
        'definition': 'No suitable narrative category identified.', 'is_other': True})
    assert len(parents) == 22 and len(labels) == 96
    assert len({row['id'] for row in labels}) == 96
    payload = {'profile': 'SemEval-2025 Task 10 public organizer taxonomy', 'version': 'pdf-' + PDF_SHA256[:12],
        'source': URL, 'source_sha256': PDF_SHA256, 'source_pages': [1, 2],
        'other_convention_source': 'https://propaganda.math.unipd.it/semeval2025task10/',
        'definition_source': 'Public label names; no private annotations or copied guideline examples.',
        'submission_scorer_parity': False, 'parents': parents, 'labels': labels,
        'counts': {'coarse': len(parents), 'fine': len(labels)},
        'browser_encoder': {'model': 'Xenova/paraphrase-multilingual-MiniLM-L12-v2',
            'revision': '2c4055b12046f11709e9df2c122e59ffbdc2f900',
            'weights_sha256': '66fc00f5f29afcaff34092e1bdd20008ca3918265a82fb9695a551e510cc4ebc',
            'library': '@xenova/transformers@2.17.2', 'quantized': True}}
    hypotheses = {
        'The West are the aggressors': 'Someone blames Western governments for starting the conflict.',
        'Ukraine is the aggressor': 'Someone blames Ukraine for starting the conflict.',
        'NATO should/will directly intervene': 'Someone argues that NATO should or will send its forces directly into the Ukraine-Russia war.',
        'NATO will destroy Russia': 'Someone predicts that NATO will destroy Russia.',
        'Climate policies have negative impact on the economy': 'Someone claims that climate policies harm the economy.',
        'Climate policies are ineffective': 'Someone claims that climate policies do not work.',
        'Renewable energy is costly': 'Someone claims that renewable energy costs too much.',
        'Renewable energy is dangerous': 'Someone claims that renewable energy is dangerous.',
        'Ad hominem attacks on key activists': 'Someone attacks climate activists personally.',
        'Blaming global elites': 'Someone blames global elites for a hidden climate agenda.'}
    for label in labels:
        name = label['parent'].split(': ', 1)[-1] if label['is_other'] else label['name']
        if name.startswith('Criticism of '):
            hypothesis = 'Someone criticizes ' + name[len('Criticism of '):] + ' in relation to climate change.'
        elif name.startswith('Discrediting '):
            hypothesis = 'Someone discredits ' + name[len('Discrediting '):] + '.'
        elif name.startswith('Praise of '):
            hypothesis = 'Someone praises ' + name[len('Praise of '):] + '.'
        else:
            hypothesis = 'Someone presents this claim: ' + name + '.'
        label['hypothesis'] = hypotheses.get(name, hypothesis)
    payload['hypothesis_source'] = 'Label-derived working NLI hypotheses; not organizer annotations or gold explanations.'
    payload['entailment_model'] = {
        'model': 'onnx-community/multilingual-MiniLMv2-L6-mnli-xnli-ONNX',
        'revision': 'ca5daf3d11b6c4b3143b1f4602a2edfb64c3ad7e',
        'weights_sha256': '55614f3c7da74184742eaa0006b978744437aa91de9ba4913db42f94d7844a8f',
        'entailment_index': 0, 'agreement_threshold': 0.7, 'minimum_entailment': 0.25, 'relative_agreement_margin': 0.05}
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + '\n'
    for destination in ['semeval/taxonomies/official.json', 'website/dist/official-taxonomy.json']:
        target = Path(destination); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(encoded, encoding='utf-8')
    print('Verified 22 coarse and 96 fine labels from the pinned organizer PDF and Other convention.')


if __name__ == '__main__':
    main()
