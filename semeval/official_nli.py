"""Pinned multilingual ONNX NLI filtering for the official-vocabulary prototype."""
from functools import lru_cache
from pathlib import Path
import json
import numpy as np

MODEL = 'onnx-community/multilingual-MiniLMv2-L6-mnli-xnli-ONNX'
REVISION = 'ca5daf3d11b6c4b3143b1f4602a2edfb64c3ad7e'
CHECKSUM = '55614f3c7da74184742eaa0006b978744437aa91de9ba4913db42f94d7844a8f'
CACHE = '.cache/official-nli'


def download():
    from huggingface_hub import snapshot_download
    from experiments.common import sha256
    path = Path(CACHE) / 'onnx/model_quantized.onnx'
    if not path.exists():
        snapshot_download(MODEL, revision=REVISION, local_dir=CACHE, max_workers=2,
            allow_patterns=['config.json', 'tokenizer*.json', 'special_tokens_map.json', 'sentencepiece.bpe.model', 'onnx/model_quantized.onnx'])
    if sha256(path) != CHECKSUM:
        raise ValueError('Official-mode NLI model checksum mismatch')
    return path


@lru_cache(maxsize=1)
def runtime():
    import onnxruntime as ort
    from transformers import AutoTokenizer
    path = download()
    config = json.loads((Path(CACHE) / 'config.json').read_text(encoding='utf-8'))
    if config['label2id']['entailment'] != 0:
        raise ValueError('Unexpected NLI label order')
    settings = ort.SessionOptions(); settings.intra_op_num_threads = 2
    model = ort.InferenceSession(str(path), sess_options=settings, providers=['CPUExecutionProvider'])
    return AutoTokenizer.from_pretrained(CACHE), model


def filter_supported(data, ranking, supports):
    labels = {row['id']: row for row in data['labels']}
    checked = [dict(row, entailment_score=score['entailment'] if score else None,
        support_score=score['entailment'] / max(score['entailment'] + score['contradiction'], 1e-12) if score else None)
        for row, score in zip(ranking, supports, strict=True)]
    eligible = [row for row in checked if row['support_score'] is not None and row['support_score'] >= .7 and row['entailment_score'] >= .25 and row['score'] >= .3]
    concrete = {row['parent'] for row in eligible if not labels[row['label']]['is_other']}
    selected = sorted([row for row in eligible if not labels[row['label']]['is_other'] or row['parent'] not in concrete],
                      key=lambda row: (-row['support_score'], -row['score']))
    selected = [row for row in selected if row['support_score'] >= selected[0]['support_score'] - .05][:3]
    return selected or [{'label': 'Other', 'parent': 'Other', 'score': None, 'support_score': None, 'evidence_id': None}], sorted(checked, key=lambda row: -(row['support_score'] if row['support_score'] is not None else -1))


def rerank(data, chunks, ranking):
    tokenizer, model = runtime()
    names = {row['id']: row for row in data['labels']}
    supports = []
    for row in ranking:
        if row['score'] < .3:
            supports.append(None); continue
        center = row['evidence_id']
        premise = ' '.join(unit['text'] for unit in chunks[max(0, center - 1):center + 2])
        encoded = tokenizer(premise, names[row['label']]['hypothesis'], return_tensors='np', truncation='only_first', max_length=384)
        feeds = {item.name: np.asarray(encoded[item.name], dtype=np.int64) for item in model.get_inputs()}
        logits = model.run(None, feeds)[0][0]
        probabilities = np.exp(logits - logits.max()); probabilities /= probabilities.sum()
        supports.append({'entailment': float(probabilities[0]), 'contradiction': float(probabilities[2])})
    return filter_supported(data, ranking, supports)
