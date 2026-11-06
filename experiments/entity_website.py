"""Publish silver agreement with explicit provenance, never official accuracy."""
import json
from pathlib import Path


def append_catalog(payload):
    path=Path('results/entity-framing.json')
    if not path.exists():return
    data=json.loads(path.read_text(encoding='utf-8'))
    for name,report in data.get('silver_agreement',{}).items():
        payload['experiments'].append(dict(id='entity-silver-'+name,study='Subtask 1 silver-label diagnostic',kind='classification',
            dataset='Public excerpt teacher-labelled entity pilot',name='Entity-role GNN · '+name,
            role='Silver agreement diagnostic',metric='fine-role exact match (silver agreement)',
            value=report['metrics']['exact_match'],unit='fraction',n=report['metrics']['count'],status='measured',
            notes='Agreement with the same named LLM teacher; not independently verified role accuracy. Sparse role coverage and selection bias.',
            details=dict(teacher=data['teacher']['name'],accepted=data['accepted'],language_counts=data['language_counts'],
                fine_role_counts=data['label_counts'],training_fine_role_counts=data['checkpoints'][name]['training_label_counts'],main_role_agreement=report['metrics']['main_role_accuracy']),
            provenance={key:data[key] for key in ('commit','input_sha256','source_sha256')}))
