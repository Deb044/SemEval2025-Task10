"""Entity-specific representations and the official 22-role hierarchy.

No sentiment-to-role mapping. Mentions and offsets are supplied, not inferred.
The public prototype may abstain; official prediction/export requires fine roles.
"""
import copy
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from .data import ROLE_GROUPS,ROLE_PARENT,check_splits,validate,read_jsonl,write_json

ROLE_SOURCE='https://propaganda.math.unipd.it/semeval2025task10/ENTITY-ROLE-TAXONOMY.pdf'
ROLE_SHA256='c55d1692f020cbd5f13135d6edfc76bbb90efe8fe6190a396051e4be47e21f59'
DEFINITIONS={
 'Guardian':'protecting people, communities or shared values',
 'Martyr':'sacrificing personal wellbeing for a greater cause',
 'Peacemaker':'resolving conflict through negotiation or reconciliation',
 'Rebel':'challenging oppression or established power for positive change',
 'Underdog':'struggling courageously against more powerful opponents',
 'Virtuous':'acting honestly, fairly and with moral integrity',
 'Instigator':'provoking conflict, unrest or violence',
 'Conspirator':'participating in secret harmful plots',
 'Tyrant':'abusing authority to oppress others',
 'Foreign Adversary':'posing an external geopolitical threat',
 'Traitor':'betraying a country, group or shared cause',
 'Spy':'secretly obtaining information for a rival',
 'Saboteur':'deliberately disrupting or damaging systems',
 'Corrupt':'abusing entrusted power for personal gain',
 'Incompetent':'causing failures through lack of competence',
 'Terrorist':'using terror or extremist violence',
 'Deceiver':'deliberately misleading people or manipulating information',
 'Bigot':'expressing discriminatory hostility toward a social group',
 'Forgotten':'being neglected or overlooked despite needing support',
 'Exploited':'being used unfairly for someone else’s benefit',
 'Victim':'suffering physical or economic harm beyond personal control',
 'Scapegoat':'being blamed unfairly for others’ failures'}
LABELS=sorted(ROLE_PARENT);PARENTS=sorted(ROLE_GROUPS)
HYPOTHESES={'Guardian': 'The target protects and saves people.', 'Martyr': 'The target sacrifices their own life for others.', 'Peacemaker': 'The target negotiates peace.', 'Rebel': 'The target resists oppression.', 'Underdog': 'The target struggles against a stronger opponent.', 'Virtuous': 'The target is honest and fair.', 'Instigator': 'The target provokes violence.', 'Conspirator': 'The target participates in a secret harmful plot.', 'Tyrant': 'The target rules through oppression.', 'Foreign Adversary': 'The target is a hostile foreign power.', 'Traitor': 'The target betrays their own country.', 'Spy': 'The target secretly gathers information for an enemy.', 'Saboteur': 'The target deliberately damages important systems.', 'Corrupt': 'The target steals money or accepts bribes.', 'Incompetent': 'The target is unable to do their job properly.', 'Terrorist': 'The target terrorizes civilians.', 'Deceiver': 'The target deliberately lies to people.', 'Bigot': 'The target discriminates against a social group.', 'Forgotten': 'The target is neglected and denied needed support.', 'Exploited': 'The target is used unfairly for another person’s benefit.', 'Victim': 'The target is injured or economically harmed.', 'Scapegoat': 'The target is blamed unfairly.'}


def taxonomy():
    return {role:dict(parent=ROLE_PARENT[role],definition=DEFINITIONS[role]) for role in LABELS}


def utf16_index(text,index):
    if type(index) is not int or index<0:raise ValueError('Offsets must be nonnegative integers')
    used=0
    for position,char in enumerate(text):
        if used==index:return position
        used+=len(char.encode('utf-16-le'))//2
        if used>index:raise ValueError('Offset splits a Unicode surrogate pair')
    if used==index:return len(text)
    raise ValueError('Offset exceeds article length')


def mentions_from_payload(payload):
    text=payload.get('text');language=payload.get('language','en')
    if not isinstance(text,str) or not text.strip() or len(text)>50000 or language not in ('en','bg','hi','pt','ru'):
        raise ValueError('Provide 1–50,000 characters and a supported task language')
    unit=payload.get('offset_unit','codepoint');ending=payload.get('offset_end','exclusive')
    if unit not in ('codepoint','utf16') or ending not in ('exclusive','inclusive'):raise ValueError('Select explicit offset units and end convention')
    values=payload.get('mentions')
    if not isinstance(values,list) or not 1<=len(values)<=30:raise ValueError('Supply 1–30 target mentions')
    article_id=payload.get('article_id','user-input.txt')
    if not isinstance(article_id,str) or not article_id or len(article_id)>120 or any(c in article_id for c in '\t\r\n'):raise ValueError('Use a short article ID without tabs or newlines')
    rows=[]
    for target in values:
        if not isinstance(target,dict):raise ValueError('Each mention must be an object')
        if not isinstance(target.get('mention'),str) or not 0<len(target['mention'])<=200:raise ValueError('Supply an exact named-entity string of 1–200 characters')
        start,end=target.get('start'),target.get('end')
        if type(start) is not int or type(end) is not int:raise ValueError('Mention offsets must be integers')
        exclusive=end+(ending=='inclusive')
        a,b=(utf16_index(text,start),utf16_index(text,exclusive)) if unit=='utf16' else (start,exclusive)
        row=dict(task=1,article_id=payload.get('article_id','user-input.txt'),language=language,text=text,
            mention=target.get('mention'),start=a,end=b,end_exclusive=b)
        # Validate the exact supplied string; never find another occurrence silently.
        validate([row]);rows.append(row)
    validate(rows)
    return rows


def context(row,radius=800):
    validate([row]);text=row['text'];start,end=row['start'],row['end_exclusive']
    a=max(0,start-radius);b=min(len(text),end+radius)
    # Keep the exact target even when a surrounding sentence is unusually long.
    previous=list(re.finditer(r'[.!?।\n]',text[a:start]))
    if previous:a+=previous[-2].end() if len(previous)>1 else 0
    following=list(re.finditer(r'[.!?।\n]',text[end:b]))
    if following:b=end+following[min(1,len(following)-1)].end()
    marked=text[a:start]+' [TARGET] '+text[start:end]+' [/TARGET] '+text[end:b]
    return dict(text=text[a:b],marked=marked,start=a,end=b,target_start=start-a,target_end=end-a)


def target_sentence(row):
    text=row['text'];start,end=row['start'],row['end_exclusive']
    before=list(re.finditer(r'[.!?।\n]',text[:start]));after=re.search(r'[.!?।\n]',text[end:])
    a=before[-1].end() if before else 0;b=end+after.end() if after else len(text)
    # Normalize only the supplied occurrence; other entities retain their names.
    a=max(a,start-250);b=min(b,end+250)
    return dict(text=text[a:b],premise=text[a:start]+'The target'+text[end:b],start=a,end=b)


def representations(rows,encoder,ablation='full'):
    if ablation not in ('full','unmarked'):raise ValueError('Unknown entity representation')
    if ablation=='unmarked':return encoder.encode([context(row)['text'] for row in rows])
    entity=encoder.encode(['Target entity: '+row['mention'] for row in rows])
    local=encoder.encode(['Target entity: '+row['mention']+'. Context: '+context(row)['marked'] for row in rows])
    global_=encoder.encode([row['text'] for row in rows])
    return np.concatenate([entity,local,global_],axis=1).astype('float32')


def decode(fine,parent,threshold=.25,allow_abstain=False):
    outputs=[]
    parent_indices=np.array([PARENTS.index(ROLE_PARENT[label]) for label in LABELS])
    for values,broad in zip(fine,parent,strict=True):
        chosen_parent=PARENTS[int(np.argmax(broad))]
        scores=values*broad[parent_indices]
        eligible=[i for i,label in enumerate(LABELS) if ROLE_PARENT[label]==chosen_parent]
        selected=[LABELS[i] for i in eligible if scores[i]>=threshold]
        best=max(eligible,key=lambda i:scores[i])
        abstained=allow_abstain and scores[best]<threshold
        if not selected and not abstained:selected=[LABELS[best]]
        outputs.append(dict(main_role=None if abstained else chosen_parent,labels=[] if abstained else selected,
            scores=dict(zip(LABELS,map(float,scores))),confidence=float(scores[best]),abstained=abstained))
    return outputs


def train(training,development,encoder_name='minilm',seed=42,ablation='full'):
    import torch
    from .features import get_encoder
    from .graph import LabelGCN,label_adjacency
    torch.set_num_threads(4);torch.manual_seed(seed)
    check_splits(training,development)
    for row in training+development:
        if row['task']!=1 or not row.get('labels') or row.get('label_schema')=='native':
            raise ValueError('Entity GNN needs official-role annotations, not sentiment or native auxiliary labels')
    encoder=get_encoder(encoder_name)
    x,xd=representations(training,encoder,ablation),representations(development,encoder,ablation)
    nodes,adjacency=label_adjacency(LABELS,taxonomy(),seed)
    initial=encoder.encode([DEFINITIONS[label]+' '+label if kind=='fine' else label for kind,label in nodes])
    head=LabelGCN(x.shape[1],torch.tensor(initial),torch.tensor(adjacency),torch.arange(25))
    def targets(rows):
        return torch.tensor([[label in row['labels'] for label in LABELS] for row in rows],dtype=torch.float32),torch.tensor([PARENTS.index(row['main_role']) for row in rows])
    y,yp=targets(training);yd,ydp=targets(development)
    x,xd=torch.tensor(x),torch.tensor(xd)
    weights=((len(y)-y.sum(0))/(y.sum(0)+1)).clamp(1,5)
    criterion=torch.nn.BCEWithLogitsLoss(pos_weight=weights);parent_loss=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(head.parameters(),lr=.001)
    best,stale,state=float('inf'),0,None
    for epoch in range(100):
        head.train()
        for indices in torch.randperm(len(x)).split(16):
            logits=head(x[indices]);loss=criterion(logits[:,:22],y[indices])+parent_loss(logits[:,22:],yp[indices])
            optimizer.zero_grad();loss.backward();optimizer.step()
        head.eval()
        with torch.inference_mode():
            logits=head(xd);loss=float(criterion(logits[:,:22],yd)+parent_loss(logits[:,22:],ydp))
        if loss<best:best,stale,state=loss,0,copy.deepcopy(head.state_dict())
        else:stale+=1
        if stale>=12:break
    head.load_state_dict(state);head.eval()
    with torch.inference_mode():
        logits=head(xd);fine=torch.sigmoid(logits[:,:22]).numpy();parent=torch.softmax(logits[:,22:],dim=1).numpy()
    selected=(float('-inf'),None)
    for threshold in (.05,.1,.15,.2,.25,.3,.4,.5):
        predicted=decode(fine,parent,threshold)
        score=sum(set(p['labels'])==set(row['labels']) for row,p in zip(development,predicted))/len(development)
        if score>selected[0]:selected=score,threshold
    provenance=sorted({row.get('annotation_source','unspecified supplied annotations') for row in training})
    return dict(kind='entity-role-gcn-v1',head=head,encoder=encoder_name,ablation=ablation,threshold=selected[1],
        seed=seed,development_exact_match=selected[0],annotation_sources=provenance,
        training_label_counts={label:int(y[:,i].sum()) for i,label in enumerate(LABELS)},
        encoder_model=getattr(encoder,'model_name','test encoder'),encoder_revision=getattr(encoder,'revision','test fixture'),
        development_signature=hashlib.sha256(json.dumps(development,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        training_signature=hashlib.sha256(json.dumps(training,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
        taxonomy_source=ROLE_SOURCE,taxonomy_sha256=ROLE_SHA256,epochs=epoch+1)


def predict(model,rows):
    import torch
    from .features import get_encoder
    if model.get('kind')!='entity-role-gcn-v1' or model.get('taxonomy_sha256')!=ROLE_SHA256:raise ValueError('Unexpected entity checkpoint')
    if any(row.get('task')!=1 for row in rows):raise ValueError('Entity predictions require Subtask 1 mention records')
    torch.set_num_threads(4)
    validate(rows);encoder=get_encoder(model['encoder']);x=representations(rows,encoder,model['ablation'])
    model['head'].eval()
    with torch.inference_mode():
        logits=model['head'](torch.tensor(x));fine=torch.sigmoid(logits[:,:22]).numpy();parent=torch.softmax(logits[:,22:],dim=1).numpy()
    decoded=decode(fine,parent,model['threshold'])
    return [dict(task=1,**{key:row[key] for key in ('article_id','language','mention','start','end')},**prediction,
        annotation_sources=model['annotation_sources']) for row,prediction in zip(rows,decoded,strict=True)]


def prototype(rows):
    from .official_nli import runtime
    tokenizer,model=runtime();results=[]
    for row in rows:
        snippet=target_sentence(row)
        premise=snippet['premise']
        fine=[]
        for role in LABELS:
            hypothesis=HYPOTHESES[role]
            encoded=tokenizer(premise,hypothesis,return_tensors='np',truncation='only_first',max_length=384)
            feeds={item.name:np.asarray(encoded[item.name],dtype=np.int64) for item in model.get_inputs()}
            logits=model.run(None,feeds)[0][0];p=np.exp(logits-logits.max());p/=p.sum()
            fine.append(float(p[0]))
        fine=np.array(fine)
        broad=np.array([max(fine[i] for i,role in enumerate(LABELS) if ROLE_PARENT[role]==p) for p in PARENTS])
        parent=PARENTS[int(broad.argmax())]
        eligible=[i for i,role in enumerate(LABELS) if ROLE_PARENT[role]==parent]
        strongest=max(float(fine[i]) for i in eligible)
        selected=[LABELS[i] for i in eligible if fine[i]>=max(.25,strongest*.85)]
        output=dict(main_role=parent if selected else None,labels=selected,abstained=not selected,confidence=strongest,
            ranking=sorted([dict(label=role,parent=ROLE_PARENT[role],score=float(fine[i])) for i,role in enumerate(LABELS)],key=lambda r:-r['score']))
        results.append(dict(task=1,**{key:row[key] for key in ('article_id','language','mention','start','end')},**output,
            evidence=snippet,engine='Multilingual NLI zero-shot; no GNN used',
            note='Source portrayal, not a factual judgment about the entity. Exploratory cutoff; official accuracy unverified.'))
    return results


def evaluate(gold,predictions):
    from .evaluate import align
    from .metrics import classification_scores
    validate(gold);predictions=align(gold,predictions)
    for row,p in zip(gold,predictions,strict=True):
        if not p['labels'] or any(label not in ROLE_PARENT or ROLE_PARENT[label]!=p['main_role'] for label in p['labels']):raise ValueError('Invalid role prediction or abstention in official evaluation')
        if any(p.get(k)!=row[k] for k in ('mention','start','end')):raise ValueError('Mention identity differs')
    result=classification_scores(gold,[p['labels'] for p in predictions],classes=LABELS)
    result['main_role_accuracy']=sum(row['main_role']==p['main_role'] for row,p in zip(gold,predictions))/len(gold)
    return dict(primary_metric='exact_match',metrics=result,annotation_sources=sorted({r.get('annotation_source','unspecified supplied labels') for r in gold}),
        scoring_provenance='Local fixed-22-role entity evaluation; organizer scorer parity not verified.',benchmark_claim=False)


def export_tsv(rows,path):
    from .evaluate import export_tsv as export
    for row in rows:
        if not row.get('labels') or row.get('abstained') or any(ROLE_PARENT.get(label)!=row.get('main_role') for label in row['labels']):
            raise ValueError('Official export needs nonempty, parent-compatible fine roles; abstentions cannot be exported')
    export(rows,path)
