"""Native BG/RU persuasion, Hindi propaganda and Portugal-publisher pilots.

Pinned public sources. No translated labels, artificial hierarchy, or SemEval
benchmark claim. Original text and source identifiers remain ignored locally.
"""
import argparse
import csv
import hashlib
import json
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from semeval.data import check_splits, fingerprint, write_json, write_jsonl

ROOT = Path('data/public/multilingual')
SOURCES = Path('experiments/configs/multilingual-sources.json')
GROUPS = {
 'Attack on Reputation': ['Name_Calling-Labeling','Questioning_the_Reputation','Doubt','Guilt_by_Association','Appeal_to_Hypocrisy'],
 'Justification': ['Appeal_to_Authority','Appeal_to_Fear-Prejudice','Appeal_to_Popularity','Appeal_to_Values','Flag_Waving'],
 'Simplification': ['Causal_Oversimplification','Consequential_Oversimplification','False_Dilemma-No_Choice','False_Equivalence'],
 'Distraction': ['Red_Herring','Straw_Man','Whataboutism','Appeal_to_Pity'],
 'Call': ['Slogans','Conversation_Killer','Appeal_to_Time'],
 'Manipulative Wording': ['Loaded_Language','Exaggeration-Minimisation','Obfuscation-Vagueness-Confusion','Repetition']}
# Publisher-location proxy, not a claim of linguistic verification per headline.
PORTUGAL_OUTLETS = {'sapo.pt','noticiasaominuto.com','expresso.pt','publico.pt','observador.pt','sicnoticias.pt',
 'cmjornal.pt','jn.pt','iol.pt','dnoticias.pt','visao.pt','jornaldenegocios.pt','tsf.pt','acorianooriental.pt',
 'dn.pt','jornaleconomico.pt','diariodominho.pt','destak.pt'}
SPLITS = ('train','dev','test')


def download():
    for name, source in json.loads(SOURCES.read_text(encoding='utf-8')).items():
        path = ROOT/'raw'/name
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            request=urllib.request.Request(source['url'],headers={'User-Agent':'Academic research dataset preparation'})
            temporary=path.with_suffix(path.suffix+'.part')
            with urllib.request.urlopen(request,timeout=120) as response:
                temporary.write_bytes(response.read())
            temporary.replace(path)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=source['sha256']:
            raise ValueError('Source checksum changed: '+name)


def record(dataset, language, identity, text, labels, source_group):
    return dict(task=2,article_id=f'{language}_{dataset}_{identity}',language=language,text=text,
        labels=sorted(set(labels)),source_group=source_group,dataset=dataset,label_schema='native')


def slavic():
    rows=[]
    with zipfile.ZipFile(ROOT/'raw/slavic.zip') as archive:
        for language in ('BG','RU'):
            base=f'lrec2026-SLavicPT-corpus/lrec2026/{language}/'
            sentences={}
            for line in archive.read(base+'sentences.txt').decode('utf-8-sig').splitlines():
                fields=line.split('\t',3)
                if len(fields)!=4 or tuple(fields[:3]) in sentences:
                    raise ValueError('Missing or duplicate publisher sentence key')
                sentences[tuple(fields[:3])]=fields[3]
            for line in archive.read(base+'sentence-level-annotations-full.txt').decode('utf-8-sig').splitlines():
                fields=line.split('\t'); key=tuple(fields[:3])
                if key not in sentences: raise ValueError('Annotation lacks publisher sentence')
                rows.append(record('slavic',language.lower(),':'.join(key),sentences[key],fields[3:],language+':'+key[0]))
    taxonomy={label:dict(parent=parent,definition=label.replace('_',' ').replace('-',' / '),
        definition_source='publisher label names; parent groups from LREC2026 Figure 1') for parent,labels in GROUPS.items() for label in labels}
    if len(taxonomy)!=25: raise ValueError('Expected 25 native techniques')
    if set(label for row in rows for label in row['labels'])!=set(taxonomy):
        raise ValueError('Slavic label inventory differs from pinned hierarchy')
    return rows,taxonomy


def hindi():
    rows=[]
    for split in ('Train','Val','Test'):
        with open(ROOT/f'raw/hindi-{split}.csv',encoding='utf-8-sig',newline='') as stream:
            for i,raw in enumerate(csv.DictReader(stream)):
                label=raw['propaganda_label']
                if label not in ('-1','1'): raise ValueError('Unknown Hindi binary label')
                # The release itself misspells this column; do not guess a new URL.
                source=raw.get('rrticle_url') or raw.get('article_url') or f'{split}:{i}'
                rows.append(record('hpropnews','hi',f'{split}:{i}',raw['article_text'],[label],source))
    taxonomy={key:dict(parent=key,definition=name,definition_source='native binary label; identity parent is bookkeeping only')
        for key,name in [('-1','Non-propagandistic'),('1','Propagandistic')]}
    return rows,taxonomy


def is_portugal(raw):
    return raw['language'].upper()=='PT' and raw['media_name'].lower().removeprefix('www.') in PORTUGAL_OUTLETS


def portuguese():
    rows=[]; excluded=Counter()
    for split in ('train','test'):
        with open(ROOT/f'raw/partisan-{split}.csv',encoding='utf-8-sig',newline='') as stream:
            for raw in csv.DictReader(stream):
                if not is_portugal(raw):
                    excluded['other_language_or_publisher']+=1;continue
                label=raw['hyperpartisan_gold_label'].lower()
                if label not in ('0','1','0.0','1.0','true','false'): raise ValueError('Unknown PartisanLens binary label')
                label='1' if label in ('1','1.0','true') else '0'
                rows.append(record('partisan-pt','pt',str(raw['id']),raw['text'],[label],str(raw['original_id'])))
    taxonomy={key:dict(parent=key,definition=name,definition_source='native hyperpartisan label; identity parent is bookkeeping only')
        for key,name in [('0','Not hyperpartisan'),('1','Hyperpartisan')]}
    return rows,taxonomy,dict(excluded)


def grouped_splits(rows):
    """Connect source groups through exact duplicate text before partitioning."""
    parents={}
    def find(key):
        parents.setdefault(key,key)
        if parents[key]!=key:parents[key]=find(parents[key])
        return parents[key]
    for row in rows:
        a,b=find('source:'+row['source_group']),find('text:'+fingerprint(row['text']))
        parents[b]=a
    grouped=defaultdict(list)
    for row in rows:grouped[fingerprint(row['text'])].append(row)
    unique=[]; rejected=Counter()
    for group in grouped.values():
        if len({tuple(row['labels']) for row in group})!=1:
            rejected['conflicting_duplicate_records']+=len(group);continue
        row=sorted(group,key=lambda row:row['article_id'])[0]
        row=dict(row,duplicate_group=find('source:'+row['source_group']))
        unique.append(row);rejected['collapsed_duplicates']+=len(group)-1
    result={split:[] for split in SPLITS}
    for language in sorted({row['language'] for row in unique}):
        pool=sorted([row for row in unique if row['language']==language],key=lambda row:row['article_id'])
        groups=[row['duplicate_group'] for row in pool]
        train,held=next(GroupShuffleSplit(n_splits=1,test_size=.3,random_state=42).split(pool,groups=groups))
        dev,test=next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=42).split(held,groups=[groups[i] for i in held]))
        for split,indices in zip(SPLITS,(train,[held[i] for i in dev],[held[i] for i in test])):
            result[split].extend(pool[i] for i in indices)
    return result,dict(rejected)


def cap(rows, size):
    if len(rows)<=size:return rows
    strata=['|'.join(row['labels']) if len({tuple(x['labels']) for x in rows})<=2 else str(bool(row['labels'])) for row in rows]
    selected,_=train_test_split(rows,train_size=size,random_state=42,stratify=strata)
    return sorted(selected,key=lambda row:row['article_id'])


def prepare():
    summaries={}
    for name,loaded in [('slavic',slavic()),('hindi',hindi()),('portuguese',portuguese())]:
        rows,taxonomy=loaded[:2]; original=len(rows)
        for row in rows:
            row['narratives']=sorted({taxonomy[label]['parent'] for label in row['labels']})
        # Compute on manageable complete examples; do not truncate their text.
        rows=[row for row in rows if 5<=len(row['text'].split()) and len(row['text'])<=6000 and '\ufffd' not in row['text']]
        parts,rejected=grouped_splits(rows)
        caps=(500,150,150) if name=='slavic' else (400,100,150)
        for split,size in zip(SPLITS,caps):
            parts[split]=[row for lang in sorted({x['language'] for x in parts[split]}) for row in cap([x for x in parts[split] if x['language']==lang],size)]
        check_splits(*parts.values())
        group_sets=[{row['duplicate_group'] for row in parts[split]} for split in SPLITS]
        if any(group_sets[i]&group_sets[j] for i in range(3) for j in range(i)):raise ValueError('Source group leakage')
        folder=ROOT/name
        for split in SPLITS:write_jsonl(folder/(split+'.jsonl'),parts[split])
        write_json(folder/'taxonomy.json',taxonomy)
        summaries[name]={'raw_selected_records':original,'after_length_and_encoding_filter':len(rows),'duplicate_filter':rejected,
            'publisher_filter':loaded[2] if len(loaded)>2 else None,'label_inventory':list(taxonomy),
            'splits':{split:{language:{'records':len(pool:=[r for r in parts[split] if r['language']==language]),
                'source_groups':len({r['duplicate_group'] for r in pool}),'positive_label_counts':dict(Counter(l for r in pool for l in r['labels']))}
                for language in sorted({r['language'] for r in parts[split]})} for split in SPLITS}}
        variants=[{'method':x} for x in ('tfidf','lsa','transformer')]
        if name=='slavic':variants += [{'method':'graph','ablation':a} for a in ('none','self-loops','shuffled-edges','random-labels')]
        config={'dataset':{'slavic':'Slavic BG/RU sentence persuasion grouped pilot','hindi':'H-Prop-News native Hindi grouped pilot','portuguese':'PartisanLens Portugal-publisher headline pilot'}[name],
            'experiment_id':'multilingual-'+name,'label_schema':'native','native_decision':'multilabel' if name=='slavic' else 'single',
            **{split:str(folder/(split+'.jsonl')) for split in SPLITS},'taxonomy':str(folder/'taxonomy.json'),
            'output':'artifacts/native-multilingual-'+name,'variants':variants}
        write_json(folder/'experiment.json',config)
    write_json(ROOT/'protocol.json',{'datasets':summaries,'seed':42,'split':'custom source/duplicate-group 70/15/15; pooled publisher partitions',
        'sources':json.loads(SOURCES.read_text(encoding='utf-8')),'hierarchy_source':'https://aclanthology.org/2026.lrec-1.289.pdf#page=3',
        'portugal_publishers':sorted(PORTUGAL_OUTLETS),'limits':['Native auxiliary labels, not official narrative labels.',
        'Slavic normalized publisher sentences joined to labels by original document/offset key; raw offsets not silently repaired.',
        'Slavic language is confounded with domain: Bulgarian parliament, Russian social media.',
        'Small capped pilot, custom grouped splits, minimum five words and maximum 6000 characters.',
        'Portugal publisher location is a language-variety proxy; syndication and per-headline variety not verified.',
        'Hindi and Portuguese binary labels have no GNN experiment or shared cross-language label space.',
        'Exact duplicates and known source IDs checked; translations and semantic near duplicates not established.',
        'No raw text or source identifiers are published.']})
    print('Prepared native language pilots:', {name:{split:{lang:stats['records'] for lang,stats in groups.items()} for split,groups in item['splits'].items()} for name,item in summaries.items()}, flush=True)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    download();prepare()

if __name__=='__main__':main()
