"""Render measured native-language extensions inside the existing five-study site."""
import json
from pathlib import Path
from statistics import mean,stdev

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SOURCE=Path('results/multilingual.json')
ROOT=Path('website/dist')
NAMES={'tfidf':'TF-IDF + LR','lsa':'LSA + LR','transformer':'MiniLM + LR','graph':'MiniLM + GCN'}

def label(run):
    return NAMES[run['method']]+(' / '+run['ablation'] if run.get('ablation','none')!='none' else '')


def append_catalog(payload):
    if not SOURCE.exists():return
    data=json.loads(SOURCE.read_text(encoding='utf-8'))
    entries=[r for r in payload['experiments'] if not r['id'].startswith('native-multilingual-')]
    for name,matrix in data['matrices'].items():
        for aggregate in matrix['aggregates']:
            key=aggregate['method']+'-'+aggregate['ablation']
            reports=[r for r in matrix['runs'] if r['method']==aggregate['method'] and r['ablation']==aggregate['ablation']]
            entries.append(dict(id='native-multilingual-'+name+'-'+key,study='Native multilingual extension',kind='classification',
                dataset=matrix['dataset'],name=label(reports[0]),role='Native-language comparison',metric='macro F1',value=aggregate['mean'],unit='fraction',
                n=reports[0]['metrics']['count'],status='measured',notes='Native auxiliary task; compare within dataset only. Custom grouped pilot.',
                details=dict(seeds=aggregate['seeds'],sample_std=aggregate['sample_std'],by_language=[r['metrics']['by_language'] for r in reports]),
                provenance={k:matrix[k] for k in ('commit','input_sha256','source_sha256')}))
    for r in data['transfer']['runs']:
        entries.append(dict(id=f"native-multilingual-transfer-{r['source_language']}-{r['target_language']}-{r['method']}",
            study='Slavic source-only transfer',kind='classification',dataset='Native 25-technique Slavic pilot',
            name=f"{r['source_language'].upper()} → {r['target_language'].upper()} · {NAMES[r['method']]}",role='Cross-lingual transfer',
            metric='macro F1',value=r['metrics']['macro_f1'],unit='fraction',n=r['metrics']['count'],status='measured',
            notes='Source-language training/dev only. One seed. Language and parliamentary/social-media domain shift are confounded.',
            details=dict(seed=r['seed'],source=r['source_language'],target=r['target_language']),
            provenance={k:data['transfer'][k] for k in ('commit','input_sha256','source_sha256')}))
    payload['experiments']=entries


def attach(payload):
    if not SOURCE.exists():return
    data=json.loads(SOURCE.read_text(encoding='utf-8'))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'path','svg.hashsalt':'native-language-v1',
        'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#314759','text.color':'#172b3a'})
    def plot(fig,name,alt):
        import xml.etree.ElementTree as ET
        path=ROOT/'plots'/name;path.parent.mkdir(parents=True,exist_ok=True)
        fig.tight_layout();fig.savefig(path,bbox_inches='tight');plt.close(fig)
        box=ET.parse(path).getroot().attrib['viewBox'].split()
        return dict(plot='plots/'+name,plot_alt=alt,plot_width=float(box[2]),plot_height=float(box[3]))
    slavic=data['matrices']['slavic'];rows=[]
    variants=slavic['aggregates'];fig,ax=plt.subplots(figsize=(9,4.4));x=np.arange(len(variants));width=.36
    for offset,language,color in [(-.5,'bg','#267785'),(.5,'ru','#728391')]:
        values=[];errors=[]
        for a in variants:
            runs=[r for r in slavic['runs'] if r['method']==a['method'] and r['ablation']==a['ablation']]
            scores=[r['metrics']['by_language'][language]['macro_f1'] for r in runs]
            values.append(mean(scores));errors.append(stdev(scores) if len(scores)>1 else 0)
        ax.bar(x+offset*width,values,width,yerr=errors,capsize=3,label=language.upper(),color=color)
    for a in variants:
        runs=[r for r in slavic['runs'] if r['method']==a['method'] and r['ablation']==a['ablation']]
        rows.append([label(runs[0]),f"{a['mean']:.4f}",*[f"{mean(r['metrics']['by_language'][l]['macro_f1'] for r in runs):.4f}" for l in ('bg','ru')],str(len(runs))])
    ax.set_xticks(x,[label(next(r for r in slavic['runs'] if r['method']==a['method'] and r['ablation']==a['ablation'])) for a in variants],rotation=25,ha='right')
    ax.set_ylabel('Fixed-inventory macro F1');ax.set_ylim(bottom=0);ax.legend()
    blocks=[dict(title='Native Bulgarian and Russian: shared hierarchy',
        description='Aim: test a shared multilingual transformer and 25-label/six-parent GNN on native data. Train on 500 BG + 500 RU sentences; select settings on 150 per language; test on 150 per language. Source documents and exact-duplicate components stay in one split. Full, self-loop, shuffled-edge and random-vector graphs each use seeds 13/42/87.',
        finding='Bars compare each language separately; graph error bars show sample standard deviation across three seeds, not confidence intervals. The pooled score uses all 300 test sentences and is not the average of language macro F1.',
        headers=['Method','Pooled F1','BG F1','RU F1','Seeds'],rows=rows,
        limits='Bulgarian parliamentary debates and Russian social media differ in domain as well as language. These are native persuasion techniques, not official narratives. Rare/absent test classes remain in the 25-label macro inventory.',
        **plot(fig,'native-slavic.svg','Bulgarian and Russian macro F1 by baseline and graph ablation; graph bars include seed standard deviation.'))]
    transfers=data['transfer']['runs'];fig,ax=plt.subplots(figsize=(8,3.5));pairs=[('bg','bg'),('bg','ru'),('ru','bg'),('ru','ru')];x=np.arange(4)
    for offset,method,color in [(-.5,'transformer','#728391'),(.5,'graph','#267785')]:
        values=[next(r['metrics']['macro_f1'] for r in transfers if r['source_language']==s and r['target_language']==t and r['method']==method) for s,t in pairs]
        ax.bar(x+offset*width,values,width,label=NAMES[method],color=color)
    ax.set_xticks(x,[f'{s.upper()} → {t.upper()}' for s,t in pairs]);ax.set_ylabel('Fixed-inventory macro F1');ax.legend();ax.set_ylim(bottom=0)
    blocks.append(dict(title='Cross-lingual transfer without target-language tuning',
        description='Aim: test whether representations transfer across languages. Each model trains on 500 source-language sentences and selects settings on 150 source-language development sentences only. The same fitted model predicts both source and target test sets. MiniLM + LR and the full GNN use seed 42.',
        finding='Source → source results provide the in-language reference. Source → other-language results measure a combined language/domain shift. Bilingual models above use more training examples, so their difference is not solely a language effect.',
        headers=['Train → test','Method','Test sentences','Macro F1'],rows=[[f"{r['source_language'].upper()} → {r['target_language'].upper()}",NAMES[r['method']],str(r['metrics']['count']),f"{r['metrics']['macro_f1']:.4f}"] for r in transfers],
        limits='One seed per transfer model and small source-grouped test sets. No target-language labels enter tuning. Results do not establish five-language official narrative performance.',
        **plot(fig,'native-transfer.svg','Source-only Bulgarian Russian transfer, contrasting MiniLM linear and graph heads in each direction.')))
    fig,axes=plt.subplots(1,2,figsize=(9,3.5));rows=[]
    for ax,name,title in zip(axes,('hindi','portuguese'),('Hindi propaganda','Portugal-publisher hyperpartisanship')):
        m=data['matrices'][name];values=[];names=[]
        for a in m['aggregates']:
            r=next(r for r in m['runs'] if r['method']==a['method']);values.append(a['mean']);names.append(NAMES[a['method']])
            rows.append([title,NAMES[a['method']],str(r['metrics']['count']),f"{a['mean']:.4f}",f"{r['metrics']['exact_match']:.4f}"])
        b=m['training_majority_baseline'];values.append(b['macro_f1']);names.append('Train-majority')
        rows.append([title,'Train-majority',str(b['count']),f"{b['macro_f1']:.4f}",f"{b['exact_match']:.4f}"])
        ax.bar(range(4),values,color=['#a2acb4','#728391','#267785','#d2d7da']);ax.set_xticks(range(4),names,rotation=25,ha='right');ax.set_title(title);ax.set_ylabel('Native binary macro F1');ax.set_ylim(0,1)
    blocks.append(dict(title='Native Hindi and European Portuguese publisher tests',
        description='Aim: check native-language classification beyond English/Arabic. H-Prop-News uses complete Hindi articles (400 train / 100 dev / 150 test). PartisanLens is restricted to named Portugal publishers (151 / 32 / 33 headlines), excluding Brazilian and unknown outlets. All methods use seed 42. No text is machine-translated.',
        finding='Each binary task is evaluated independently, with a baseline that always predicts the training-set majority class. Their labels are not merged: Hindi predicts propaganda, whereas Portuguese predicts hyperpartisanship.',
        headers=['Task','Method','Test n','Macro F1','Accuracy'],rows=rows,
        limits='Neither binary task has a native hierarchy, so no artificial GNN hierarchy is introduced. Portugal test n=33 gives substantial uncertainty; publisher location is a language-variety proxy, not sentence-level verification. Pilot filters retain complete texts with 5+ words and at most 6000 characters.',
        **plot(fig,'native-hindi-portuguese.svg','Separate native Hindi and Portugal-publisher binary macro F1 charts including training-majority controls.')))
    study=next(s for s in payload['studies'] if s['id']=='bilingual')
    study['title']='How do models handle multiple languages?'
    study['aim']='Compare encoders on English/Arabic news, then test native Bulgarian, Russian, Hindi and Portugal-publisher text, with source-only transfer where labels are shared.'
    study['extensions']=blocks
    study['extension_source']='results/multilingual.json'
    payload['multilingual_extension']='Native BG/RU/HI/PT pilot; independent auxiliary tasks, not official SemEval narrative scores.'
