"""Dedicated official-role ST1 commands; raw data and checkpoints stay local."""
import argparse
import json
from pathlib import Path
from .data import read_jsonl,write_json,write_jsonl


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('train');p.add_argument('--train',required=True);p.add_argument('--dev',required=True);p.add_argument('--out',required=True)
    p.add_argument('--encoder',choices=('minilm','xlmr'),default='minilm');p.add_argument('--seed',type=int,default=42);p.add_argument('--ablation',choices=('full','unmarked'),default='full')
    p=sub.add_parser('predict');p.add_argument('--model',required=True);p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('prototype');p.add_argument('--input',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('evaluate');p.add_argument('--gold',required=True);p.add_argument('--predictions',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('export');p.add_argument('--predictions',required=True);p.add_argument('--out',required=True)
    args=parser.parse_args();from . import entity
    if args.command=='train':
        import joblib
        model=entity.train(read_jsonl(args.train),read_jsonl(args.dev),args.encoder,args.seed,args.ablation)
        Path(args.out).parent.mkdir(parents=True,exist_ok=True);joblib.dump(model,args.out)
        write_json(args.out+'.json',{k:v for k,v in model.items() if k!='head'})
    elif args.command=='predict':
        import joblib
        write_jsonl(args.out,entity.predict(joblib.load(args.model),read_jsonl(args.data)))
    elif args.command=='prototype':
        payload=json.loads(Path(args.input).read_text(encoding='utf-8'))
        write_json(args.out,entity.prototype(entity.mentions_from_payload(payload)))
    elif args.command=='evaluate':write_json(args.out,entity.evaluate(read_jsonl(args.gold),read_jsonl(args.predictions)))
    else:entity.export_tsv(read_jsonl(args.predictions),args.out)

if __name__=='__main__':main()
