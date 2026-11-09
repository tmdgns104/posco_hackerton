"""Headless offline smoke tests and report export for G7 model comparison."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from . import slag_compare as c
from .adaptive_audit import audit


def main():
    p=argparse.ArgumentParser(description='SteelTrust G7 model audit - research only')
    q=p.add_subparsers(dest='command',required=True)
    q.add_parser('check')
    r=q.add_parser('report');r.add_argument('--out',default='reports/g7_reliability_report.json')
    v=q.add_parser('predict');v.add_argument('--reference',required=True);v.add_argument('--target',required=True);v.add_argument('--out',default='reports/g7_uploaded_predictions.csv')
    a=p.parse_args()
    if a.command in ('check','report'):
        result=c.full_report();seal=audit()
        result['legacy_sealed_files']={'tested':seal['n_checked'],'failures':seal['n_failed']}
        if a.command=='report':
            out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
            out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
            print('Saved',out)
        print(json.dumps({'models':result['summary'],'adoption':result['adoption'],
                          'original_integrity':result['legacy_sealed_files']},ensure_ascii=False,indent=2))
        if seal['n_failed']: raise SystemExit(2)
    elif a.command=='predict':
        result=c.external_inference(Path(a.reference).read_bytes(),Path(a.target).read_bytes())
        out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
        result.to_csv(out,index=False,encoding='utf-8-sig')
        print(f'Saved {len(result)} windows to {out}. RESEARCH ONLY.')


if __name__=='__main__':main()
