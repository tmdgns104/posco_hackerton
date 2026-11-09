"""Offline CLI for the independent (unsealed) adaptive red-team experiment.

Examples:
 python -m src.adaptive_cli audit --out ./adaptive_runs/audit.json
 python -m src.adaptive_cli surface --model V0 --samples Cr_1 --budget 24 --out ./adaptive_runs/surface.json
 python -m src.adaptive_cli signal --model C1 --file S-4.csv --window 0 --out ./adaptive_runs/signal.json
"""
import argparse
import json
from pathlib import Path

from .adaptive_audit import audit
from .adaptive_redteam import run_surface,summary_table
from .surface_data import ROOT


def main():
    parser=argparse.ArgumentParser(description='SteelTrust Adaptive Red Team (read-only model evaluation)')
    sub=parser.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('audit');a.add_argument('--out',required=True)
    b=sub.add_parser('surface');b.add_argument('--model',choices=['V0','V1','V2','G1'],default='V0')
    b.add_argument('--samples',nargs='+',required=True)
    b.add_argument('--budget',type=int,default=24);b.add_argument('--warmup',type=int,default=8)
    b.add_argument('--seed',type=int,default=2026);b.add_argument('--out',required=True)
    c=sub.add_parser('signal');c.add_argument('--model',choices=['C0','C1','A1','J'],default='C1')
    c.add_argument('--file',default='S-4.csv');c.add_argument('--window',type=int,default=0)
    c.add_argument('--budget',type=int,default=20);c.add_argument('--warmup',type=int,default=8)
    c.add_argument('--seed',type=int,default=2026);c.add_argument('--out',required=True)
    args=parser.parse_args()
    path=Path(args.out).resolve()
    path.parent.mkdir(parents=True,exist_ok=True)
    if args.cmd=='audit':data=audit(ROOT)
    else:
        chk=audit(ROOT)
        if chk['n_failed']:
            parser.error('file hash checks failed, refusing to infer')
        if args.cmd=='surface':
            data=run_surface(ROOT,args.samples,model_id=args.model,budget=args.budget,
                             warmup=args.warmup,seed=args.seed)
        else:
            from .adaptive_signal import run_signal
            data=run_signal(file_id=args.file,window_idx=args.window,model_id=args.model,
                            budget=args.budget,warmup=args.warmup,seed=args.seed)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    if args.cmd=='surface':
        table=summary_table(data)
        if not table.empty:table.to_csv(path.with_suffix('.csv'),index=False)
    print(str(path))
    if args.cmd=='audit':print('checks=',data['n_checked'],'failed=',data['n_failed'])

if __name__=='__main__':main()
