"""Comparison gate. Never deploy solely on SSFD CV results."""
import argparse,csv,json
from pathlib import Path
from g6 import write_json

def load(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    result={}
    for r in rows:
        phase=r['phase'] if 'phase' in r else r['stage']
        key=(int(r['group']),phase,round(float(r['start_s']),5))
        if key in result:raise ValueError('Duplicate key')
        result[key]=(int(r['true_slag']),int(r['pred_slag']))
    return result

def metric(d):
    s=[v for v in d.values() if v[0]==1];n=[v for v in d.values() if v[0]==0]
    return {'TP':sum(v[1] for v in s),'S':len(s),'FP':sum(v[1] for v in n),'B':len(n),
            'slag_recall':sum(v[1] for v in s)/len(s), 'false_alarm_rate':sum(v[1] for v in n)/len(n)}

def check(baseline,candidate,out):
    b=load(baseline);c=load(candidate)
    if b.keys()!=c.keys():raise ValueError('Candidate and baseline use different held-out samples / incomplete CV')
    if any(b[k][0]!=c[k][0] for k in b):raise ValueError('Ground truth mismatch')
    m=metric(b);n=metric(c)
    research_improvement=(n['slag_recall']>m['slag_recall'] and n['false_alarm_rate']<=m['false_alarm_rate'])
    aspirational=(n['slag_recall']>=.95 and n['false_alarm_rate']<=.05)
    rep={'model_comparison':'same sample paired CV only','baseline':m,'candidate':n,
         'research_improvement':research_improvement,
         'aspirational_95recall_5fpr_met':aspirational,
         'production_approved':False,
         'reason':'Independent casting records and an operational B-reference validation are mandatory for deployment.'}
    write_json(out,rep);print(json.dumps(rep,indent=2))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--baseline',default='trained_models/oof_predictions.csv')
    a.add_argument('--candidate',default='runs/g6_deep/oof_predictions.csv');a.add_argument('--out',default='runs/g6_deep/compare_gate.json')
    x=a.parse_args();check(x.baseline,x.candidate,x.out)
