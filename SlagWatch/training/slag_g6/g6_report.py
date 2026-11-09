"""Paired error-change and group bootstrap report for G6 vs archived G5 OOF.
Intervals reflect 16 observed occurrences and are exploratory, not certification.
"""
import csv,json
from pathlib import Path
import numpy as np
from g6 import write_csv,write_json

def load(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def analyze(g5,g6,out):
    old=load(g5);new=load(g6)
    idx=lambda r:(int(r['group']),r['stage'] if 'stage' in r else r['phase'],round(float(r['start_s']),5))
    a={idx(r):r for r in old};b={idx(r):r for r in new}
    if a.keys()!=b.keys():raise ValueError('Cannot compare unmatched windows')
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    rows=[]
    for key in sorted(a):
        r1=a[key];r2=b[key];truth=int(r2['true_slag'])
        if truth!=int(r1['true_slag']):raise ValueError('Truth mismatch')
        p1=int(r1['pred_slag']);p2=int(r2['pred_slag'])
        rows.append(dict(group=key[0],stage=key[1],start_s=key[2],y=truth,g5=p1,g6=p2,
                         state='fixed' if p1!=truth and p2==truth else 'regressed' if p1==truth and p2!=truth else 'unchanged'))
    group_rows=[]
    for g in range(1,17):
        q=[r for r in rows if r['group']==g]
        s=[r for r in q if r['y']==1];n=[r for r in q if r['y']==0]
        group_rows.append({'group':g,'S_windows':len(s),'B_windows':len(n),
                           'G5_S_TP':sum(r['g5'] for r in s),'G6_S_TP':sum(r['g6'] for r in s),
                           'G5_B_FP':sum(r['g5'] for r in n),'G6_B_FP':sum(r['g6'] for r in n)})
    write_csv(out/'paired_windows.csv',rows);write_csv(out/'per_group.csv',group_rows)
    # Bootstrap by independent casting occurrence, NEVER by correlated windows.
    rng=np.random.default_rng(2026);boot=[]
    group_map={g:[r for r in rows if r['group']==g] for g in range(1,17)}
    for _ in range(6000):
        groups=rng.choice(np.arange(1,17),size=16,replace=True)
        sampled=[r for g in groups for r in group_map[int(g)]]
        t=[r for r in sampled if r['y']==1];n=[r for r in sampled if r['y']==0]
        boot.append([(sum(r['g6'] for r in t)-sum(r['g5'] for r in t))/len(t),
                     (sum(r['g6'] for r in n)-sum(r['g5'] for r in n))/len(n)])
    ci=np.percentile(boot,[2.5,97.5],axis=0)
    def any_alarm(lst,model,streak):
        v=[r[model] for r in sorted(lst,key=lambda r:r['start_s'])]
        return any(all(v[i:i+streak]) for i in range(len(v)-streak+1))
    events={}
    for model in ['g5','g6']:
        events[model]={}
        for streak in [1,2,3]:
            cnt={'S_detected':0,'B_false_event':0}
            for g in range(1,17):
                for stage in ('S','B'):
                    seq=[r for r in group_map[g] if r['stage']==stage]
                    if any_alarm(seq,model,streak):
                        cnt['S_detected' if stage=='S' else 'B_false_event']+=1
            events[model][str(streak)]=cnt
    result={'n_independent_occurrences':16,'point_delta_slag_recall':5/160,'point_delta_non_slag_fpr':0.0,
            'group_bootstrap_95pct_delta_recall':ci[:,0].tolist(),
            'group_bootstrap_95pct_delta_fpr':ci[:,1].tolist(),
            'fixed':sum(r['state']=='fixed' for r in rows),'regressed':sum(r['state']=='regressed' for r in rows),
            'event_proxy':events,'caveats':[
                'Same 16 groups reused during iterative model development; bootstrap not independent prospective confirmation',
                'B and S come from separate 5-second files, so event proxy cannot measure actual detection delay',
                'B-only reference is independently confirmed and required in operation']}
    write_json(out/'comparison.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    analyze('reports/G5_B_reference_oof.csv','trained_models/oof_predictions.csv','reports')
