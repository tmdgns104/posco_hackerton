"""Evidence-first, paired slag model verification for SteelTrust G7.

Only score G5-B, G6-B, and G6-Deep on exact same grouped 0.5-second
B/S evaluation windows. Every metric is from OUT-OF-FOLD saved predictions,
not from all-data fitted model files. No deployment claims.
"""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, roc_auc_score)
from . import slag_g6_core as g6

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'benchmarks'/'slag_b2'
MODEL_FILES={
    'G5-B SVM':'G5B_oof.csv',
    'G6-B ExtraTrees':'G6B_oof.csv',
    'G6 Deep MLP':'G6Deep_oof.csv',
}
MODEL_SHORT={'G5-B SVM':'G5B','G6-B ExtraTrees':'G6B','G6 Deep MLP':'G6Deep'}
EXPECTED_GROUPS=set(range(1,17))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def load_one(name, base=EVIDENCE):
    if name not in MODEL_FILES: raise KeyError(name)
    t=pd.read_csv(Path(base)/MODEL_FILES[name], encoding='utf-8-sig')
    if 'stage' in t.columns:t=t.rename(columns={'stage':'phase'})
    if 'cut' in t.columns:t=t.rename(columns={'cut':'threshold'})
    required={'group','phase','start_s','true_slag','score','threshold','pred_slag','fold'}
    if not required.issubset(t.columns):raise ValueError(f'{name}: missing columns {sorted(required-set(t.columns))}')
    t=t[list(required)].copy()
    t['group']=t.group.astype(int)
    t['phase']=t.phase.astype(str).str.upper()
    t['start_s']=t.start_s.astype(float).round(6)
    t['true_slag']=t.true_slag.astype(int)
    t['pred_slag']=t.pred_slag.astype(int)
    t['score']=t.score.astype(float)
    t['threshold']=t.threshold.astype(float)
    t['fold']=t.fold.astype(int)
    key=['group','phase','start_s']
    if t.duplicated(key).any():raise ValueError(f'{name}: duplicate windows')
    if set(t.group)!=EXPECTED_GROUPS or set(t.phase)!={'B','S'}:raise ValueError(f'{name}: wrong groups/stages')
    if not np.isfinite(t[['score','threshold','start_s']].to_numpy()).all():raise ValueError(f'{name}: nonfinite values')
    if not t.true_slag.isin([0,1]).all() or not t.pred_slag.isin([0,1]).all():raise ValueError('invalid labels')
    if not ((t.phase=='S').astype(int)==t.true_slag).all():raise ValueError('phase and truth mismatch')
    if not (t.score.ge(t.threshold).astype(int)==t.pred_slag).all():raise ValueError(f'{name}: score/threshold/prediction mismatch')
    counts=t.groupby(['group','phase']).size()
    for g in EXPECTED_GROUPS:
        if counts.get((g,'B'))!=5 or counts.get((g,'S'))!=10:raise ValueError(f'{name}: expected group {g}: 5 B + 10 S scored windows')
    if len(t)!=240:raise ValueError(f'{name}: expected 240 matched windows')
    if any(t.groupby('group').fold.nunique()!=1):raise ValueError(f'{name}: multiple folds per occurrence')
    return t.sort_values(key).reset_index(drop=True)


def aligned(base=EVIDENCE):
    frames={name:load_one(name,base) for name in MODEL_FILES}
    keys=['group','phase','start_s']
    canonical=frames['G6-B ExtraTrees'][keys+['true_slag']].copy()
    for name,t in frames.items():
        m=canonical.merge(t,on=keys,how='outer',validate='one_to_one',indicator=True,suffixes=('_canonical',''))
        if len(m)!=240 or (m['_merge']!='both').any() or not (m.true_slag_canonical==m.true_slag).all():
            raise ValueError(f'{name}: evaluation samples or truths do not match')
    return frames


def measures(t):
    y=t.true_slag.to_numpy(dtype=int); p=t.pred_slag.to_numpy(dtype=int);s=t.score.to_numpy(dtype=float)
    tp=int(((y==1)&(p==1)).sum());fn=int(((y==1)&(p==0)).sum())
    fp=int(((y==0)&(p==1)).sum());tn=int(((y==0)&(p==0)).sum())
    return {'TP':tp,'FN':fn,'FP':fp,'TN':tn,
            'recall':tp/(tp+fn),'fpr':fp/(fp+tn),
            'accuracy':(tp+tn)/len(t),
            'balanced_accuracy':float(balanced_accuracy_score(y,p)),
            'f1_B':float(f1_score(y,p,pos_label=0,zero_division=0)),
            'f1_S':float(f1_score(y,p,pos_label=1,zero_division=0)),
            'macro_f1':float(f1_score(y,p,average='macro',zero_division=0)),
            'roc_auc':float(roc_auc_score(y,s)),
            'pr_auc':float(average_precision_score(y,s))}


def summary(base=EVIDENCE):
    return pd.DataFrame([{'model':name,**measures(t)} for name,t in aligned(base).items()])


def paired(a,b):
    cols=['group','phase','start_s'];c=a.merge(b,on=cols,how='inner',validate='one_to_one',suffixes=('_before','_after'))
    if len(c)!=len(a) or len(c)!=len(b) or not (c.true_slag_before==c.true_slag_after).all():
        raise ValueError('Cannot compare unmatched independent test windows')
    good_before=c.pred_slag_before==c.true_slag_before
    good_after=c.pred_slag_after==c.true_slag_after
    c['change']=np.select([~good_before&good_after,good_before&~good_after],['fixed','new_error'],default='unchanged')
    return c


def group_table(frames):
    rows=[]
    for name,t in frames.items():
        for group,part in t.groupby('group'):
            m=measures(part)
            rows.append({'model':name,'group':int(group),**m})
    return pd.DataFrame(rows)


def bootstrap_delta(a,b,n=1200,seed=2026):
    paired_rows=paired(a,b)
    by_group={k:t for k,t in paired_rows.groupby('group')}
    rng=np.random.default_rng(seed);d=[]
    for _ in range(n):
        sample=pd.concat([by_group[k] for k in rng.choice(sorted(by_group),size=16,replace=True)],ignore_index=True)
        yy=sample.true_slag_before.to_numpy(int);pa=sample.pred_slag_before.to_numpy(int);pb=sample.pred_slag_after.to_numpy(int)
        d.append([(pb[yy==1].mean()-pa[yy==1].mean()),(pb[yy==0].mean()-pa[yy==0].mean())])
    return {'delta_recall_95pct':list(map(float,np.percentile(d,[2.5,97.5],axis=0)[:,0])),
            'delta_fpr_95pct':list(map(float,np.percentile(d,[2.5,97.5],axis=0)[:,1])),
            'n_independent_occurrences':16,
            'note':'Exploratory group bootstrap, repeated model development on 16 SSFD occurrences.'}


def events(t,persistence=2):
    out=[]
    for (group,phase),g in t.groupby(['group','phase']):
        x=g.sort_values('start_s').pred_slag.to_numpy(dtype=int)
        found=any(bool(np.all(x[i:i+persistence]==1)) for i in range(len(x)-persistence+1))
        out.append({'group':int(group),'phase':phase,'alarm':int(found)})
    return pd.DataFrame(out)


def integrity(base=EVIDENCE):
    paths=[*([Path(base)/v for v in MODEL_FILES.values()]), Path(base)/'G6B_model.joblib',
           Path(base)/'G5B_model.joblib',Path(base)/'G6Deep_summary.json']
    out=[]
    for path in paths:
        out.append({'artifact':path.name,'exists':path.is_file(),
                    'sha256':digest(path) if path.is_file() else None,
                    'size_bytes':path.stat().st_size if path.is_file() else None})
    return out


def g5_feature(raw):
    """Replicates the 36-dimensional G5-B SVM spectral feature transform."""
    w=np.asarray(raw,dtype=np.float64)
    if w.shape!=(3200,3) or not np.isfinite(w).all():raise ValueError('G5 feature shape must be (3200,3) finite')
    w=w-w.mean(axis=0)
    f,p=welch(w,fs=6400,nperseg=512,noverlap=256,axis=0)
    bands=[0,25,50,100,200,400,800,1600,2200,3201]
    bp=np.array([np.log(np.maximum(p[(f>=a)&(f<b)].mean(axis=0),1e-12)) for a,b in zip(bands[:-1],bands[1:])]).T.ravel()
    return np.concatenate([bp,np.log(np.std(w,axis=0)+1e-7),
        np.log(np.percentile(np.abs(w),95,axis=0)+1e-7),kurtosis(w,axis=0)]).astype(float)


def reference_features(ref, extractor):
    if len(ref)<16000:raise ValueError('B reference must contain >= 16000 samples (2.5 s)')
    return np.median(np.stack([extractor(ref[i:i+3200]) for i in range(0,16000,3200)]),axis=0)


def external_inference(reference_csv, target_csv, base=EVIDENCE):
    """Research-only prediction using ALL-SSFD fitted G5-B and G6-B models.

    Raw data NEVER enters B-reference automatically; caller must confirm true B.
    Do NOT apply held-out CV percentages to these full-data fitted artifacts.
    """
    if len(reference_csv)>25_000_000 or len(target_csv)>25_000_000:
        raise ValueError('25 MB per CSV limit; split long recordings')
    ref=g6.load_csv(reference_csv,'reference')
    target=g6.load_csv(target_csv,'target')
    if len(ref)<16000 or len(target)<3200: raise ValueError('B reference >= 2.5s and target >= 0.5s required')
    g5=joblib.load(Path(base)/'G5B_model.joblib')
    g6b=joblib.load(Path(base)/'G6B_model.joblib')
    if g5.get('protocol')!='B2' or g5.get('window_samples')!=3200:
        raise ValueError('G5-B model incompatible')
    if g6b.get('name')!='G6B_refspec_v1' or g6b.get('features')!=75 or g6b.get('fs')!=6400:
        raise ValueError('G6-B model incompatible')
    f5=reference_features(ref,g5_feature);f6=g6.baseline(ref)
    rows=[]
    for i in range(0,len(target)-3199,3200):
        w=target[i:i+3200]
        p5=float(g5['model'].predict_proba((g5_feature(w)-f5)[None,:])[0,1]);p6=float(g6b['estimator'].predict_proba((g6.feature(w)-f6)[None,:])[0,1])
        rows.append({'start_s':i/6400,'end_s':(i+3200)/6400,
                     'G5B_score':p5,'G5B_threshold':float(g5['threshold']),'G5B_flag':int(p5>=g5['threshold']),
                     'G6B_score':p6,'G6B_threshold':float(g6b['threshold']),'G6B_flag':int(p6>=g6b['threshold'])})
    return pd.DataFrame(rows)


def adoption_report(base=EVIDENCE):
    frames=aligned(base); m={name:measures(t) for name,t in frames.items()}
    baseline=m['G6-B ExtraTrees'];candidate=m['G6 Deep MLP']
    return {'research_deep_better_than_g6b': bool(candidate['recall']>baseline['recall'] and candidate['fpr']<=baseline['fpr']),
            'g6b_meets_aspirational_window_gate':bool(baseline['recall']>=.95 and baseline['fpr']<=.05),
            'g6deep_meets_aspirational_window_gate':bool(candidate['recall']>=.95 and candidate['fpr']<=.05),
            'independent_site_validation':False,
            'industrial_approved':False,
            'warning':'All group-held-out records are from 16 repeatedly studied SSFD occurrences. Confirmed B reference is mandatory.'}


def full_report(base=EVIDENCE):
    frames=aligned(base)
    return {'summary':summary(base).to_dict('records'),
            'g6b_vs_deep':bootstrap_delta(frames['G6-B ExtraTrees'],frames['G6 Deep MLP']),
            'regression_changes':paired(frames['G6-B ExtraTrees'],frames['G6 Deep MLP']).change.value_counts().to_dict(),
            'adoption':adoption_report(base),'integrity':integrity(base)}
