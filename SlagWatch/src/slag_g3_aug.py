"""SSFD G3 noise-aware calibrated slag model; grouped/nested research evaluation.

This file does not change or overwrite any G2 checkpoints or their sealed results.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.model_selection import GroupKFold

from . import slag_g3 as base
from .data import load_dataset,read_recording

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/slag_g3_aug'
RATES=0.15
RUN_NAME='G3R'
CONDITIONS=('clean','noise_0p05','noise_0p08','gain_0p85','spike_x')


def deterministic_noise(w,idx,sigma):
    arr=np.asarray(w,dtype=float)
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f'stg3:{idx}:{sigma}'.encode()).digest()[:8],'big'))
    return arr+rng.normal(0,sigma*arr.std(axis=0),size=arr.shape)


def get_feature_bank(meta,waves):
    x_clean=np.stack([base.feature_window(w) for w in waves])
    p=meta.phase.to_numpy();g=meta.group_id.to_numpy();idx=meta.window_idx.to_numpy()
    ref=np.zeros_like(x_clean)
    for k in range(1,17):
        a=(g==k)&(p=='E')&(idx<base.CALIBRATION_WINDOWS)
        if a.sum()!=base.CALIBRATION_WINDOWS:raise ValueError('incorrect reference length')
        ref[g==k]=np.median(x_clean[a],axis=0)
    banks={'clean':x_clean-ref}
    for sigma in (.02,.05,.08):
        banks[f'noise_{sigma}']=np.stack([base.feature_window(deterministic_noise(w,i,sigma)) for i,w in enumerate(waves)])-ref
    banks['gain_0p85']=np.stack([base.feature_window(.85*w) for w in waves])-ref
    banks['spike_x']=np.stack([base.feature_window(base.perturb(w,'spike_x',meta.file_id.iloc[i],int(idx[i]))) for i,w in enumerate(waves)])-ref
    eligible=(p!='E')|(idx>=base.CALIBRATION_WINDOWS)
    assert len(waves)==480 and eligible.sum()==400
    return banks,eligible


def train_x(banks,idx):
    return np.vstack([banks['clean'][idx],banks['noise_0.02'][idx],banks['noise_0.05'][idx]])


def prepare():
    meta,w,_=load_dataset(ROOT/'data/raw')
    banks,eligible=get_feature_bank(meta,w)
    return meta,w,banks,eligible


def run():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'models').mkdir(exist_ok=True)
    meta,w,banks,eligible=prepare()
    g=meta.group_id.to_numpy();y=meta.y_true.to_numpy();p=meta.phase.to_numpy();ix=meta.window_idx.to_numpy()
    rows=[];calibration=[];files=[]
    for fold,(train0,test0) in enumerate(GroupKFold(4).split(banks['clean'],y,g)):
        tr=train0[eligible[train0]];te=test0[eligible[test0]]
        if set(g[tr])&set(g[te]):raise ValueError('outer group leakage')
        # Inner OOF: augmented model trained only on inner train groups.
        inner_clean=np.full(len(tr),np.nan)
        inner_noise=np.full(len(tr),np.nan)
        for a,b in GroupKFold(3).split(banks['clean'][tr],y[tr],g[tr]):
            train_indices=tr[a]; val_indices=tr[b]
            if set(g[train_indices])&set(g[val_indices]):raise ValueError('inner group leakage')
            m=base.estimator().fit(train_x(banks,train_indices),np.tile(y[train_indices],3))
            inner_clean[b]=base.scores(m,banks['clean'][val_indices])
            inner_noise[b]=base.scores(m,banks['noise_0.05'][val_indices])
        cut=base.choose_cut(np.r_[y[tr],y[tr]],np.r_[inner_clean,inner_noise],RATES)
        calibration.append(dict(fold=fold,cut=cut,train_groups=','.join(map(str,sorted(set(g[tr])))),
                    test_groups=','.join(map(str,sorted(set(g[te]))))))
        m=base.estimator().fit(train_x(banks,tr),np.tile(y[tr],3))
        model_path=OUT/'models'/f'fold_{fold}_{RUN_NAME}.joblib'
        joblib.dump(dict(estimator=m,cut=cut,name='R36F',version='G3R-noise-augmented',
                    train_groups=sorted(map(int,set(g[tr]))),test_groups=sorted(map(int,set(g[te]))),
                    calibration_windows=5,reference_required=True),model_path,compress=3)
        files.append(dict(path=str(model_path.relative_to(ROOT)),sha256=base.file_sha(model_path)))
        for scenario in CONDITIONS:
            v=base.scores(m,banks['noise_0.05' if scenario=='noise_0p05' else 'noise_0.08' if scenario=='noise_0p08' else scenario][te]);pred=v>=cut
            for k,j in enumerate(te):
                rows.append(dict(model_id=RUN_NAME,scenario=scenario,file_id=meta.file_id.iloc[j],
                  phase=p[j],group_id=int(g[j]),window_idx=int(ix[j]),fold_id=fold,
                  y_true=int(y[j]),score=float(v[k]),threshold=float(cut),prediction=int(pred[k])))
    predictions=pd.DataFrame(rows).sort_values(['scenario','group_id','file_id','window_idx'])
    predictions.to_csv(OUT/'predictions.csv',index=False)
    summaries=[]
    for condition,s in predictions.groupby('scenario'):
        summaries.append(dict(model_id=RUN_NAME,scenario=condition,**base.stats(s)))
    pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(calibration).to_csv(OUT/'inner_cuts.csv',index=False)
    # Refit all heats for prospective/unlabeled deployment ONLY, not evaluation.
    idx=np.flatnonzero(eligible)
    inner0=np.full(len(idx),np.nan); inner5=np.full(len(idx),np.nan)
    for a,b in GroupKFold(4).split(banks['clean'][idx],y[idx],g[idx]):
        tr=idx[a];te=idx[b]
        m=base.estimator().fit(train_x(banks,tr),np.tile(y[tr],3))
        inner0[b]=base.scores(m,banks['clean'][te]);inner5[b]=base.scores(m,banks['noise_0.05'][te])
    cut=base.choose_cut(np.r_[y[idx],y[idx]],np.r_[inner0,inner5],RATES)
    full_model=base.estimator().fit(train_x(banks,idx),np.tile(y[idx],3))
    full_path=OUT/'models'/f'full_{RUN_NAME}.joblib'
    joblib.dump(dict(estimator=full_model,cut=cut,name='R36F',version='G3R-noise-augmented',
                train_groups=list(range(1,17)),reference_required=True,full_data_fit=True,
                calibration_windows=5),full_path,compress=3)
    files.append(dict(path=str(full_path.relative_to(ROOT)),sha256=base.file_sha(full_path)))
    # J comparison only on same 400 scored windows with identical outer group holdout.
    old=ROOT/'experiments/detector_v1/predictions.csv'
    if old.exists():
        j=pd.read_csv(old);j=j[(j.scenario=='clean')&(j.model_id=='J')&(j.budget==0.15)]
        j=j[(j.phase!='E')|(j.window_idx>=5)]
        pnew=predictions[predictions.scenario=='clean']
        paired=pnew.merge(j[['file_id','window_idx','y_true','prediction','score']],on=['file_id','window_idx'],validate='one_to_one',suffixes=('_G3R','_J'))
        if len(paired)!=400 or not (paired.y_true_G3R==paired.y_true_J).all():raise ValueError('inconsistent paired baseline')
        paired.to_csv(OUT/'paired_vs_J.csv',index=False)
    manifest=dict(name='G3R noise-aware reference-relative spectral model',model_family='ExtraTreesClassifier (two robust/frequency representations concatenated)',
      data_source='SSFD v3 provided by user',data_manifest_sha256=base.file_sha(ROOT/'data/raw/manifest.json'),
      source_segments=48,source_occurrences=16,scored_windows=400,calibration_only_windows=80,
      n_outer_folds=4,n_inner_folds=3,reference='E, first 5x0.5s windows, confirmed slag-free same heat',
      default_cut=float(cut),random_state=42,model_files=files,
      pipeline='clean + Gaussian noise (.02,.05) training augmentation; threshold inner OOF mixed clean+.05 FPR<=15%; group-held-out outer clean/stress evaluation',
      protocol_limits=['Early E reference is explicitly supplied, not learned from held-out label at runtime.',
      'Grouped 16 known heats only. Model family/variants explored on this dataset; no untouched external test.',
      'Noise_0.05 is a seen perturbation family, noise_0.08 is a severity not used in training.',
      'File-level E/B/S labels cannot validate exact onset timing.',
      'Fails to meet industrial-grade S recall; never authorize automatic shutdown.'])
    (OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(pd.DataFrame(summaries).to_string(index=False));print('Default full cut:',cut)
    return predictions


def run_predict(reference,target,fold=None):
    refs=read_recording(Path(reference));w=read_recording(Path(target))
    path=OUT/'models'/(f'fold_{fold}_{RUN_NAME}.joblib' if fold is not None else f'full_{RUN_NAME}.joblib')
    model=joblib.load(path)
    return [dict(end_s=(i+1)*.5,**base.infer_with_reference(refs,v,model)) for i,v in enumerate(w)]


def main():
    ap=argparse.ArgumentParser()
    sub=ap.add_subparsers(dest='cmd',required=True)
    sub.add_parser('train')
    p=sub.add_parser('predict')
    p.add_argument('--reference',required=True,type=Path);p.add_argument('--target',required=True,type=Path)
    p.add_argument('--fold',type=int,choices=[0,1,2,3])
    a=ap.parse_args()
    if a.cmd=='train':run()
    else:print(json.dumps(run_predict(a.reference,a.target,a.fold),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
