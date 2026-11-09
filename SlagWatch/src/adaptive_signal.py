"""Adaptive red-team exploration for out-of-fold SSFD CNN/legacy detectors.

Data labels remain the original E/B/S phases, and synthetic perturbations require
physical interpretation and human review. This is *not* an independent test.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .adaptive_redteam import seed_of,suggestion
from .surface_data import ROOT

LOW=np.array([0.00,0.82,0.00,0.82])
HIGH=np.array([0.23,1.18,1.0,1.18])
PARAMS=('noise_fraction','global_gain','impulse_fraction','y_axis_gain')


def signal_change(raw, params,noise_field):
    p=np.asarray(params,dtype=float)
    if p.shape!=(4,) or not np.isfinite(p).all() or (p<LOW).any() or (p>HIGH).any():
        raise ValueError('invalid signal changes')
    if raw.shape!=(3200,3) or not np.isfinite(raw).all():
        raise ValueError('expected 3200x3 clean 0.5-second window')
    std=np.std(raw,axis=0).clip(1e-8)
    x=np.asarray((raw + noise_field*std*p[0])*p[1],dtype=np.float64)
    x[1580:1588,0]+=float(std[0]*p[2])*np.hanning(8)
    x[:,1]*=float(p[3])
    rmse=float(np.sqrt(np.mean((x-raw)**2))/max(float(np.sqrt(np.mean(std**2))),1e-8))
    return x,rmse


def _signal_objective(result,label,rmse):
    if result['input_status']!='VALID':
        return dict(prediction=None,wrong=False,priority=-3.0,margin=None,score=None,input_status=result['input_status'])
    pred=int(result['prediction'])
    score=float(result['score']);threshold=float(result['threshold'])
    margin=(score-threshold)*(1 if label==1 else -1)
    wrong=pred!=label
    priority=(2 if wrong else 0)-float(np.tanh(margin*10))-.1*rmse
    return dict(prediction=pred,wrong=wrong,priority=priority,margin=float(margin),score=score,threshold=threshold,
                input_status='VALID')


def _fold_for_group(group):
    from .hi_study import read_config
    folds=[x['fold_id'] for x in read_config()['folds'] if group in x['test']]
    if len(folds)!=1:raise ValueError('group must belong to exactly one out-of-fold test set')
    return folds[0]


def run_signal(file_id='S-4.csv',window_idx=0,model_id='C0',budget=20,warmup=8,seed=2026,compare_random=True):
    from .data import recording_paths,read_recording
    from .pair_detector import detector,file_hash
    from .pair_runs import CNN
    if model_id not in ('C0','C1','A1','J'):raise ValueError('supported signal models: C0,C1,A1,J')
    if not 8<=budget<=80 or not 2<=warmup<=budget or not 0<=window_idx<=9:
        raise ValueError('invalid signal trial settings')
    paths=recording_paths(ROOT/'data/raw')
    if file_id not in paths:raise ValueError('unknown official SSFD file')
    phase,n=file_id[:-4].split('-')
    group=int(n)
    label=int(phase=='S')
    fold=_fold_for_group(group)
    d=detector(model_id,fold,CNN if model_id in ('C0','C1') else None,
               file_hash(CNN/'manifest.json') if model_id in ('C0','C1') else None)
    if group not in d.test or group in d.train:
        raise ValueError('training-group leakage detected')
    original=read_recording(paths[file_id])[window_idx].astype(np.float64)
    def predict(w):return dict(**d.infer([w])[0],threshold=d.threshold)
    clean=predict(original)
    if clean['prediction']!=label:
        return dict(schema='steeltrust.adaptive_signal.v1',model_id=model_id,file_id=file_id,
                    window_idx=window_idx,skip='original_already_wrong',clean=clean,
                    warning='No valid induced-failure rate can be computed on a clean misclassification.')
    rng_noise=np.random.default_rng(seed_of('signal_noise',file_id,window_idx,seed))
    noise_field=rng_noise.normal(size=original.shape)
    results=[]
    for strategy in ('random','adaptive') if compare_random else ('adaptive',):
        rng=np.random.default_rng(seed_of('signal_trial',file_id,window_idx,seed))
        history=[]
        for i in range(budget):
            p=suggestion(history,rng,strategy,warmup,low=LOW,high=HIGH)
            x,rmse=signal_change(original,p,noise_field)
            r=dict(trial=i+1, parameters_vector=p.tolist(),parameters={k:float(v) for k,v in zip(PARAMS,p)},
                   input_sha256=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),
                   relative_rmse=rmse,visually_valid_unreviewed=bool(rmse<=.55))
            if r['visually_valid_unreviewed']:
                r.update(_signal_objective(predict(x),label,rmse))
            else:
                r.update(dict(prediction=None,wrong=False,priority=-3.0,score=None,margin=None,input_status='PERTURBATION_TOO_LARGE'))
            history.append(r)
        valid=[x for x in history if x['wrong']]
        results.append(dict(strategy=strategy,trials=history,failures=len(valid),
                            first_failure_trial=valid[0]['trial'] if valid else None,
                            minimum_failure_rmse=min([x['relative_rmse'] for x in valid],default=None)))
    return dict(schema='steeltrust.adaptive_signal.v1',generated_utc=datetime.now(timezone.utc).isoformat(),
                kind='exploratory_out_of_fold_reuse_NOT_independent_test',model_id=model_id,
                file_id=file_id,window_idx=window_idx,group=group,fold=fold,label=label,model_identity=d.identity,
                clean=clean,seed=seed,results=results,
                warning='This uses previously evaluated out-of-fold data for adaptive development; do not claim new independent test performance. Real phase labels may not be preserved by perturbations.')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--file',default='S-4.csv');p.add_argument('--window',type=int,default=0)
    p.add_argument('--model',choices=['C0','C1','A1','J'],default='C0')
    p.add_argument('--budget',type=int,default=12)
    a=p.parse_args()
    print(json.dumps(run_signal(file_id=a.file,window_idx=a.window,model_id=a.model,budget=a.budget),ensure_ascii=False,indent=2))
