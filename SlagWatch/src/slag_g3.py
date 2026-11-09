"""SteelTrust G3 -- SSFD slag flow detection with per-heat early-steel reference.

Exploratory nested group CV; NEVER train on outer labels. No file/group IDs as
classifier features. Requires 2.5 s known slag-free E-phase calibration from
this same heat. Original G2 inference remains available independently.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from .data import load_dataset, read_recording, recording_paths, SIZE
from .detector import fine_spectrum
from .hi_models import robust36
from .quality import quality

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/slag_g3'
SEED=42
CALIBRATION_WINDOWS=5
# Frozen features from the published G2 detector protocol, not fitted on outer test.
WELCH_SETTINGS={
 'welch': {'fs':6400,'window':'hann','nperseg':512,'noverlap':256,
           'detrend':'constant','average':'median','scaling':'density'},
 'band_starts_hz':[2,25,50,100,200,400,800,1600], 'eps':1e-12}
CONFIG={
 'name':'G3-Reference-Spectral',
 'group_folds':4,'inner_folds':3,'target_inner_fpr':0.15,
 'calibration_windows':CALIBRATION_WINDOWS,
 'primary':'R36F',
 'candidates':{'R36F':'robust 36 + fine spectral 192; difference from per-heat E reference',
               'F192':'fine spectral 192; difference from per-heat E reference'},
 'estimator':{'n_estimators':180,'min_samples_leaf':6,'max_features':0.5,'class_weight':'balanced','random_state':42,'n_jobs':2},
 'limitations':'Exploratory reuse of 16 labelled heats; group-held-out, but model variants were compared on the same dataset. Fresh heats required for independent validation.'}


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def feature_window(w):
    w=np.asarray(w,dtype=np.float64)
    if reason:=quality(w):raise ValueError('Invalid reference or window: '+reason)
    return np.r_[robust36(w,WELCH_SETTINGS),fine_spectrum(w)].astype(np.float64)


def matrix(meta,waves):
    # The first 5 E windows are known slag-free CALIBRATION, not scored as tests.
    x=np.stack([feature_window(w) for w in waves])
    g=meta.group_id.to_numpy();phase=meta.phase.to_numpy();wi=meta.window_idx.to_numpy()
    diff=np.empty_like(x)
    for group in sorted(np.unique(g)):
        ref=(g==group)&(phase=='E')&(wi<CALIBRATION_WINDOWS)
        if ref.sum()!=CALIBRATION_WINDOWS:raise ValueError('Missing E-phase calibration for group '+str(group))
        diff[g==group]=x[g==group]-np.median(x[ref],axis=0)
    eligible=(phase!='E')|(wi>=CALIBRATION_WINDOWS)
    return diff,eligible


def view(diff,name):
    if name=='R36F':return diff
    if name=='F192':return diff[:,36:]
    raise ValueError('unknown model '+name)


def estimator():return ExtraTreesClassifier(**CONFIG['estimator'])


def scores(model,x):return np.asarray(model.predict_proba(x)[:,list(model.classes_).index(1)],dtype=float)


def choose_cut(y,s,target):
    """Max TP with FP <= inner OOF budget; deterministic tie breaks."""
    y=np.asarray(y);s=np.asarray(s)
    if len(y)!=len(s) or not np.isfinite(s).all() or len(set(y))!=2:raise ValueError('calibration labels/scores invalid')
    limit=int(np.floor(target*np.sum(y==0)))
    best=None
    for cut in np.r_[np.unique(s),np.nextafter(s.max(),np.inf)]:
        pred=s>=cut; fp=int(np.sum(pred&(y==0)))
        if fp>limit:continue
        tp=int(np.sum(pred&(y==1)))
        candidate=(tp,-fp,float(cut))
        if best is None or candidate>best:best=candidate
    return float(best[2])


def inner_oof(x,y,g):
    value=np.full(len(y),np.nan)
    for train,val in GroupKFold(CONFIG['inner_folds']).split(x,y,g):
        if set(g[train])&set(g[val]):raise AssertionError('inner group leakage')
        model=estimator().fit(x[train],y[train]);value[val]=scores(model,x[val])
    if not np.isfinite(value).all():raise AssertionError('unscored calibration rows')
    return value


def training_data():
    meta,waves,_=load_dataset(ROOT/'data/raw')
    if len(waves)!=480:raise ValueError('Expected 480 SSFD windows')
    diff,eligible=matrix(meta,waves)
    if eligible.sum()!=400:raise ValueError('Expected 400 scored windows, 80 E reference windows')
    return meta,waves,diff,eligible


def perturb(w,condition,file_id,idx):
    w=np.array(w,copy=True,dtype=float)
    if condition=='clean':return w
    if condition=='gain_0p85':return .85*w
    if condition=='noise_0p05':
        seed=int.from_bytes(hashlib.sha256(f'{file_id}/{idx}/noise_0p05/steeltrustg3'.encode()).digest()[:8],'big')
        rng=np.random.default_rng(seed)
        return w+rng.normal(0,.05*w.std(axis=0),size=w.shape)
    if condition=='spike_x':
        w[1600:1608,0]+=.4*np.std(w[:,0])*np.hanning(8)
        return w
    raise ValueError('unknown condition')


def stats(rows):
    y=rows.y_true.to_numpy();p=rows.prediction.to_numpy();s=rows.score.to_numpy()
    tp=int(((y==1)&(p==1)).sum());fp=int(((y==0)&(p==1)).sum())
    fn=int(((y==1)&(p==0)).sum());tn=int(((y==0)&(p==0)).sum())
    return dict(n=len(rows),tp=tp,fn=fn,fp=fp,tn=tn,recall=tp/(tp+fn),fpr=fp/(fp+tn),
                precision=tp/(tp+fp) if tp+fp else None,
                roc_auc=float(roc_auc_score(y,s)),average_precision=float(average_precision_score(y,s)))


def evaluate():
    meta,waves,diff,eligible=training_data()
    g=meta.group_id.to_numpy();y=meta.y_true.to_numpy();p=meta.phase.to_numpy();ix=meta.window_idx.to_numpy()
    idx=np.flatnonzero(eligible)
    splits=list(GroupKFold(4).split(diff,y,g))
    allpred=[];inner_report=[];model_info=[]
    # The training population and candidate list are fixed before outer evaluation.
    for fold,(tr0,te0) in enumerate(splits):
        tr=tr0[eligible[tr0]];te=te0[eligible[te0]]
        if set(g[tr]) & set(g[te]):raise AssertionError('outer group leakage')
        for name in CONFIG['candidates']:
            x=view(diff,name)
            oof=inner_oof(x[tr],y[tr],g[tr]);cut=choose_cut(y[tr],oof,CONFIG['target_inner_fpr'])
            inner_pred=(oof>=cut)
            inner_report.append(dict(fold=fold,model=name,cut=cut,inner_tp=int(((y[tr]==1)&inner_pred).sum()),
                                    inner_fp=int(((y[tr]==0)&inner_pred).sum()),train_groups=','.join(map(str,sorted(set(g[tr])))),
                                    test_groups=','.join(map(str,sorted(set(g[te]))))))
            model=estimator().fit(x[tr],y[tr])
            fold_path=OUT/'models'/f'fold_{fold}_{name}.joblib'
            joblib.dump(dict(estimator=model,cut=cut,name=name,settings=WELCH_SETTINGS,
                             calibration_windows=CALIBRATION_WINDOWS,train_groups=sorted(set(map(int,g[tr]))),
                             test_groups=sorted(set(map(int,g[te])))),fold_path,compress=3)
            model_info.append(dict(path=str(fold_path.relative_to(ROOT)),sha256=file_sha(fold_path)))
            base=diff[te]
            refs={int(group):np.median(np.stack([feature_window(waves[k]) for k in np.flatnonzero((g==group)&(p=='E')&(ix<5))]),axis=0)
                  for group in sorted(set(g[te]))}
            for condition in ('clean','noise_0p05','gain_0p85','spike_x'):
                if condition=='clean':xt=base
                else:
                    feat=np.stack([feature_window(perturb(waves[j],condition,meta.file_id.iloc[j],int(ix[j]))) for j in te])
                    xt=feat-np.stack([refs[int(g[j])] for j in te])
                v=scores(model,view(xt,name));pred=(v>=cut).astype(int)
                for k,j in enumerate(te):
                    allpred.append(dict(file_id=meta.file_id.iloc[j],phase=p[j],group_id=int(g[j]),
                      window_idx=int(ix[j]),fold_id=fold,model_id=name,scenario=condition,
                      y_true=int(y[j]),score=float(v[k]),threshold=float(cut),prediction=int(pred[k])))
    predictions=pd.DataFrame(allpred).sort_values(['model_id','scenario','group_id','file_id','window_idx'])
    predictions.to_csv(OUT/'predictions.csv',index=False)
    summary=[]
    for (model,condition),group in predictions.groupby(['model_id','scenario']):
        summary.append(dict(model_id=model,scenario=condition,**stats(group)))
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(inner_report).to_csv(OUT/'inner_cuts.csv',index=False)
    full_models=[]
    for name in CONFIG['candidates']:
        x=view(diff[eligible],name);yg=y[eligible];gg=g[eligible]
        inner=inner_oof(x,yg,gg);cut=choose_cut(yg,inner,CONFIG['target_inner_fpr'])
        mdl=estimator().fit(x,yg)
        path=OUT/'models'/f'full_{name}.joblib'
        joblib.dump(dict(estimator=mdl,cut=cut,name=name,settings=WELCH_SETTINGS,calibration_windows=CALIBRATION_WINDOWS,
                         train_groups=sorted(set(map(int,gg))),full_data_fit=True),path,compress=3)
        full_models.append(dict(path=str(path.relative_to(ROOT)),sha256=file_sha(path),cut=cut))
    # Legacy J scores and decisions are from previously generated grouped outer fold experiments.
    legacy=ROOT/'experiments/detector_v1/predictions.csv'
    paired_results=[]
    if legacy.exists():
        df=pd.read_csv(legacy)
        df=df[(df.scenario=='clean')&(df.model_id=='J')&(df.budget==0.15)]
        matched=df[(df.phase!='E')|(df.window_idx>=5)]
        legacy_summary=stats(matched)
        (OUT/'legacy_J_same_400.json').write_text(json.dumps(legacy_summary,indent=2),encoding='utf-8')
        assert len(matched)==400
        for name in CONFIG['candidates']:
            mine=predictions[(predictions.model_id==name)&(predictions.scenario=='clean')]
            joined=mine.merge(matched[['file_id','window_idx','y_true','prediction']],on=['file_id','window_idx'],
                              suffixes=('_new','_J'),validate='one_to_one')
            assert len(joined)==400 and (joined.y_true_new==joined.y_true_J).all()
            before=joined.prediction_J.to_numpy();after=joined.prediction_new.to_numpy();truth=joined.y_true_new.to_numpy()
            paired_results.append(dict(model_id=name,old_J_tp=int(((truth==1)&(before==1)).sum()),
                  old_J_fp=int(((truth==0)&(before==1)).sum()),
                  new_tp=int(((truth==1)&(after==1)).sum()),new_fp=int(((truth==0)&(after==1)).sum()),
                  new_detected=int(((truth==1)&(before==0)&(after==1)).sum()),
                  newly_missed=int(((truth==1)&(before==1)&(after==0)).sum()),
                  removed_fp=int(((truth==0)&(before==1)&(after==0)).sum()),
                  new_fp_cases=int(((truth==0)&(before==0)&(after==1)).sum())))
        pd.DataFrame(paired_results).to_csv(OUT/'paired_vs_J.csv',index=False)
    report=dict(config=CONFIG,source_sha256=file_sha(ROOT/'data/raw/manifest.json'),
                dataset='48 records x 5 sec; 16 E/B/S triplets; E first 2.5sec is calibration input',
                scored_windows=400,excluded_reference_windows=80,
                outer_models=model_info,full_models=full_models,
                warnings=['Development/selection dataset; not a fresh locked independent test',
                          'Full models trained on all groups; their own predictions cannot establish generalization',
                          'Reference must be a certified early slag-free phase from the SAME heat',
                          'File-level phase labels do not measure precise slag onset time',
                          'Artificial stress simulations do not establish field safety'])
    (OUT/'run_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(summary).to_string(index=False))
    if paired_results:print('PAIRED',paired_results)
    return predictions


def infer_with_reference(ref_windows,target_window,bundle):
    """Operational contract: E baseline is provided separately by the caller.

    Caller must know E truly slag-free; no automatic classifier assumption.
    """
    if len(ref_windows)<CALIBRATION_WINDOWS:return dict(status='NEEDS_REFERENCE',prediction=None)
    try:
        reference=np.median(np.stack([feature_window(w) for w in ref_windows[:CALIBRATION_WINDOWS]]),axis=0)
        feature=feature_window(target_window)-reference
    except ValueError as e:
        return dict(status='INPUT_HOLD',reason=str(e),prediction=None)
    s=float(scores(bundle['estimator'],view(feature.reshape(1,-1),bundle['name']))[0])
    return dict(status='VALID',prediction=int(s>=bundle['cut']),score=s,cut=float(bundle['cut']),
                model=bundle['name'],reference_seconds=2.5)


def predict_csv(reference,target,model_id='R36F',fold=None):
    ref=read_recording(Path(reference));other=read_recording(Path(target))
    path=OUT/'models'/(f'full_{model_id}.joblib' if fold is None else f'fold_{fold}_{model_id}.joblib')
    bundle=joblib.load(path)
    result=[]
    for i,w in enumerate(other):result.append(dict(end_s=(i+1)*0.5,**infer_with_reference(ref,w,bundle)))
    return result


def main():
    parser=argparse.ArgumentParser(description='Train or run SSFD slag-flow G3 calibrated detector')
    sub=parser.add_subparsers(dest='action',required=True)
    sub.add_parser('train')
    p=sub.add_parser('predict');p.add_argument('--reference',required=True,type=Path);p.add_argument('--target',required=True,type=Path)
    p.add_argument('--model',choices=['R36F','F192'],default='R36F');p.add_argument('--fold',type=int,choices=[0,1,2,3])
    args=parser.parse_args()
    if args.action=='train':
        OUT.mkdir(parents=True,exist_ok=True);(OUT/'models').mkdir(exist_ok=True)
        evaluate()
    else:print(json.dumps(predict_csv(args.reference,args.target,args.model,args.fold),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
