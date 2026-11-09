"""Read-only, cohort-aware model catalogue for SteelTrust G8.2.

Historical records are never rewritten. Paired comparisons require exactly the
same observation keys and class labels. Distinct experimental protocols are
never assigned one global ranking.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from . import g8_core
from . import slag_compare

ROOT = Path(__file__).resolve().parents[1]
SURFACE_MODELS = ('V0', 'V1', 'V2', 'G1')
SLAG_COHORTS = {
    '기초 진동 · E/B/S 480구간': ('A1', 'J', 'C0', 'C1'),
    'E 기준 진동 · E/B/S 400구간': ('J', 'G3R'),
    'B 기준 진동 · B/S 240구간': ('G5-B', 'G6-B', 'G6 Deep'),
}
SLAG_MODELS = ('A1', 'J', 'C0', 'C1', 'G3R', 'G5-B', 'G6-B', 'G6 Deep')
SLAG_TECH = {
    'A1':'초기 주파수 기준', 'J':'주파수 특징 SVM', 'C0':'1D CNN',
    'C1':'RMS 정규화 CNN', 'G3R':'E 기준 ExtraTrees', 'G5-B':'B 기준 SVM',
    'G6-B':'B 기준 ExtraTrees', 'G6 Deep':'Residual MLP',
}
SLAG_LABEL = {v:k for k,v in {
    'G5-B SVM':'G5-B','G6-B ExtraTrees':'G6-B','G6 Deep MLP':'G6 Deep'}.items()}


def _need(df: pd.DataFrame, columns, source: str):
    missing=set(columns)-set(df.columns)
    if missing: raise ValueError(f'{source}: 필수 열 누락 {sorted(missing)}')


def _exactly_same_keys(frames: dict[str,pd.DataFrame], keys, truth, hash_key=None):
    ref=None
    for name,df in frames.items():
        _need(df,[*keys,truth],name)
        if df[list(keys)].isna().any().any() or df.duplicated(list(keys)).any():
            raise ValueError(f'{name}: 평가 ID 누락 또는 중복')
        cols=[*keys,truth]+([hash_key] if hash_key else [])
        x=df[cols].sort_values(list(keys)).reset_index(drop=True)
        if ref is not None and not x.equals(ref):
            raise ValueError(f'{name}: 비교 모델의 시험 입력·라벨이 서로 다릅니다.')
        ref=x


@lru_cache(maxsize=2)
def surface_records(split='validation'):
    """Validation includes all four; untouched test is V0/G1 only."""
    if split=='validation':
        v=pd.read_csv(ROOT/'experiments/surface_v1/validation_predictions.csv.gz')
        g=pd.read_csv(ROOT/'experiments/surface_g1/validation_predictions.csv.gz')
        frames={n:v[v.model_id==n].copy() for n in ('V0','V1','V2')}
        frames['G1']=g[g.model_id=='G1'].copy()
    elif split=='test':
        g=pd.read_csv(ROOT/'experiments/surface_g1/test_predictions.csv.gz')
        frames={n:g[g.model_id==n].copy() for n in ('V0','G1')}
    else:
        raise ValueError('알 수 없는 평가 분할')
    key=['sample_id','sigma','seed']
    _exactly_same_keys(frames,key,'label','raw_input_hash')
    result={}
    for name,df in frames.items():
        t=df[key+['label','prediction']].copy()
        t['correct']=t.prediction.eq(t.label)
        result[name]=t.sort_values(key).reset_index(drop=True)
        if len(t)!=3600:raise ValueError(f'{name}: 예상 평가 입력 3600개와 불일치')
    return result


def surface_summary(frames):
    rows=[]
    for name,t in frames.items():
        clean=t[t.sigma.eq(0)]
        noise=t[t.sigma.gt(0)]
        per_condition=noise.groupby(['sigma','seed']).apply(
            lambda d:f1_score(d.label,d.prediction,average='macro',zero_division=0),
            include_groups=False)
        rows.append({'모델':name,'원본 정확도(%)':100*clean.correct.mean(),
                     '잡음 정확도(%)':100*noise.correct.mean(),
                     '잡음 Macro F1(%)':100*per_condition.mean(),
                     '잡음 오분류(회)':int((~noise.correct).sum()),
                     '원본 오분류(회)':int((~clean.correct).sum())})
    return pd.DataFrame(rows)


def surface_conditions(frames):
    rows=[]
    for name,t in frames.items():
        for sigma,part in t.groupby('sigma'):
            rows.append({'모델':name,'잡음 σ':float(sigma),'정확도(%)':100*part.correct.mean(),
                         '오류(회)':int((~part.correct).sum()),'입력 수':len(part)})
    return pd.DataFrame(rows)


def surface_pair(a,b, frames):
    if a==b or a not in frames or b not in frames: raise ValueError('서로 다른 지원 모델 두 개를 선택하세요.')
    k=['sample_id','sigma','seed']
    p=frames[a].merge(frames[b],on=k,validate='one_to_one',suffixes=('_before','_after'))
    if len(p)!=len(frames[a]) or len(p)!=len(frames[b]) or not p.label_before.eq(p.label_after).all():
        raise ValueError('시험 표본 또는 정답이 다릅니다.')
    p['change']=np.select([~p.correct_before&p.correct_after,
                           p.correct_before&~p.correct_after,
                           ~p.correct_before&~p.correct_after],
                          ['fixed','new_error','both_wrong'],default='both_correct')
    return p


def _slag_frame(df, name, key, group, phase, true, pred, scenario, fold='fold_id'):
    part=df.copy()
    _need(part,[key,group,phase,true,pred],name)
    part=part.rename(columns={key:'window',group:'group',phase:'phase',true:'truth',pred:'pred',fold:'fold'})
    part['model']=name
    part['window']=part.window.astype(float).round(6)
    part['group']=part.group.astype(int)
    part['truth']=part.truth.astype(int)
    part['pred']=part.pred.astype(int)
    part['phase']=part.phase.astype(str).str.upper()
    if not part.truth.isin((0,1)).all() or not part.pred.isin((0,1)).all():
        raise ValueError(f'{name}: 비정상 라벨')
    if not part.truth.eq(part.phase.eq('S').astype(int)).all():
        raise ValueError(f'{name}: 슬래그 정답·단계 불일치')
    if not (part.score.ge(part.threshold).astype(int)==part.pred).all():
        raise ValueError(f'{name}: 점수·임계값·판정 불일치')
    return part[['model','group','phase','window','truth','pred','score','threshold','fold']].copy()


@lru_cache(maxsize=1)
def slag_records():
    first=pd.read_csv(ROOT/'experiments/cnn_v1/predictions.csv')
    first=first[first.scenario.eq('clean')]
    early={n:_slag_frame(first[first.model_id.eq(n)],n,'window_idx','group_id','phase','y_true','prediction','clean')
           for n in ('A1','J','C0','C1')}
    _exactly_same_keys(early,['group','phase','window'],'truth')
    if not all(len(v)==480 for v in early.values()):raise ValueError('기초 E/B/S 480구간 불일치')

    g3=pd.read_csv(ROOT/'experiments/slag_g3_aug/predictions.csv')
    g3=g3[(g3.model_id=='G3R')&(g3.scenario=='clean')]
    g3=_slag_frame(g3,'G3R','window_idx','group_id','phase','y_true','prediction','clean')
    j=early['J'].merge(g3[['group','phase','window']],on=['group','phase','window'],validate='one_to_one')
    mid={'J':j,'G3R':g3}
    _exactly_same_keys(mid,['group','phase','window'],'truth')
    if len(j)!=400 or len(g3)!=400:raise ValueError('E 기준 비교 400구간 불일치')

    raw=slag_compare.aligned()
    late={}
    for name,filename in [('G5-B','G5-B SVM'),('G6-B','G6-B ExtraTrees'),('G6 Deep','G6 Deep MLP')]:
        t=raw[filename]
        x=t.rename(columns={'start_s':'window','true_slag':'truth','pred_slag':'pred'})
        x['model']=name
        x=x[['model','group','phase','window','truth','pred','score','threshold','fold']].copy()
        late[name]=x
    _exactly_same_keys(late,['group','phase','window'],'truth')
    if not all(len(v)==240 for v in late.values()):raise ValueError('B 기준 비교 240구간 불일치')
    return {'기초 진동 · E/B/S 480구간':early,
            'E 기준 진동 · E/B/S 400구간':mid,
            'B 기준 진동 · B/S 240구간':late}


def slag_measures(t):
    y=t.truth.to_numpy(int);p=t.pred.to_numpy(int)
    tp=int(np.sum((y==1)&(p==1)));fp=int(np.sum((y==0)&(p==1)))
    fn=int(np.sum((y==1)&(p==0)));tn=int(np.sum((y==0)&(p==0)))
    d={'TP':tp,'FN':fn,'FP':fp,'TN':tn,'슬래그 검출률(%)':100*tp/(tp+fn),
       '오경보율(%)':100*fp/(fp+tn),'정확도(%)':100*(tp+tn)/len(t),
       'Macro F1(%)':100*f1_score(y,p,average='macro',zero_division=0)}
    if len(set(y))==2: d['ROC-AUC']=float(roc_auc_score(y,t.score))
    else:d['ROC-AUC']=float('nan')
    return d


def slag_summary(cohorts):
    result=[]
    for cohort,frames in cohorts.items():
        for name,t in frames.items():
            result.append({'평가 그룹':cohort,'모델':name,'기술':SLAG_TECH[name],
                           '평가 구간':len(t),**slag_measures(t)})
    return pd.DataFrame(result)


def slag_pair(cohort,a,b, cohorts):
    if cohort not in cohorts:raise ValueError('알 수 없는 평가 그룹')
    f=cohorts[cohort]
    if a==b or a not in f or b not in f:raise ValueError('동일 평가 그룹의 서로 다른 두 모델을 선택하세요.')
    k=['group','phase','window']
    paired=f[a].merge(f[b],on=k,validate='one_to_one',suffixes=('_before','_after'))
    if len(paired)!=len(f[a]) or len(paired)!=len(f[b]) or not paired.truth_before.eq(paired.truth_after).all():
        raise ValueError('동일한 진동 구간과 정답만 비교할 수 있습니다.')
    paired['change']=np.select([paired.pred_before.ne(paired.truth_before)&paired.pred_after.eq(paired.truth_after),
                                paired.pred_before.eq(paired.truth_before)&paired.pred_after.ne(paired.truth_after),
                                paired.pred_before.ne(paired.truth_before)&paired.pred_after.ne(paired.truth_after)],
                               ['fixed','new_error','both_wrong'],default='both_correct')
    return paired


def _fresh():
    """Used only by tests when deliberately mutating copies."""
    surface_records.cache_clear();slag_records.cache_clear()
