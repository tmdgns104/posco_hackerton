"""Fit R0, examine inner R0 failures, then fit R1 on the same frozen representations."""
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from aeon.transformations.collection.convolution_based import MiniRocket
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .cnn_eval import rank_metrics
from .data import load_dataset
from .faults import inject_fault
from .hi_study import ROOT, read_config, select_cut
from .pair_detector import digest, file_hash
from .rocket_detector import RUN, NAMES, channel_first, decisions, representation_hash, transform, versions

PROTOCOL=RUN/"protocol.json"
DIAG={"noise":("noise","all",.2,0),"spike":("spike","x",5.,16),
      "new_y_spike":("spike","y",4.,24),"new_z_spike":("spike","z",6.,12)}


def write_json(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf8")


def seed_for(namespace,fold,inner,source,condition):
    entropy=[42,namespace,fold,inner,source,condition]
    value=int(np.random.SeedSequence(entropy).generate_state(1)[0])
    return (namespace<<32) | value


def changed_windows(meta,waves,scale,scope,fold,inner,condition,log):
    training=scope=="train_aug"
    namespace=313 if training else 719
    conditions=("noise","impulse") if training else tuple(DIAG)
    changed=[]
    for r in meta.itertuples():
        seed=seed_for(namespace,fold,inner,r.Index,conditions.index(condition))
        if training:
            rng=np.random.default_rng(seed)
            if condition=="noise":
                spec=("noise","all",rng.uniform(.1,.3),0)
            else:
                spec=("spike","xyz"[rng.integers(3)],rng.uniform(3,6),int(rng.integers(8,25)))
        else:
            spec=DIAG[condition]
        typ,axis,amp,count=spec
        out,pos,signs=inject_fault(waves[r.Index],typ,axis,amp,count,seed,scale)
        changed.append(out)
        log.append(dict(outer_fold=fold,inner_id=inner,source=scope,file_id=r.file_id,window_idx=r.window_idx,
                        group_id=r.group_id,condition=condition,axis=axis,scale=amp,count=count,seed=seed,
                        positions=json.dumps(pos,separators=(",",":")),signs=json.dumps(signs,separators=(",",":")),
                        input_sha256=digest(out)))
    return changed


def fit_clean(train,waves,timings,fold,inner):
    raw=[waves[i] for i in train.index]
    scale=np.median([np.std(w,axis=0) for w in raw],axis=0)
    start=perf_counter()
    rocket=MiniRocket(n_kernels=10000,random_state=42,n_jobs=4)
    data=channel_first(raw)
    rocket.fit(data)
    fitted=perf_counter()
    features=rocket.transform(data)
    scaler=StandardScaler(with_mean=False).fit(features)
    x=scaler.transform(features).astype(np.float64)
    done=perf_counter()
    head=RidgeClassifier(alpha=1.).fit(x,train.y_true)
    timings.append(dict(fold=fold,inner=inner,stage="fit_clean",fit_transformer_s=fitted-start,
                        features_scaler_s=done-fitted,ridge_s=perf_counter()-done,n=len(train)))
    return dict(rocket=rocket,scaler=scaler,heads={"R0":head},scale=scale,
                feature_hash=representation_hash(rocket,scaler),n_features=x.shape[1],
                train_groups=sorted(map(int,train.group_id.unique())),versions=versions(),
                weights=[.5,.25,.25],class_order=[0,1]),x


def fit_aug(bundle,x,train,waves,fold,inner,log,timings):
    start=perf_counter()
    groups=[x]
    for condition in ("noise","impulse"):
        augmented=changed_windows(train,waves,bundle["scale"],"train_aug",fold,inner,condition,log)
        groups.append(transform(bundle,augmented))
    features_done=perf_counter()
    sample_weight=np.repeat(bundle["weights"],len(train))
    head=RidgeClassifier(alpha=1.).fit(np.concatenate(groups),np.tile(train.y_true,3),sample_weight=sample_weight)
    bundle["heads"]["R1"]=head
    if representation_hash(bundle["rocket"],bundle["scaler"])!=bundle["feature_hash"]:
        raise ValueError("The shared representation changed during augmentation")
    timings.append(dict(fold=fold,inner=inner,stage="fit_aug",features_scaler_s=features_done-start,
                        ridge_s=perf_counter()-features_done,n=len(train)*3))


def scored(meta,score,fold,inner,name,part,condition="clean"):
    return meta.assign(score=np.asarray(score,dtype=float),fold_id=fold,inner_id=inner,
                       model_id=name,partition=part,scenario=condition)


def profile(rows,threshold):
    d=rank_metrics(rows)
    pred=rows.score>=threshold
    d.update(n=len(rows),mean=rows.score.mean(),std=rows.score.std(ddof=0),min=rows.score.min(),
             q10=rows.score.quantile(.1),median=rows.score.median(),q90=rows.score.quantile(.9),max=rows.score.max(),
             threshold=threshold,alarm_rate=pred.mean(),actual_fpr=pred[rows.y_true==0].mean())
    return d


def add_diagnostics(rows,rate):
    clean=rows[rows.scenario=="clean"]
    cut=select_cut(clean.y_true,clean.score,rate)["threshold"]
    out=rows.assign(threshold=cut,prediction=(rows.score>=cut).astype(int))
    orig=out[out.scenario=="clean"][["file_id","window_idx","score","prediction"]].rename(
        columns={"score":"clean_score","prediction":"clean_prediction"})
    out=out.merge(orig,on=["file_id","window_idx"],validate="many_to_one")
    out["induced_error"]=out.clean_prediction.eq(out.y_true)&out.prediction.ne(out.y_true)
    out["resolved_error"]=out.clean_prediction.ne(out.y_true)&out.prediction.eq(out.y_true)
    return cut,out


def train_partition(meta,waves,split,cfg):
    tag=str(split["fold_id"]); fold=4 if tag=="full" else int(tag)
    records=[];logs=[];timings=[];stats=[];memory=[]
    for i,inner in enumerate(split["inner"]):
        tr=meta[meta.group_id.isin(inner["train"])]
        va=meta[meta.group_id.isin(inner["val"])]
        assert set(inner["train"]).isdisjoint(inner["val"]) and set(inner["train"]+inner["val"])==set(split["train"])
        b,x=fit_clean(tr,waves,timings,fold,i)
        start=perf_counter()
        val={"clean":transform(b,[waves[j] for j in va.index])}
        for cond in DIAG:
            v=changed_windows(va,waves,b["scale"],"inner_validation",fold,i,cond,logs)
            val[cond]=transform(b,v)
        ready=perf_counter()
        for cond,feat in val.items():
            records.append(scored(va,decisions(b["heads"]["R0"],feat),tag,i,"R0","inner_oof",cond))
        timings.append(dict(fold=fold,inner=i,stage="inner_valid",features_scaler_s=ready-start,
                            inference_s=perf_counter()-ready,n=len(va)*len(val)))
        memory.append((b,x,val,tr,va))
    t0,r0=add_diagnostics(pd.concat(records),cfg["target_fp_rate"])
    assert set(r0.group_id)==set(split["train"]) and not r0.duplicated(["file_id","window_idx","scenario"]).any()
    # Freeze/persist the observed R0 validation evidence before R1's supervised fit.
    where=RUN/f"fit_{tag}"
    where.mkdir(exist_ok=True)
    r0.to_csv(where/"r0_inner.csv.gz",index=False)
    print(f"{tag}: inner R0 induced errors = {r0.induced_error.sum()}",flush=True)
    records=[]
    cuts={"R0":t0}
    for i,(b,x,val,tr,va) in enumerate(memory):
        fit_aug(b,x,tr,waves,fold,i,logs,timings)
        b["test_groups"]=sorted(map(int,va.group_id.unique()))
        b["role"]="inner";b["fold_id"]=tag;b["inner_id"]=i
        for cond,feat in val.items():
            records.append(scored(va,decisions(b["heads"]["R1"],feat),tag,i,"R1","inner_oof",cond))
    cuts["R1"],r1=add_diagnostics(pd.concat(records),cfg["target_fp_rate"])
    r1.to_csv(where/"r1_inner.csv.gz",index=False)
    for i,(b,x,val,tr,va) in enumerate(memory):
        b["thresholds"]=cuts
        joblib.dump(b,where/f"inner_{i}.joblib",compress=3)
        for name in NAMES:
            s=scored(tr,decisions(b["heads"][name],x),tag,i,name,"inner_fit")
            stats.append(dict(model_id=name,partition="inner_fit",inner_id=i,**profile(s,cuts[name])))
    for g in (r0,r1):
        name=g.model_id.iloc[0]
        stats.append(dict(model_id=name,partition="inner_oof",inner_id=-1,**profile(g[g.scenario=="clean"],cuts[name])))
    del memory
    train=meta[meta.group_id.isin(split["train"])]
    b,x=fit_clean(train,waves,timings,fold,3)
    fit_aug(b,x,train,waves,fold,3,logs,timings)
    b.update(fold_id=tag,test_groups=split.get("test",[]),role="full" if tag=="full" else "outer",thresholds=cuts,
             protocol_sha256=file_hash(PROTOCOL),settings=cfg["model"])
    for name in NAMES:
        s=scored(train,decisions(b["heads"][name],x),tag,-1,name,"refit_train")
        stats.append(dict(model_id=name,partition="refit_train",inner_id=-1,**profile(s,cuts[name])))
    joblib.dump(b,RUN/f"models/{tag}.joblib",compress=3)
    pd.DataFrame(logs).to_csv(where/"perturbations.csv.gz",index=False)
    pd.DataFrame(stats).assign(fold_id=tag).to_csv(where/"scores.csv",index=False)
    write_json(where/"fit.json",dict(split=split,thresholds=cuts,features=b["n_features"],scale=b["scale"].tolist(),
                                   feature_sha256=b["feature_hash"],timings=timings))
    print(tag,"refit complete",cuts,flush=True)


def main():
    cfg=json.loads(PROTOCOL.read_text(encoding="utf8"))
    if cfg["versions"]!=versions() or file_hash(ROOT/"data/raw/manifest.json")!=cfg["dataset_sha256"]:
        raise ValueError("Run environment or dataset differs from frozen protocol")
    for name,sha in cfg["code_sha256"].items():
        if file_hash(ROOT/name)!=sha:
            raise ValueError("Training recipe source changed: "+name)
    meta,w,_=load_dataset(ROOT/"data/raw")
    with threadpool_limits(limits=2,user_api="blas"):
        for split in cfg["splits"]:
            train_partition(meta,w,split,cfg)


if __name__=="__main__":
    main()
