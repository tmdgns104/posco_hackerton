"""Limited H/I experiment: evaluate after inner selection, never select a threshold on outer labels."""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from .hi_audit import DEST,counts
from .hi_models import spectral36,robust36,scores,predict_matrix
from .improve import ROOT,OUT,signal
from .multi_axis import z_check
from .data import load_dataset
from .quality import quality

CONFIG=DEST/"protocol.json"
MODELS=("A0","A1","H","I")


def read_config():
    return json.loads(CONFIG.read_text(encoding="utf8"))


def select_cut(y,s,rate):
    y=np.asarray(y);s=np.asarray(s)
    assert np.isfinite(s).all() and set(y)=={0,1}
    candidates=np.r_[np.unique(s),np.nextafter(s.max(),np.inf)]
    options=[]
    for t in candidates:
        p=s>=t;fp=int((p&(y==0)).sum())
        if fp<=np.floor(rate*(y==0).sum()):
            options.append((int((p&(y==1)).sum()),-fp,float(t)))
    tp,neg_fp,t=max(options)
    return dict(threshold=t,tp=tp,fp=-neg_fp,n_S=int(y.sum()),n_nonS=int((y==0).sum()))


def calibrate(train,x,estimator,inner,rate):
    oof=np.full(len(train),np.nan)
    for split in inner:
        fit=train.group_id.isin(split["train"]).to_numpy();valid=train.group_id.isin(split["val"]).to_numpy()
        if np.any(fit&valid) or not np.all(fit|valid) or np.isfinite(oof[valid]).any():raise ValueError("inner leakage")
        m=clone(estimator).fit(x[fit],train.y_true.to_numpy()[fit])
        oof[valid]=scores(m,x[valid])
    return select_cut(train.y_true,oof,rate),oof


def make_wave(w,scale,row,condition):
    return z_check(w,scale,row) if condition=="new_z_spike" else signal(w,scale,row,condition)


def summarize(g):
    c=counts(g.y_true,g.prediction);tp,fn,fp,tn=(c[k] for k in ("tp","fn","fp","tn"))
    correct=g.prediction==g.y_true
    old=g.A1==g.y_true
    return dict(**c,n=len(g),recall=tp/(tp+fn),fpr=fp/(fp+tn),
        precision=tp/(tp+fp) if tp+fp else None,balanced_accuracy=(tp/(tp+fn)+tn/(fp+tn))/2,
        fp_E=int(((g.phase=="E")&(g.prediction==1)).sum()),fp_B=int(((g.phase=="B")&(g.prediction==1)).sum()),
        fixed_vs_A1=int((~old&correct).sum()),new_error_vs_A1=int((old&~correct).sum()))


def aggregate(rows,keys):
    return pd.DataFrame([dict(zip(keys,k),**summarize(g)) for k,g in rows.groupby(keys,sort=True)])


def decision(summary,fold_summary,c):
    rules=c["selection"];report={}
    for model in c["candidates"]:
        cand=summary[summary.model_id==model].set_index("scenario")
        base=summary[summary.model_id=="A1"].set_index("scenario")
        clean=cand.loc["clean"];ref=base.loc["clean"]
        ft=fold_summary[fold_summary.scenario=="clean"].pivot(index="fold_id",columns="model_id",values="tp")
        delta=ft[model]-ft.A1
        stress=(cand[["tp","fp"]]-base[["tp","fp"]]).drop(index="clean")
        checks=dict(clean_extra_tp=bool(clean.tp-ref.tp>=rules["min_clean_extra_tp"]),
                    clean_no_extra_fp=bool(clean.fp<=ref.fp),clean_fp_limit=bool(clean.fpr<=rules["max_fp_rate"]),
                    clean_fold_spread=bool((delta>=0).sum()>=rules["min_nondeclining_folds"] and (delta>0).sum()>=rules["min_improving_folds"]),
                    stress_recall=bool((stress.tp>=-rules["max_stress_lost_tp"]).all()),
                    stress_fpr=bool((stress.fp<=rules["max_stress_extra_fp"]).all()))
        report[model]=dict(checks=checks,eligible=all(checks.values()),fold_tp_delta=delta.to_dict(),
            bad_recall_conditions=stress.index[stress.tp< -rules["max_stress_lost_tp"]].tolist(),
            bad_fp_conditions=stress.index[stress.fp>rules["max_stress_extra_fp"]].tolist())
    eligible=[k for k,v in report.items() if v["eligible"]]
    clean=summary[(summary.scenario=="clean")&summary.model_id.isin(eligible)].sort_values(["tp","fp","model_id"],ascending=[False,True,True])
    return dict(adopt=clean.iloc[0].model_id if len(clean) else None,reference="A1",candidates=report)


def run():
    c=read_config();meta,w,_=load_dataset(ROOT/"data/raw")
    xx=np.stack([spectral36(v) for v in w]);xi=np.stack([robust36(v,c["features_I"]) for v in w])
    modeldir=DEST/"models";modeldir.mkdir(exist_ok=True)
    predictions=[];cuts=[];inner_rows=[]
    for fold in c["folds"]:
        f=fold["fold_id"];path=ROOT/f"artifacts/fold_{f}.joblib"
        if hashlib.sha256(path.read_bytes()).hexdigest()!=fold["base_sha256"]:raise ValueError("v4 model hash mismatch")
        b=joblib.load(path)
        tr=meta[meta.group_id.isin(fold["train"])];te=meta[meta.group_id.isin(fold["test"])]
        assert len(tr)==360 and len(te)==120 and set(tr.group_id).isdisjoint(te.group_id)
        np.testing.assert_allclose(b["scale"],np.median([w[i].std(axis=0) for i in tr.index],axis=0),rtol=1e-12)
        models={"A0":dict(estimator=b["spectral"],threshold=.5,representation="original")}
        for model,x in [("A1",xx),("H",xx),("I",xi)]:
            proto=b["spectral"] if model=="A1" else make_pipeline(StandardScaler(),SVC(**c["svm"]))
            train_x=x[tr.index]
            calibration,inner_scores=calibrate(tr,train_x,proto,fold["inner"],c["target_fp_rate"])
            cuts.append(dict(fold_id=f,model_id=model,**calibration))
            inner_rows.append(tr.assign(outer_fold=f,model_id=model,score=inner_scores,threshold=calibration["threshold"]))
            estimator=proto if model=="A1" else clone(proto).fit(train_x,tr.y_true)
            models[model]=dict(estimator=estimator,threshold=calibration['threshold'],representation="robust" if model=="I" else "original")
        cuts.append(dict(fold_id=f,model_id="A0",threshold=.5))
        joblib.dump(dict(models=models,train_groups=fold['train'],test_groups=fold['test'],scale=b["scale"]),modeldir/f"fold_{f}.joblib",compress=3)
        for condition in c["conditions"]:
            waves=[make_wave(w[r.Index],b["scale"],r,condition) for r in te.itertuples()]
            if any(quality(v) for v in waves): raise ValueError("Unexpected invalid input in fixed population")
            hashes=[hashlib.sha256(v.tobytes()).hexdigest() for v in waves]
            x0=xx[te.index] if condition=="clean" else np.stack([spectral36(v) for v in waves])
            x1=xi[te.index] if condition=="clean" else np.stack([robust36(v,c["features_I"]) for v in waves])
            for model,bundle in models.items():
                score,p=predict_matrix(bundle,x1 if model=="I" else x0)
                predictions.append(te.assign(fold_id=f,scenario=condition,model_id=model,score=score,threshold=bundle["threshold"],
                                             prediction=p,input_sha256=hashes))
    frame=pd.concat(predictions,ignore_index=True)
    keys=["file_id","window_idx","scenario","fold_id"]
    ref=frame[frame.model_id=="A1"][keys+["prediction"]].rename(columns={"prediction":"A1"})
    frame=frame.merge(ref,on=keys,validate="many_to_one")
    frame.to_csv(DEST/"predictions.csv",index=False)
    pd.DataFrame(cuts).to_csv(DEST/"thresholds.csv",index=False)
    pd.concat(inner_rows).to_csv(DEST/"inner_oof_clean.csv",index=False)
    keys=["model_id","scenario"]
    summary=aggregate(frame,keys)
    fold_sum=aggregate(frame,keys+["fold_id"])
    summary.to_csv(DEST/"summary.csv",index=False);fold_sum.to_csv(DEST/"by_fold.csv",index=False)
    aggregate(frame,keys+["group_id"]).to_csv(DEST/"by_group.csv",index=False)
    verdict=decision(summary,fold_sum,c)
    (DEST/"adoption.json").write_text(json.dumps(verdict,indent=2))
    print(summary[["scenario","model_id","tp","fn","fp","tn","fp_E","fp_B","fixed_vs_A1","new_error_vs_A1"]].to_string(index=False))
    print(json.dumps(verdict,indent=2))
if __name__=="__main__":run()
