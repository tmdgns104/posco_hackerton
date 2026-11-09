"""One decision per complete 5-second recording, with the predeclared clean stop."""
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_dataset
from .hi_study import ROOT, make_wave, read_config, select_cut

RUN=ROOT/"experiments/temporal5s"
META=dict(task_id="ssfd_file5s",sample_unit="5s_file",evaluation="development_group_cv",n_files=48)


def vector(w):
    windows=np.asarray(w).reshape(10,3200,3)
    rms=np.sqrt(((windows-windows.mean(axis=1,keepdims=True))**2).mean(axis=1))
    avg=rms.mean(axis=0)
    if not np.all(avg>0): raise ValueError("Invalid AC amplitude")
    f,p=welch(windows.reshape(32000,3),fs=6400,window="hann",nperseg=1024,noverlap=512,
              detrend="constant",scaling="density",axis=0)
    bands=np.add.reduceat(p,np.searchsorted(f,[0,25,100,400,800,1600]),axis=0)
    return np.r_[np.log(avg),rms.std(axis=0,ddof=1)/avg,np.log(bands/bands.sum(axis=0)+1e-8).T.ravel()]


def clf():
    return make_pipeline(StandardScaler(),LogisticRegression(C=.1,solver="lbfgs",class_weight="balanced",max_iter=2000))


def j_means(df):
    if not (df.groupby("file_id").window_idx.nunique().eq(10).all() and
            not df.duplicated(["file_id","window_idx"]).any()):raise ValueError("Incomplete J windows")
    return df.groupby("file_id").score.mean()


def run():
    cfg=read_config();meta,w,_=load_dataset(ROOT/"data/raw")
    groups=meta.groupby("file_id",sort=True)
    rows=groups.first().reset_index()[["file_id","phase","group_id","y_true"]]
    source={name:np.stack([w[i] for i in part.sort_values("window_idx").index]) for name,part in groups}
    features=np.stack([vector(source[n]) for n in rows.file_id])
    inner=pd.read_csv(ROOT/"experiments/detector_v1/inner_oof.csv.gz",float_precision="round_trip").query("model_id=='J'")
    outer=pd.read_csv(ROOT/"experiments/detector_v1/predictions.csv",float_precision="round_trip").query("model_id=='J' and budget==.15")
    windows_meta={name:part.sort_values("window_idx") for name,part in groups}
    fits={};inner_rows=[]
    for split in cfg["folds"]:
        f=split["fold_id"];tr=rows.group_id.isin(split["train"])
        scorej=j_means(inner[inner.fold_id==f]);assert set(scorej.index)==set(rows.file_id[tr])
        bundle={"train":split["train"],"test":split["test"],"models":{}}
        for name,n in [("T1",6),("T2",24),("J_file5s",0)]:
            oof=np.full(len(rows),np.nan)
            if n:
                for iv in split["inner"]:
                    a=rows.group_id.isin(iv["train"]);b=rows.group_id.isin(iv["val"])
                    assert not (a&b).any() and (a|b).equals(tr)
                    oof[b]=clf().fit(features[a,:n],rows.y_true[a]).decision_function(features[b,:n])
                model=clf().fit(features[tr,:n],rows.y_true[tr])
            else:
                oof[tr]=rows.file_id[tr].map(scorej);model=None
            cut=select_cut(rows.y_true[tr],oof[tr],.15)["threshold"]
            bundle["models"][name]=(model,n,cut)
            inner_rows.append(rows[tr].assign(model_id=name,fold_id=f,score=oof[tr],threshold=cut,**META))
        joblib.dump(bundle,RUN/f"models/fold_{f}.joblib",compress=3);fits[f]=bundle
    outputs=[];summaries=[];changes=[]
    for cond in cfg["conditions"]:
        chunks=[]
        for f,bundle in fits.items():
            te=rows[rows.group_id.isin(bundle["test"])]
            j=j_means(outer[(outer.fold_id==f)&(outer.scenario==cond)])
            scale=joblib.load(ROOT/f"experiments/detector_v1/models/fold_{f}.joblib")["scale"]
            arr=[np.stack([make_wave(source[n][r.window_idx],scale,r,cond) for r in windows_meta[n].itertuples()]) for n in te.file_id]
            x=np.stack([vector(v) for v in arr]); hashes=[hashlib.sha256(v.tobytes()).hexdigest() for v in arr]
            for name,(model,n,cut) in bundle["models"].items():
                s=model.decision_function(x[:,:n]) if n else te.file_id.map(j).to_numpy()
                chunks.append(te.assign(model_id=name,fold_id=f,score=s,threshold=cut,prediction=(s>=cut).astype(int),
                                        scenario=cond,input_sha256=hashes,**META))
        frame=pd.concat(chunks);outputs.append(frame)
        for name,g in frame.groupby("model_id"):
            s=g.y_true==1;p=g.prediction==1
            summaries.append(dict(**META,model_id=name,scenario=cond,tp=(s&p).sum(),fn=(s&~p).sum(),fp=(~s&p).sum(),tn=(~s&~p).sum(),
                                  fp_E=(p&g.phase.eq("E")).sum(),fp_B=(p&g.phase.eq("B")).sum()))
        tab=frame.pivot(index=["file_id","group_id","y_true"],columns="model_id",values="prediction").reset_index()
        a=tab.T1.eq(1);b=tab.T2.eq(1);s=tab.y_true.eq(1)
        changes.append(dict(**META,scenario=cond,rescued_FN=(s&~a&b).sum(),new_FN=(s&a&~b).sum(),
                            removed_FP=(~s&a&~b).sum(),new_FP=(~s&~a&b).sum()))
        if cond=="clean":
            st=pd.DataFrame(summaries).set_index("model_id")
            t=st.loc["T2"];ref=st.loc["T1"]
            if not (t.tp>=12 and t.fp<=4 and t.tp>ref.tp and t.fp<=ref.fp):break
    pred=pd.concat(outputs);pred.to_csv(RUN/"predictions.csv",index=False)
    summary=pd.DataFrame(summaries);summary.to_csv(RUN/"summary.csv",index=False)
    gs=pred.assign(tp=(pred.y_true==1)&(pred.prediction==1),fn=(pred.y_true==1)&(pred.prediction==0),
                   fp=(pred.y_true==0)&(pred.prediction==1),tn=(pred.y_true==0)&(pred.prediction==0))
    gs.groupby(["group_id","model_id","scenario"])[["tp","fn","fp","tn"]].sum().reset_index().assign(**META,n_group_files=3).to_csv(RUN/"groups.csv",index=False)
    pd.concat(inner_rows).to_csv(RUN/"inner_oof.csv",index=False)
    pd.DataFrame(changes).to_csv(RUN/"changes.csv",index=False)
    stress=summary.query("model_id=='T2' and scenario!='clean'")
    passed=len(stress)==4 and bool((stress.tp.ge(10)&stress.fp.le(6)).all())
    (RUN/"decision.json").write_text(json.dumps(dict(passed=passed,stress_evaluated=bool(len(stress)),next_task="ssfd_file5s" if passed else "neu_cls6")))
    print(summary.to_string(index=False));print(changes)
if __name__=="__main__":run()
