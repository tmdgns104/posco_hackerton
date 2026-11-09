"""Evaluate frozen checkpoints and score each model pair on identical raw arrays."""
import json
import platform
from importlib.metadata import version
from .pair_runs import CNN_PAIRS as PAIRS

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

from .data import load_dataset
from .detector import RUN as OLD_RUN
from .hi_study import read_config, make_wave, ROOT
from .pair_detector import detector,digest,file_hash,identifier
from .raw_cnn import RUN, CFG

NAMES=("A1","J","C0","C1")
KEY=["fold_id","file_id","window_idx","scenario"]


def metrics(g):
    y=g.y_true.eq(1);p=g.prediction.eq(1)
    ns,nn=int(y.sum()),int((~y).sum())
    tp,fp=int((y&p).sum()),int((~y&p).sum())
    return dict(n=len(g),n_S=ns,n_nonS=nn,tp=tp,fn=ns-tp,fp=fp,tn=nn-fp,recall=tp/ns,fpr=fp/nn,
                precision=tp/(tp+fp) if tp+fp else np.nan,fp_B=int((p & g.phase.eq("B")).sum()),
                fp_E=int((p & g.phase.eq("E")).sum()))


def rank_metrics(g):
    out=dict(auc=roc_auc_score(g.y_true,g.score),ap=average_precision_score(g.y_true,g.score))
    for phase in "BE":
        pair=g[g.phase.isin(["S",phase])]
        out["auc_S"+phase]=roc_auc_score(pair.y_true,pair.score)
    return out


def paired(g,before,after):
    left=g[g.model_id==before];right=g[g.model_id==after]
    sub=left.merge(right,on=KEY,suffixes=("_before","_after"),validate="one_to_one")
    if len(sub)!=len(left) or len(sub)!=len(right) or not sub.input_sha256_before.eq(sub.input_sha256_after).all():
        raise ValueError("모델 쌍의 구간 또는 입력 해시 불일치")
    assert sub.y_true_before.eq(sub.y_true_after).all()
    a=sub.prediction_before.eq(1);b=sub.prediction_after.eq(1);s=sub.y_true_before.eq(1)
    sub["change"]=np.select([s&~a&b,s&a&~b,~s&a&~b,~s&~a&b],
                            ["rescued_FN","new_FN","removed_FP","new_FP"],default="unchanged")
    return sub


def run():
    cfg=read_config();meta,w,_=load_dataset(ROOT/"data/raw")
    ref=pd.read_csv(OLD_RUN/"predictions.csv",float_precision="round_trip").query("budget==.15 and model_id in ['A1','J']")
    source_sha=file_hash(ROOT/"data/raw/manifest.json")
    preds=[]; diagnostics=[]; checkpoint_ids=[]
    for split in cfg["folds"]:
        f=split["fold_id"]
        te=meta[meta.group_id.isin(split["test"])]
        tr=meta[meta.group_id.isin(split["train"])]
        base=detector("J",f)
        for name in NAMES:
            d=detector(name,f)
            assert set(d.test)==set(split["test"]) and set(d.train)==set(split["train"])
            checkpoint_ids.append(d.identity)
            train=tr.assign(score=[r["score"] for r in d.infer([w[i] for i in tr.index])])
            diagnostics.append(dict(fold_id=f,model_id=name,partition="train",**rank_metrics(train)))
        for scenario in cfg["conditions"]:
            raw=[make_wave(w[r.Index],base.legacy["scale"],r,scenario) for r in te.itertuples()]
            hashes=[digest(v) for v in raw]
            for name in NAMES:
                d=detector(name,f)
                out=pd.DataFrame(d.infer(raw))
                if not out.input_status.eq("VALID").all():
                    raise ValueError("An input in the fixed population did not produce a prediction")
                block=te.assign(model_id=name,fold_id=f,scenario=scenario,input_sha256=hashes,model_run_id=d.run_id,
                                score=out.score.to_numpy(),threshold=d.threshold,prediction=out.prediction.to_numpy())
                preds.append(block)
                if scenario=="clean":
                    diagnostics.append(dict(fold_id=f,model_id=name,partition="outer",**rank_metrics(block)))
    for name in ("C0","C1"):
        checkpoint_ids.append(detector(name,"full").identity)
    predictions=pd.concat(preds,ignore_index=True)
    match=predictions.merge(ref,on=KEY+["model_id"],validate="one_to_one")
    assert len(match)==len(ref)==4800
    for col in ("input_sha256","threshold","prediction"):
        np.testing.assert_array_equal(match[col+"_x"],match[col+"_y"])
    np.testing.assert_allclose(match.score_x,match.score_y,rtol=0,atol=1e-12)
    predictions.to_csv(RUN/"predictions.csv",index=False)
    p=pd.read_csv(RUN/"predictions.csv",float_precision="round_trip")
    pd.DataFrame(diagnostics).to_csv(RUN/"train_outer_auc.csv",index=False)
    for suffix,extra in [("summary",[]),("by_fold",["fold_id"]),("by_group",["fold_id","group_id"])]:
        rows=[]
        for keys,sub in p.groupby(["model_id","scenario"]+extra):
            vals=metrics(sub)
            if extra==["fold_id"]:
                vals.update(rank_metrics(sub))
            if not extra:
                assert vals["n_S"]==160 and vals["n_nonS"]==320
                ranks=pd.DataFrame([rank_metrics(g) for _,g in sub.groupby("fold_id")]).mean()
                vals.update({"mean_fold_"+k:v for k,v in ranks.items()})
            rows.append(dict(zip(["model_id","scenario"]+extra,keys),**vals))
        pd.DataFrame(rows).to_csv(RUN/(suffix+".csv"),index=False)
    changes=[]
    for before,after in PAIRS:
        sub=paired(p,before,after)
        for scenario,g in sub.groupby("scenario"):
            changes.append(dict(before=before,after=after,scenario=scenario,
                                **{k:int(g.change.eq(k).sum()) for k in ("rescued_FN","new_FN","removed_FP","new_FP")}))
    pd.DataFrame(changes).to_csv(RUN/"changes.csv",index=False)
    summary=pd.read_csv(RUN/"summary.csv")
    comparison=pd.DataFrame(changes)
    for side in ("before","after"):
        comparison=comparison.merge(summary.add_suffix("_"+side),left_on=[side,"scenario"],
                                    right_on=["model_id_"+side,"scenario_"+side],validate="many_to_one")
    comparison.to_csv(RUN/"pair_summary.csv",index=False)
    clean=summary[summary.scenario=="clean"].set_index("model_id")
    verdict={}
    for name,g in summary.groupby("model_id"):
        checks={}
        for r in g.itertuples():
            target=CFG["goals"]["clean" if r.scenario=="clean" else "stress"]
            checks[r.scenario]=bool(r.tp>=target["min_tp"] and r.fp<=target["max_fp"])
        verdict[name]=dict(conditions=checks,all_pass=all(checks.values()))
    for name in ("C0","C1"):
        verdict[name]["adopt"]=bool(verdict[name]["all_pass"] and clean.loc[name,"tp"]>clean.loc["J","tp"] and
                                    clean.loc[name,"fp"]<=clean.loc["J","fp"] and
                                    (summary.query("model_id==@name and scenario!='clean'").tp.to_numpy() >
                                     summary.query("model_id=='J' and scenario!='clean'").tp.to_numpy()).all())
    (RUN/"verdict.json").write_text(json.dumps(verdict,indent=2))
    env={p:version(p) for p in ("torch","numpy","scikit-learn","streamlit","pandas","scipy")}
    (RUN/"manifest.json").write_text(json.dumps(dict(dataset_manifest=source_sha,versions=env,os=platform.system(),
                       python=platform.python_version(),device="cpu",protocol_sha256=file_hash(RUN/"protocol.json"),
                       models=checkpoint_ids,predictions_sha256=file_hash(RUN/"predictions.csv"),
                       report_hashes={n:file_hash(RUN/n) for n in ["pair_summary.csv","summary.csv","changes.csv","verdict.json"]}),indent=2))
    evidence=[]
    meta["rms"]=[np.sqrt(np.mean(x*x)) for x in w]
    for f in range(4):
        inner=pd.read_csv(RUN/f"inner_{f}.csv").query("model_id=='C0'").merge(meta[["file_id","window_idx","rms"]],on=["file_id","window_idx"])
        for k,g in inner.groupby("inner_id"):
            evidence.append(dict(fold=f,inner_id=k,rms_score_spearman=spearmanr(g.rms,g.score).statistic,
                                 rms_q90_q10=g.rms.quantile(.9)/g.rms.quantile(.1)))
    pd.DataFrame(evidence).to_csv(RUN/"inner_gain_evidence.csv",index=False)
    print(summary[["model_id","scenario","tp","fn","fp","tn"]].to_string(index=False))


if __name__=="__main__":
    run()
