"""Account for both correct alarms and errors hidden by review; no detector training."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from .hi_study import ROOT, OUT, DEST, make_wave
from .data import load_dataset
from .coverage import gate_inputs

SAFETY=OUT/"safety_support"
KEY=["file_id","window_idx","fold_id"]
CONFIG=json.loads((SAFETY/"protocol.json").read_text(encoding="utf-8"))
SEEDS=tuple(CONFIG["random_seeds"])
PEERS={"A0":"H","A1":"H","H":"A1","I":"A1"}
IDENT=["model_id","scenario","mode","policy","seed"]
COUNT_COLS=["N","N_S","N_N","N_E","N_B","TP_pass","FN_pass","FP_pass","TN_pass",
            "H_TP","H_FN","H_FP","H_TN","pass_count","hold_count","S_pass","E_pass","B_pass"]
RATIOS={"fn_pass_rate":("FN_pass","N_S"),"tp_pass_rate":("TP_pass","N_S"),
        "fp_pass_rate":("FP_pass","N_N"),"coverage":("pass_count","N"),"coverage_S":("S_pass","N_S"),
        "coverage_E":("E_pass","N_E"),"coverage_B":("B_pass","N_B"),"coverage_nonS":("nonS_pass","N_N")}


def priority(score,threshold,peer_prediction):
    """Only detector outputs are inputs. Disagreement has priority over distance."""
    margin=1/(1+np.abs(score-threshold))
    return {"margin":margin,"proposed":margin+((score>=threshold)!=peer_prediction).astype(float)}


def lowest_80(values):
    # Input arrives in fixed record order, which is also the final tie break.
    n=len(values)
    keep=np.zeros(n,dtype=bool)
    keep[np.argsort(values,kind="stable")[:int(n*.8)]]=True
    return keep


def random_decisions(n,fold,seed,mode):
    r=np.random.default_rng(np.random.SeedSequence([87011,int(seed),int(fold)])).random(n)
    return lowest_80(r) if mode=="matched" else r<=.8


def base_views(rows):
    return rows[["file_id","window_idx","group_id","fold_id","phase","y_true","model_id","scenario","score","threshold","prediction"]].copy()


def hi_cases():
    inner=pd.read_csv(DEST/"inner_oof_clean.csv",float_precision="round_trip")
    pred=pd.read_csv(DEST/"predictions.csv",float_precision="round_trip")
    res=[]; calibration=[]
    for fold,iv in inner.groupby("outer_fold"):
        i_scores=iv.pivot(index=["file_id","window_idx"],columns="model_id",values="score")
        i_cuts=iv.groupby("model_id").threshold.first().to_dict()
        i_scores["A0"]=i_scores.A1;i_cuts["A0"]=.5
        cuts={}
        for model,peer in PEERS.items():
            risk=priority(i_scores[model].to_numpy(),i_cuts[model],(i_scores[peer]>=i_cuts[peer]).to_numpy())
            for name,v in risk.items():
                cuts[model,name]=float(np.quantile(v,.8))
                calibration.append(dict(model_id=model,fold_id=int(fold),policy=name,peer=peer,
                                        train_cut=cuts[model,name],source="inner_oof_clean"))
        for _,run in pred[pred.fold_id==fold].groupby("scenario"):
            aligned=run.pivot(index=["file_id","window_idx"],columns="model_id",values="prediction").sort_index()
            for model,peer in PEERS.items():
                t=base_views(run[run.model_id==model].sort_values(["file_id","window_idx"])).reset_index(drop=True)
                if not pd.MultiIndex.from_frame(t[["file_id","window_idx"]]).equals(aligned.index):
                    raise ValueError("Model/peer rows do not match")
                r=priority(t.score.to_numpy(),t.threshold.to_numpy(),aligned[peer].to_numpy())
                for name,v in r.items():
                    res.append(t.assign(policy=name,risk_score=v,cut=cuts[model,name],
                                        pass_matched=lowest_80(v),pass_calibrated=v<=cuts[model,name]))
    return pd.concat(res,ignore_index=True),calibration


def additional_e_z():
    """Only missing legacy E/Z cell. Frozen RF, learned gate and cuts; no fit operation."""
    meta,w,_=load_dataset(ROOT/"data/raw")
    parts=[]
    for fold in range(4):
        b=joblib.load(ROOT/f"artifacts/fold_{fold}.joblib")
        saved=joblib.load(OUT/f"models/gate_{fold}.joblib")
        t=meta[meta.group_id.isin(b["test_groups"])]
        waves=[make_wave(w[r.Index],b["scale"],r,"new_z_spike") for r in t.itertuples()]
        q,h=gate_inputs(b["rf"],waves)
        for name,v in [("margin80",-h[:,0]),("risk80",saved["policy"].predict_proba(h)[:,1])]:
            cut=saved["cuts"][name]
            parts.append(t.assign(fold_id=fold,scenario="new_z_spike",policy=name,model_score=q,
                                  risk_score=v,calibration_cut=cut,pass_matched=lowest_80(v),pass_calibrated=v<=cut))
    return pd.concat(parts,ignore_index=True)


def legacy_cases():
    parts=[]
    for model,file in [("E","gate_coverage.csv"),("G","gate_coverage_G.csv")]:
        t=pd.read_csv(OUT/file,float_precision="round_trip")
        if model=="E":t=pd.concat([t,additional_e_z()],ignore_index=True)
        t=t.rename(columns={"model_score":"score","calibration_cut":"cut"})
        t["policy"]=t.policy.map({"margin80":"margin","risk80":"proposed"})
        t=t.assign(model_id=model,threshold=.5,prediction=(t.score>=.5).astype(int))
        parts.append(t[list(base_views(t).columns)+["policy","risk_score","cut","pass_matched","pass_calibrated"]])
    return pd.concat(parts,ignore_index=True)


def with_rates(summary):
    summary["nonS_pass"]=summary.E_pass+summary.B_pass
    for rate,(num,den) in RATIOS.items():
        summary[rate]=summary[num]/summary[den].replace(0,np.nan)
    return summary


def counts_by_group(sub,passed):
    """One array per record, one row per group. A held positive remains H_TP/H_FN."""
    y=sub.y_true.to_numpy(dtype=bool);p=sub.prediction.to_numpy(dtype=bool)
    phase=sub.phase.to_numpy()
    e=phase=="E";b=phase=="B";hold=~passed
    terms=[np.ones(len(sub),bool),y,~y,e,b,passed&y&p,passed&y&~p,passed&~y&p,passed&~y&~p,
           hold&y&p,hold&y&~p,hold&~y&p,hold&~y&~p,passed,hold,passed&y,passed&e,passed&b]
    groups=sub.group_id.to_numpy(dtype=int)
    return np.stack([np.bincount(groups,weights=v,minlength=17)[1:17] for v in terms],axis=1).astype(int)


def summarize_trials(cases):
    arrays=[];indices=[]
    def add(g,passed,meta):
        arrays.append(counts_by_group(g,passed));indices.extend([(*meta,group) for group in range(1,17)])
    for (model,cond),run in cases.groupby(["model_id","scenario"]):
        main=run[run.policy=="margin"].sort_values(KEY).reset_index(drop=True)
        assert len(main)==480 and not main.duplicated(KEY).any()
        add(main,np.ones(len(main),bool),(model,cond,"reference","none",-1))
        for policy,g in run.groupby("policy"):
            g=g.sort_values(KEY).reset_index(drop=True)
            assert g[KEY+["score","prediction","y_true"]].equals(main[KEY+["score","prediction","y_true"]])
            for mode in ("matched","calibrated"):
                mask=g["pass_"+mode].to_numpy(bool)
                add(g,mask,(model,cond,mode,policy,-1))
        fold_indices=[(f,v.index.to_numpy()) for f,v in main.groupby("fold_id")]
        for seed in SEEDS:
            for mode in ("matched","calibrated"):
                mask=np.zeros(len(main),bool)
                for f,idx in fold_indices:
                    # No phase, truth, fault identity, model score or scenario is passed.
                    mask[idx]=random_decisions(len(idx),f,seed,mode)
                add(main,mask,(model,cond,mode,"random",seed))
    bygroup=pd.DataFrame(np.concatenate(arrays),columns=COUNT_COLS,index=pd.MultiIndex.from_tuples(indices,names=IDENT+["group_id"])).reset_index()
    total=bygroup.groupby(IDENT,as_index=False)[COUNT_COLS].sum()
    return with_rates(total),with_rates(bygroup)


def bands(trials,keys):
    rnd=trials[trials.policy=="random"]
    metrics=COUNT_COLS+list(RATIOS)
    agg=rnd.groupby(keys)[metrics].agg(["mean","std","min","max"])
    agg.columns=["_".join(k) for k in agg.columns]
    return agg.reset_index()


def build():
    cases,calibration=hi_cases()
    cases=pd.concat([cases,legacy_cases()],ignore_index=True)
    cases.to_csv(SAFETY/"cases.csv.gz",index=False)
    pd.DataFrame(calibration).to_csv(SAFETY/"training_cuts.csv",index=False)
    total,groups=summarize_trials(cases)
    total.to_csv(SAFETY/"all_runs.csv.gz",index=False)
    groups.to_csv(SAFETY/"group_runs.csv.gz",index=False)
    total[total.policy!="random"].to_csv(SAFETY/"policies.csv",index=False)
    keys=["model_id","scenario","mode"]
    bands(total,keys).to_csv(SAFETY/"random_summary.csv",index=False)
    bands(groups,keys+["group_id"]).to_csv(SAFETY/"random_groups.csv.gz",index=False)
    report=total.query("model_id in ['I','E','G'] and scenario in ['clean','spike'] and mode!='calibrated' and policy!='random'")
    print(report[["model_id","scenario","mode","policy","TP_pass","FN_pass","FP_pass","H_TP","H_FN","pass_count"]].to_string(index=False))

if __name__=="__main__":build()
