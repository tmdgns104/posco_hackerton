"""Evaluate on every held-out window under each fixed scenario; search trials are a separate population."""
import json
import platform
from pathlib import Path
import pandas as pd
from .gate import apply_gate,MARGIN,MARGINS
from .search import ROOT,load_context,digest


def ratio(n,d):
    return n/d if d else None


def evaluate_rows(frame,model,policy,margin):
    other="spectral" if model=="rf" else "rf"
    rows=[]
    for row in frame.to_dict("records"):
        reason=row["reason_codes"] if pd.notna(row["reason_codes"]) else ""
        dec,why,forward=apply_gate(reason,row[model+"_score"],row[other+"_pred"],policy,margin)
        rows.append(dict(decision=dec,reasons=" / ".join(why),forwarded_pred=forward))
    return pd.concat([frame.reset_index(drop=True),pd.DataFrame(rows)],axis=1)


def compute_gate_metrics(frame,model):
    y=frame.y_true
    p=frame[model+"_pred"]
    valid=p.notna()
    passed=frame.decision=="PASS"
    held=frame.decision=="HOLD"
    wrong=p.ne(y)&valid
    ok=p.eq(y)&valid
    n=len(y)
    fp=int((passed & wrong & (y==0)).sum())
    fn=int((passed & wrong & (y==1)).sum())
    caught=int((held&wrong).sum())
    extra=int((held&ok).sum())
    forwarded=int(passed.sum())
    pos=int(y.sum())
    return dict(N=n,N_S=pos,N_nonS=n-pos,TP=int((passed&ok&(y==1)).sum()),TN=int((passed&ok&(y==0)).sum()),FP=fp,FN=fn,
                HOLD_S=int((held&(y==1)).sum()),HOLD_nonS=int((held&(y==0)).sum()),no_prediction=int((frame.decision=="NO_PREDICTION").sum()),
                quality_held=int((held&~valid).sum()),wrong=int(wrong.sum()),correct=int(ok.sum()),caught=caught,
                unnecessary_hold=extra,passed=forwarded,error_pass_rate=ratio(fp+fn,n),block_rate=ratio(caught,int(wrong.sum())),
                unnecessary_rate=ratio(extra,int(ok.sum())),coverage=ratio(forwarded,n),conditional_error=ratio(fp+fn,forwarded),
                missed_s_rate=ratio(fn,pos),false_alarm_rate=ratio(fp,n-pos))


def build():
    p=pd.read_csv(ROOT/"reports/predictions.csv")
    data=[];groups=[]
    for model in ("rf","spectral"):
        for policy in ("G0","G1","G2","G3"):
            for margin in MARGINS if policy=="G2" else [MARGIN]:
                evaluated=evaluate_rows(p,model,policy,margin)
                for scenario,subset in evaluated.groupby("scenario"):
                    key=dict(model=model,policy=policy,margin=margin,scenario=scenario)
                    data.append(dict(**key,**compute_gate_metrics(subset,model)))
                    if margin==MARGIN:
                        for group,part in subset.groupby("group_id"):
                            groups.append(dict(**key,group_id=group,**compute_gate_metrics(part,model)))
    (ROOT/"reports/gate_metrics.json").write_text(json.dumps(data,indent=2,allow_nan=False),encoding="utf-8")
    pd.DataFrame(groups).to_csv(ROOT/"reports/gate_by_group.csv",index=False)
    ctx=load_context()
    search=pd.read_csv(ROOT/"reports/search_results.csv")
    manifest=dict(dataset_sha256=ctx["dataset_hash"],model_sha256=ctx["versions"],default_model="rf",
                  feature_versions={"rf":"v2-12","spectral":"v2-36"},
                  rf_parameters=ctx["bundles"][0]["rf"].get_params(),training_parameters=json.loads((ROOT/"reports/metrics.json").read_text())["rf_training"],
                  groups={str(k):v["test_groups"] for k,v in ctx["bundles"].items()},
                  train_scales={str(k):v["scale"].tolist() for k,v in ctx["bundles"].items()},
                  default_margin=MARGIN,margin_grid=MARGINS,policy_evaluation="fixed settings, descriptive OOF comparison",
                  elapsed_trial_ms=int(search.elapsed_ms.max()),trials=len(search),unique_windows=len(search.drop_duplicates(["file_id","window_idx"])),
                  budget={"max_trials":2000,"max_seconds":90},run_id=search.search_run_id.iloc[0],python=platform.python_version(),
                  original_report_sha256={n:digest((ROOT/"reports"/n).read_bytes()) for n in ["predictions.csv","metrics.json"]})
    (ROOT/"reports/run_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")

if __name__=="__main__":build()
