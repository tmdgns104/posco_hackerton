"""Use frozen checkpoints for paired outer evaluation. This entry point never fits a model."""
import json
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .cnn_eval import KEY, metrics, rank_metrics, paired
from .data import load_dataset
from .hi_study import make_wave
from .pair_detector import detector, digest, file_hash
from .pair_runs import ROOT, CNN, ROCKET as RUN, ROCKET_PAIRS
from .rocket_study import profile, write_json

NAMES=("J","C1","R0","R1")
CHANGE=("rescued_FN","new_FN","removed_FP","new_FP")


def adoption(summary,goals):
    out={}
    grids={}
    for name,g in summary.groupby("model_id"):
        vals=g.set_index("scenario")[["tp","fp"]].sort_index()
        grids[name]=vals
        checks={c:bool(r.tp>=goals["clean" if c=="clean" else "stress"]["min_tp"] and
                       r.fp<=goals["clean" if c=="clean" else "stress"]["max_fp"]) for c,r in vals.iterrows()}
        out[name]=dict(conditions=checks,all_pass=all(checks.values()))
    a,b=grids["J"].loc["clean"],grids["R0"].loc["clean"]
    out["R0"]["comparison_eligible"]=bool(b.tp>a.tp and b.fp<=a.fp)
    d=grids["R1"]-grids["R0"]
    out["R1"]["comparison_eligible"]=bool((d.tp>=0).all() and (d.fp<=0).all() and ((d.tp>0)|(d.fp<0)).any())
    for name in ("R0","R1"):
        out[name]["adopt"]=out[name]["all_pass"] and out[name]["comparison_eligible"]
    return out


def inner_evidence(splits):
    all_summaries=[]
    for split in splits:
        tag=str(split["fold_id"])
        for name in ("R0","R1"):
            g=pd.read_csv(RUN/f"fit_{tag}/{name.lower()}_inner.csv.gz")
            for condition,part in g.groupby("scenario"):
                wrong=part.prediction!=part.y_true
                all_summaries.append(dict(fold_id=tag,model_id=name,scenario=condition,n=len(part),
                    clean_wrong=(part.clean_prediction!=part.y_true).sum(),wrong=wrong.sum(),induced=part.induced_error.sum(),
                    resolved=part.resolved_error.sum(),induced_FN=(part.induced_error&(part.y_true==1)).sum(),
                    induced_FP=(part.induced_error&(part.y_true==0)).sum()))
    return pd.DataFrame(all_summaries)


def source_evidence(cfg,evidence):
    result={}
    base_ranks=pd.read_csv(CNN/"train_outer_auc.csv").query("model_id=='J'")
    for split in cfg["splits"]:
        f=split["fold_id"]
        if f=="full":
            continue
        folder=RUN/f"fit_{f}"
        points=pd.read_csv(folder/"r0_inner.csv.gz",float_precision="round_trip")
        failures=points[points.induced_error]
        first=None
        if not failures.empty:
            r=failures.iloc[0]
            params=pd.read_csv(folder/"perturbations.csv.gz",float_precision="round_trip")
            matches=params[(params.source=="inner_validation") & (params.inner_id==r.inner_id) &
                           (params.file_id==r.file_id) & (params.window_idx==r.window_idx) & (params.condition==r.scenario)]
            row,=matches.to_dict("records")
            first=dict(origin="inner_oof",outer_fold=f,inner_id=int(r.inner_id),file_id=r.file_id,group_id=int(r.group_id),
                       window_idx=int(r.window_idx),phase=r.phase,y_true=int(r.y_true),condition=r.scenario,
                       clean_score=float(r.clean_score),perturbed_score=float(r.score),threshold=float(r.threshold),
                       input_sha256=row["input_sha256"],seed=int(row["seed"]))
        g=evidence[(evidence.fold_id==str(f))&(evidence.model_id=="R0")&(evidence.scenario!="clean")]
        j=base_ranks[base_ranks.fold_id==f].set_index("partition")
        result[str(f)]=dict(inner_example=first,inner_trials=int(g.n.sum()),induced=int(g.induced.sum()),
                            resolved=int(g.resolved.sum()),j_train_auc=float(j.loc["train","auc"]),
                            j_outer_auc=float(j.loc["outer","auc"]))
    return result


def run():
    cfg=json.loads((RUN/"protocol.json").read_text())
    meta,w,_=load_dataset(ROOT/"data/raw")
    old=pd.read_csv(CNN/"predictions.csv",float_precision="round_trip").query("model_id in ['J','C1']")
    blocks=[]; identities=[];times=[];profiles=[]
    for split in cfg["splits"]:
        f=split["fold_id"]
        if f=="full":
            identities.extend(detector(name,f,RUN).identity for name in ("R0","R1"))
            continue
        te=meta[meta.group_id.isin(split["test"])]
        base=detector("J",f,RUN)
        detectors={name:detector(name,f,RUN) for name in NAMES}
        for d in detectors.values():
            assert set(d.test)==set(split["test"]) and set(d.train)==set(split["train"])
            identities.append(d.identity)
        for scenario in cfg["conditions"]:
            waves=[make_wave(w[r.Index],base.legacy["scale"],r,scenario) for r in te.itertuples()]
            hashes=[digest(x) for x in waves]
            for name,d in detectors.items():
                start=perf_counter()
                results=pd.DataFrame(d.infer(waves))
                times.append(dict(fold=f,scenario=scenario,model_id=name,raw_to_prediction_s=perf_counter()-start,n=len(te)))
                if not results.input_status.eq("VALID").all():
                    raise ValueError("Invalid window in the fixed population.")
                block=te.assign(model_id=name,fold_id=f,scenario=scenario,input_sha256=hashes,
                                model_run_id=d.run_id,threshold=d.threshold,score=results.score.to_numpy(),
                                prediction=results.prediction.to_numpy())
                profiles.append(dict(fold_id=f,model_id=name,partition="outer_test",scenario=scenario,**profile(block,d.threshold)))
                blocks.append(block)
    p=pd.concat(blocks,ignore_index=True)
    m=p.merge(old,on=KEY+["model_id"],validate="one_to_one")
    assert len(m)==len(old)==4800
    for k in ("threshold","input_sha256","model_run_id","prediction"):
        np.testing.assert_array_equal(m[k+"_x"],m[k+"_y"])
    np.testing.assert_allclose(m.score_x,m.score_y,rtol=0,atol=1e-5)
    assert p.groupby(KEY).input_sha256.nunique().eq(1).all()
    p.to_csv(RUN/"predictions.csv",index=False)
    p=pd.read_csv(RUN/"predictions.csv",float_precision="round_trip")
    for filename,subkeys in [("summary",[]),("by_fold",["fold_id"]),("by_group",["group_id","fold_id"])]:
        keys=["model_id","scenario"]+subkeys;rows=[]
        for vals,g in p.groupby(keys):
            stats=metrics(g)
            if not subkeys:
                assert stats["n_S"]==160 and stats["n_nonS"]==320 and (g.phase.value_counts()==160).all()
                ranks=pd.DataFrame([rank_metrics(v) for _,v in g.groupby("fold_id")]).mean()
                stats.update({"mean_fold_"+k:v for k,v in ranks.items()})
            elif subkeys==["fold_id"]:
                stats.update(rank_metrics(g))
            rows.append(dict(zip(keys,vals),**stats))
        pd.DataFrame(rows).to_csv(RUN/(filename+".csv"),index=False)
    evidence=inner_evidence(cfg["splits"])
    evidence.to_csv(RUN/"inner_diagnostics.csv",index=False)
    write_json(RUN/"evidence.json",source_evidence(cfg,evidence))
    pd.DataFrame(profiles).to_csv(RUN/"outer_scores.csv",index=False)
    pd.DataFrame(times).to_csv(RUN/"inference_times.csv",index=False)
    summary=pd.read_csv(RUN/"summary.csv")
    changes=[]
    for before,after in ROCKET_PAIRS:
        diff=paired(p,before,after)
        for cond,g in diff.groupby("scenario"):
            changes.append(dict(before=before,after=after,scenario=cond,**{k:int(g.change.eq(k).sum()) for k in CHANGE}))
    pairs=pd.DataFrame(changes)
    pairs.to_csv(RUN/"changes.csv",index=False)
    for side in ("before","after"):
        pairs=pairs.merge(summary.add_suffix("_"+side),left_on=[side,"scenario"],
                          right_on=["model_id_"+side,"scenario_"+side],validate="many_to_one")
    pairs.to_csv(RUN/"pair_summary.csv",index=False)
    write_json(RUN/"verdict.json",adoption(summary,cfg["goals"]))
    reports=["evidence.json","changes.csv","pair_summary.csv","summary.csv","inner_diagnostics.csv","verdict.json"]
    write_json(RUN/"manifest.json",dict(study=RUN.name,dataset_manifest=cfg["dataset_sha256"],versions=cfg["versions"],
                protocol_sha256=file_hash(RUN/"protocol.json"),models=identities,predictions_sha256=file_hash(RUN/"predictions.csv"),
                checkpoints={p.name:file_hash(p) for p in (RUN/"models").glob("*.joblib")},
                report_hashes={n:file_hash(RUN/n) for n in reports}))
    print(summary[["model_id","scenario","tp","fn","fp","tn"]].to_string(index=False))


if __name__=="__main__":
    with threadpool_limits(limits=2,user_api="blas"):
        run()
