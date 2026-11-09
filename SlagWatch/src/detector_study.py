"""Nested development CV and full-data refit. Run from the project root."""
import hashlib
import json
import platform
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from .data import load_dataset
from .detector import Encoder, REP, RUN, model_bundle
from .detector_diagnostics import diagnose
from .hi_models import predict_matrix
from .hi_study import DEST, ROOT, calibrate, make_wave, read_config, select_cut

PLAN = json.loads((RUN/"protocol.json").read_text(encoding="utf8"))
ORDER = PLAN["models_in_tie_order"]
BUDGETS = PLAN["budgets"]
ID = ["file_id", "window_idx"]
READ = {"float_precision":"round_trip"}


def write_json(name, obj):
    (RUN/name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(meta, waves, settings, scenario="clean", scale=None):
    matrices = {rep: [] for rep in set(REP.values())}
    hashes, ids = [], []
    for _, group in meta.groupby("file_id", sort=False):
        encoder = Encoder(matrices, settings)
        group = group.sort_values("window_idx")
        if group.window_idx.tolist() != list(range(10)):
            raise ValueError("A full file must be present; context starts at its first window")
        for r in group.itertuples():
            w = waves[r.Index] if scenario == "clean" else make_wave(waves[r.Index], scale, r, scenario)
            values, err = encoder.step(w)
            if err:
                raise ValueError(f"Unexpected invalid evaluation input {r.file_id}/{r.window_idx}: {err}")
            for rep in matrices:
                matrices[rep].append(values[rep])
            hashes.append(hashlib.sha256(w.tobytes()).hexdigest())
            ids.append(r.Index)
    if ids != meta.index.tolist():
        raise ValueError("Wave, label and feature row order differs")
    return {key:np.stack(rows) for key, rows in matrices.items()}, hashes


def check_splits(train, inner):
    groups = set(train.group_id.unique())
    validations = []
    for split in inner:
        a, b = set(split["train"]), set(split["val"])
        if a & b or a | b != groups:
            raise ValueError("Invalid inner split")
        validations.extend(b)
    if sorted(validations) != sorted(groups):
        raise ValueError("Inner validation groups are missing or repeat")


def pipeline_for(model, original):
    return clone(original["A1" if model in ("A1","R") else "I"]["estimator"])


def select_models(train, matrices, inner, original, archived=None):
    """No outer examples, scores or labels enter selection."""
    check_splits(train, inner)
    stats, inner_rows = [], []
    for model in ORDER:
        if archived is not None and model in archived.model_id.unique():
            rows = archived[archived.model_id == model]
            joined = train[ID+["y_true","group_id"]].merge(rows, on=ID, suffixes=("", "_stored"),
                                                           how="left", validate="one_to_one")
            if (joined.score.isna().any() or not joined.y_true.equals(joined.y_true_stored)
                    or not joined.group_id.equals(joined.group_id_stored)):
                raise ValueError("Archived inner prediction does not match training groups")
            s = joined.score.to_numpy()
        else:
            _, s = calibrate(train, matrices[REP[model]], pipeline_for(model, original), inner, .15)
        inner_rows.append(train[ID+["y_true","group_id"]].assign(model_id=model, score=s))
        for rate in BUDGETS:
            cut = select_cut(train.y_true, s, rate)
            stats.append(dict(model_id=model, budget=rate, auc=roc_auc_score(train.y_true,s),
                              ap=average_precision_score(train.y_true,s), **cut))
    table = pd.DataFrame(stats)
    cuts = {}
    winners = {}
    for rate, rows in table.groupby("budget"):
        choice = rows.assign(order=rows.model_id.map(ORDER.index)).sort_values(
            ["tp","fp","order"], ascending=[False,True,True]).iloc[0]
        winners[rate] = choice.model_id
        cuts[rate] = rows.set_index("model_id").threshold.to_dict()
    return table, pd.concat(inner_rows), cuts, winners


def fit_outer(fold, meta, clean, cfg, archive):
    train = meta[meta.group_id.isin(fold["train"])]
    if set(train.group_id) & set(fold["test"]):
        raise ValueError("Outer leakage")
    x = {k:v[train.index] for k,v in clean.items()}
    f = fold["fold_id"]
    old = joblib.load(DEST/f"models/fold_{f}.joblib")
    if set(old["train_groups"]) != set(fold["train"]) or set(old["test_groups"]) != set(fold["test"]):
        raise ValueError("Wrong archived model")
    table, inn, cuts, winners = select_models(train, x, fold["inner"], old["models"], archive[archive.outer_fold==f])
    models = {}
    for model in ORDER:
        if model in ("A1","H","I"):
            est = old["models"][model]["estimator"]
        else:
            est = pipeline_for(model, old["models"]).fit(x[REP[model]], train.y_true)
        models[model] = est
    result = dict(estimators=models, settings=cfg["features_I"], cuts=cuts, winners=winners,
                  scale=old["scale"], train_groups=fold["train"], test_groups=fold["test"])
    joblib.dump(result, RUN/f"models/fold_{f}.joblib", compress=3)
    return table.assign(fold_id=f), inn.assign(fold_id=f), winners


def refit_full(meta, clean, cfg):
    original = joblib.load(DEST/"models/fold_0.joblib")["models"]
    inner = [dict(train=sorted(map(int,meta.iloc[a].group_id.unique())),
                  val=sorted(map(int,meta.iloc[b].group_id.unique())))
             for a,b in GroupKFold(3).split(meta,groups=meta.group_id)]
    table, _, cuts, winners = select_models(meta, clean, inner, original)
    rate = PLAN["primary_budget"]
    model = winners[rate]
    full = dict(model_id=model, estimator=pipeline_for(model, original).fit(clean[REP[model]],meta.y_true),
                settings=cfg["features_I"], threshold=cuts[rate][model], train_groups=sorted(map(int,meta.group_id.unique())))
    joblib.dump(full, RUN/"models/full.joblib", compress=3)
    table.to_csv(RUN/"full_inner_summary.csv", index=False)
    return dict(model_id=model, threshold=cuts[rate][model], train_groups=full["train_groups"], inner=inner)


def evaluate(meta, waves, cfg):
    rows = []
    for fold in cfg["folds"]:
        f = fold["fold_id"]
        bundle = joblib.load(RUN/f"models/fold_{f}.joblib")
        te = meta[meta.group_id.isin(fold["test"])]
        for scenario in cfg["conditions"]:
            x, hashes = encode(te, waves, cfg["features_I"], scenario, bundle["scale"])
            for rate in BUDGETS:
                for model in ORDER:
                    m = model_bundle(bundle, model, rate)
                    s, p = predict_matrix(m, x[REP[model]])
                    block = te.assign(fold_id=f, scenario=scenario, budget=rate, model_id=model,
                                      configuration=model, score=s, threshold=m["threshold"], prediction=p,
                                      input_sha256=hashes)
                    rows.append(block)
                    if model == bundle["winners"][rate]:
                        rows.append(block.assign(model_id="selected"))
    pred = pd.concat(rows, ignore_index=True)
    pred["sample_id"] = pred.file_id.str.removesuffix(".csv")+"#"+pred.window_idx.astype(str)
    # Anchor both 10% baseline scores and all current endpoint hashes to the previous run.
    old = pd.read_csv(DEST/"predictions.csv", **READ).query("model_id in ['A1','I','H']")
    frozen = pred.query("budget == 0.10 and model_id in ['A1','I','H']")
    key = ID+["scenario","fold_id","model_id"]
    pair = frozen.merge(old, on=key, validate="one_to_one")
    assert len(pair) == len(frozen) == len(old) == 7200
    for name in ["input_sha256","prediction","threshold"]:
        np.testing.assert_array_equal(pair[name+"_x"], pair[name+"_y"])
    np.testing.assert_allclose(pair.score_x, pair.score_y, atol=1e-12, rtol=0)
    for base in ("A1", "I"):
        key = ID+["scenario","fold_id","budget"]
        ref = pred[pred.model_id==base][key+["prediction"]].rename(columns={"prediction":base})
        pred = pred.merge(ref,on=key,validate="many_to_one")
    return pred


def counts(g):
    y = g.y_true == 1
    p = g.prediction == 1
    tp, fp = int((y&p).sum()), int((~y&p).sum())
    ns, nn = int(y.sum()), int((~y).sum())
    out = dict(n=len(g), n_s=ns, n_nonS=nn, tp=tp, fn=ns-tp, fp=fp, tn=nn-fp,
               recall=tp/ns, fpr=fp/nn, precision=tp/(tp+fp) if tp+fp else np.nan,
               fp_E=int(((g.phase=="E")&p).sum()), fp_B=int(((g.phase=="B")&p).sum()))
    for base in ("A1","I"):
        b = g[base] == 1
        out.update({f"rescued_vs_{base}":int((y&p&~b).sum()), f"lost_vs_{base}":int((y&~p&b).sum()),
                    f"added_fp_vs_{base}":int((~y&p&~b).sum()), f"removed_fp_vs_{base}":int((~y&~p&b).sum())})
    return out


def summarize(pred):
    primary = ["budget","model_id","scenario"]
    frames = []
    for extra, filename in [(["fold_id"],"by_fold"), (["group_id","fold_id"],"by_group"), ([],"summary")]:
        keys = primary+extra
        rows = []
        for values, g in pred.groupby(keys):
            out = counts(g)
            if extra == ["fold_id"]:
                out.update(roc_auc=roc_auc_score(g.y_true,g.score), ap=average_precision_score(g.y_true,g.score))
            rows.append(dict(zip(keys,values), **out))
        result = pd.DataFrame(rows)
        if not extra:
            fold_avgs = frames[0].groupby(primary,as_index=False)[["roc_auc","ap"]].mean().rename(
                columns={"roc_auc":"mean_fold_roc_auc","ap":"mean_fold_ap"})
            result = result.merge(fold_avgs,on=primary,validate="one_to_one")
        result.to_csv(RUN/f"{filename}.csv",index=False)
        frames.append(result)
    return frames[-1]


def verdict(summary):
    current = summary[summary.budget == PLAN["primary_budget"]]
    selected = current[current.model_id=="selected"].set_index("scenario")
    clean = selected.loc["clean"]
    base = current.query("model_id=='A1' and scenario=='clean'").iloc[0]
    checks = {}
    for condition, row in selected.iterrows():
        goals = PLAN["targets"]["clean" if condition=="clean" else "stress"]
        checks[condition] = bool(row.recall >= goals["recall"] and row.fpr <= goals["fpr"])
    fair = bool(clean.tp>=base.tp and clean.fp<=base.fp and (clean.tp>base.tp or clean.fp<base.fp))
    return dict(targets_pass=checks, fair_clean_improvement=fair, adopt=all(checks.values()) and fair)


def run():
    (RUN/"models").mkdir(exist_ok=True)
    if not (RUN/"control_R_inner.csv").is_file():
        diagnose()
    cfg = read_config()
    meta, waves, _ = load_dataset(ROOT/"data/raw")
    assert len(meta)==480 and meta.y_true.sum()==160 and meta.group_id.nunique()==16
    clean, _ = encode(meta, waves, cfg["features_I"])
    archive = pd.concat([pd.read_csv(DEST/"inner_oof_clean.csv", **READ),pd.read_csv(RUN/"control_R_inner.csv", **READ)])
    tables, inner, selection = [], [], {}
    for fold in cfg["folds"]:
        table, rows, winners = fit_outer(fold, meta, clean, cfg, archive)
        tables.append(table)
        inner.append(rows)
        selection[fold["fold_id"]] = winners
    pd.concat(tables).to_csv(RUN/"inner_thresholds.csv",index=False)
    pd.concat(inner).to_csv(RUN/"inner_oof.csv.gz",index=False)
    full = refit_full(meta, clean, cfg)
    write_json("selection.json", dict(outer=selection, full=full))
    print("Frozen inner choices", selection, "full", full["model_id"], flush=True)
    # No threshold, representation or model choice is made after this point.
    pred = evaluate(meta, waves, cfg)
    pred.to_csv(RUN/"predictions.csv",index=False)
    summary = summarize(pd.read_csv(RUN/"predictions.csv", **READ))
    write_json("verdict.json", verdict(summary))
    sources = ["protocol.json","../hi_v1/protocol.json","../hi_v1/inner_oof_clean.csv",
               "control_R_inner.csv","../../data/raw/manifest.json"]
    write_json("manifest.json",dict(python=platform.python_version(), platform=platform.system(),
               versions={p:version(p) for p in ("numpy","pandas","scipy","scikit-learn","joblib","streamlit")},
               seed=42, settings=cfg["svm"], rf=pipeline_for("A1",joblib.load(DEST/"models/fold_0.joblib")["models"]).get_params(),
               source_sha256={p:sha(RUN/p) for p in sources},
               artifacts_sha256={path.name:sha(path) for path in sorted((RUN/"models").glob("*.joblib"))}))
    print(summary.query("scenario=='clean' and budget==0.15")[["model_id","tp","fn","fp","tn"]].to_string(index=False))


if __name__ == "__main__":
    run()
