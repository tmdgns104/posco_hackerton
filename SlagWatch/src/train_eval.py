"""Train on clean groups only and evaluate each held-out group under five inputs."""
import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier

from .data import load_dataset
from .faults import SCENARIOS, seed_for
from .features import NAMES, SPECTRAL_NAMES, band_powers, features, magnitude
from .inference import classify_window
from .quality import quality

RF_CONFIG = dict(n_estimators=100, max_depth=6, min_samples_leaf=3,
                 class_weight="balanced", random_state=42, n_jobs=-1)
BASE_CONFIG = dict(max_depth=1, class_weight="balanced", random_state=42)


def ratio(n, d):
    return float(n / d) if d else None


def summarize(rows, column):
    actual, pred = rows.y_true, rows[column]
    held = pred.isna()
    pos, neg = actual == 1, actual == 0
    tp = int((pos & (pred == 1)).sum())
    fp = int((neg & (pred == 1)).sum())
    missing = int((pos & (pred == 0)).sum())
    injected_bad = rows.fault_injected & rows.scenario.isin(["missing", "freeze"])
    score = f1_score(actual[~held], pred[~held], labels=[0,1], average="macro",
                     zero_division=0) if (~held).any() else None
    return dict(windows=len(rows), tp=tp, fp=fp, fn=missing, tn=int((neg & (pred == 0)).sum()),
                held=int(held.sum()), held_s=int((held & pos).sum()), held_eb=int((held & neg).sum()),
                detection_rate=ratio(tp,pos.sum()), false_alarm_rate=ratio(fp,neg.sum()),
                eligible=int((~held).sum()), coverage=ratio((~held).sum(),len(rows)),
                macro_f1=score, injected_hold_rate=ratio((held & injected_bad).sum(),injected_bad.sum()),
                unchanged_hold_rate=ratio((held & ~rows.fault_injected).sum(),(~rows.fault_injected).sum()))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),encoding="utf-8")


def train(data_dir, out, reports):
    meta, windows, manifest = load_dataset(data_dir)
    if any(quality(w) for w in windows):
        raise ValueError("원본 구간 입력 검사 실패")
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    x = np.array([features(w) for w in windows])
    augmented = np.column_stack((x, np.array([band_powers(w) for w in windows])))
    records = meta.to_dict("records")
    predictions, fold_map = [], {}
    for fold, (train_ids, test_ids) in enumerate(GroupKFold(4).split(x, groups=meta.group_id)):
        training = sorted(set(meta.group_id.iloc[train_ids].tolist()))
        testing = sorted(set(meta.group_id.iloc[test_ids].tolist()))
        assert set(training).isdisjoint(testing)
        rf = RandomForestClassifier(**RF_CONFIG).fit(x[train_ids], meta.y_true.iloc[train_ids])
        # Serial prediction avoids dispatching 100 tiny trees for each interactive selection.
        spectral = RandomForestClassifier(**RF_CONFIG).fit(augmented[train_ids], meta.y_true.iloc[train_ids])
        spectral.n_jobs = rf.n_jobs = 1
        baseline = DecisionTreeClassifier(**BASE_CONFIG).fit(magnitude(x[train_ids]), meta.y_true.iloc[train_ids])
        bundle = dict(rf=rf, spectral=spectral, spectral_feature_names=NAMES+SPECTRAL_NAMES, baseline=baseline, train_groups=training, test_groups=testing,
                      feature_names=NAMES, scale=np.median(x[train_ids, 1::4], axis=0))
        joblib.dump(bundle, out / f"fold_{fold}.joblib", compress=3)
        fold_map.update({str(g):fold for g in testing})
        for j in test_ids:
            row = records[j]
            idx = row["window_idx"]
            for scenario in SCENARIOS:
                seed = seed_for(row["group_id"],row["phase"],idx,scenario)
                _, result = classify_window(windows[j], bundle, scenario, idx, seed)
                injected = scenario in ("noise","spike") or (scenario in ("missing","freeze") and idx%2 == 0)
                predictions.append(dict(**row, fold_id=fold, scenario=scenario, seed=seed,
                                        fault_injected=injected, **result))
    table = pd.DataFrame(predictions)
    metrics = []
    per_group = []
    for scenario, rows in table.groupby("scenario", sort=False):
        for model in ("rf","baseline","spectral"):
            metrics.append(dict(scenario=scenario, model=model, **summarize(rows, model+"_pred")))
            for group, subset in rows.groupby("group_id"):
                per_group.append(dict(group_id=int(group),scenario=scenario,model=model,
                                      **summarize(subset,model+"_pred")))
    table.to_csv(reports/"predictions.csv", index=False)
    manifest.to_csv(reports/"manifest.csv", index=False)
    pd.DataFrame(per_group).to_csv(reports/"by_group.csv",index=False)
    write_json(out/"fold_map.json",fold_map)
    write_json(reports/"metrics.json",dict(dataset="SSFD v3", purpose="해커톤 개발 v2 · 2026-11-07", spectral_features=NAMES+SPECTRAL_NAMES,
        rf_training=RF_CONFIG, baseline_training=BASE_CONFIG, feature_names=NAMES,
        sha256=dict(zip(manifest.file_id,manifest.sha256)),
        python=platform.python_version(),versions={p:version(p) for p in
        ("numpy","pandas","scikit-learn","joblib","streamlit","plotly")}, metrics=metrics))
    print(pd.DataFrame(metrics)[["scenario","model","tp","fn","fp","held","macro_f1"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key, default in [("data-dir","data/raw"),("out","artifacts"),("reports","reports")]:
        parser.add_argument("--"+key, default=Path(default), type=Path)
    args = parser.parse_args()
    train(args.data_dir,args.out,args.reports)
