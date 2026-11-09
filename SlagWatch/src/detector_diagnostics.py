"""Inspect only the outer training groups and their archived clean inner-validation scores."""
import joblib
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from .hi_models import robust36
from .hi_study import DEST, OUT, ROOT, calibrate, read_config, select_cut
from .data import load_dataset

RUN = OUT / "detector_v1"


def diagnose():
    conf = read_config()
    meta, waves, _ = load_dataset(ROOT/"data/raw")
    archive = pd.read_csv(DEST/"inner_oof_clean.csv", float_precision="round_trip")
    x = np.stack([robust36(w, conf["features_I"]) for w in waves])
    # Compare repeatability inside a file with differences between files, within training only.
    powers = np.stack([welch(w, fs=6400, axis=0, nperseg=1024, noverlap=512, average="median")[1][1:]
                       .reshape(64, 8, 3).sum(axis=1) for w in waves])
    log_power = np.log(np.maximum(powers, 1e-12))
    rf_scores, controls, evidence = [], [], []
    for fold in conf["folds"]:
        f = fold["fold_id"]
        tr = meta[meta.group_id.isin(fold["train"])]
        model = joblib.load(DEST/f"models/fold_{f}.joblib")["models"]["A1"]["estimator"]
        _, pred = calibrate(tr, x[tr.index], clone(model), fold["inner"], .15)
        new = tr.assign(outer_fold=f, model_id="R", score=pred)
        rf_scores.append(new)
        for model_id, rows in pd.concat([archive[archive.outer_fold == f], new]).groupby("model_id"):
            cut = select_cut(rows.y_true, rows.score, .15)
            controls.append(dict(fold=f, model_id=model_id, auc=roc_auc_score(rows.y_true,rows.score), **cut))
        ix = tr.index
        curve = log_power[ix]
        rms = np.stack([np.sqrt(np.mean(waves[j]**2, axis=0)) for j in ix])
        med = np.median(curve, axis=1, keepdims=True)
        detail = curve - med
        per_file = np.stack([np.median(log_power[g.index], axis=0) for _,g in tr.groupby("file_id")])
        within = [np.mean(np.abs(log_power[g.index]-np.median(log_power[g.index],axis=0))) for _,g in tr.groupby("file_id")]
        inner_band = []
        for n in range(64):
            # Differences within the eight coarse band summaries can be masked by summation.
            inner_band.append(np.mean(detail[tr.y_true.to_numpy()==1,n,:],axis=0)
                              - np.mean(detail[tr.y_true.to_numpy()==0,n,:],axis=0))
        delta = np.array(inner_band)
        evidence.append(dict(fold=f, groups=",".join(map(str,sorted(tr.group_id.unique()))),
                             wide_band_min_S_minus_non=float(delta[32:].min()),
                             wide_band_max_S_minus_non=float(delta[32:].max()),
                             within_file_log_power_mad=np.mean(within),
                             across_file_log_power_mad=np.mean(abs(per_file-np.median(per_file,axis=0))),
                             rms_auc_x=roc_auc_score(tr.y_true,rms[:,0]),rms_auc_y=roc_auc_score(tr.y_true,rms[:,1]),
                             rms_auc_z=roc_auc_score(tr.y_true,rms[:,2])))
    RUN.mkdir(exist_ok=True)
    pd.concat(rf_scores).to_csv(RUN/"control_R_inner.csv",index=False)
    pd.DataFrame(controls).to_csv(RUN/"inner_controls.csv",index=False)
    pd.DataFrame(evidence).to_csv(RUN/"train_diagnostics.csv",index=False)
    print(pd.DataFrame(controls).groupby("model_id")[["tp","fp","auc"]].mean())
    print(pd.DataFrame(evidence).to_string(index=False))


if __name__ == "__main__":
    diagnose()
