"""Read-only diagnosis of the frozen J model; no fit, threshold or model selection."""
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .data import load_dataset
from .detector import RUN
from .detector_study import encode
from .hi_study import ROOT, read_config


def run():
    cfg = read_config()
    meta, raw, _ = load_dataset(ROOT/"data/raw")
    rows = []
    for split in cfg["folds"]:
        f = split["fold_id"]
        b = joblib.load(RUN/f"models/fold_{f}.joblib")
        scaler, svc = b["estimators"]["J"]
        for part, groups in [("outer_train", split["train"]), ("outer_test", split["test"])]:
            sub = meta[meta.group_id.isin(groups)]
            for scenario in cfg["conditions"]:
                x, _ = encode(sub, raw, cfg["features_I"], scenario, b["scale"])
                z = scaler.transform(x["fine"])
                nearest = cdist(z, svc.support_vectors_, "sqeuclidean").min(axis=1)
                affinity = np.exp(-svc._gamma * nearest)
                score = svc.decision_function(z)
                threshold = b["cuts"][.15]["J"]
                rows.append(dict(fold=f, partition=part, scenario=scenario, n=len(z),
                                 median_abs_z_fine=np.median(abs(z[:,36:])),
                                 median_nearest_similarity=np.median(affinity), intercept=svc.intercept_[0],
                                 median_score= np.median(score), max_score=score.max(), threshold=threshold,
                                 n_above_threshold=(score>=threshold).sum()))
    pd.DataFrame(rows).to_csv(RUN/"J_score_shift.csv",index=False)


if __name__=="__main__":
    run()
