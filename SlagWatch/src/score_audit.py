"""Replay frozen A1/I detectors and inspect the cost of the two archived policies."""
import hashlib

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .data import load_dataset
from .hi_models import spectral36, robust36
from .hi_study import DEST, OUT, ROOT, make_wave, read_config, select_cut
from .safety_eval import SAFETY, hi_cases

AUDIT = OUT / "score_audit"
RECORD = ["fold_id", "scenario", "file_id", "window_idx"]
PAIR_NAMES = {(0, 1): "I만 검출", (1, 0): "A1만 검출",
              (1, 1): "모두 검출", (0, 0): "모두 미검출"}


def inner_diagnostics(inner):
    """Two existing reference points only, using clean inner validation scores."""
    records = []
    for fold, rows in inner[inner.model_id == "I"].groupby("outer_fold"):
        selected, = rows.threshold.unique()
        y = rows.y_true.to_numpy()
        scores = rows.score.to_numpy()
        auc = roc_auc_score(y, scores)
        for point, cut in [("native", 0.), ("selected", selected)]:
            positive = scores >= cut
            tp = int(positive[y == 1].sum())
            fp = int(positive[y == 0].sum())
            n_s, n_non = int((y == 1).sum()), int((y == 0).sum())
            records.append(dict(fold_id=fold, point=point, threshold=cut, n_s=n_s,
                                n_nonS=n_non, tp=tp, fp=fp, recall=tp/n_s, fpr=fp/n_non,
                                roc_auc=auc))
    return pd.DataFrame(records)


def paired_slags(predictions, cases):
    """Attach evaluation labels after aligning original predictions and existing masks."""
    i_predictions = predictions[predictions.model_id == "I"].set_index(RECORD)
    a1_predictions = predictions[predictions.model_id == "A1"].set_index(RECORD)
    paired = i_predictions[["prediction", "input_sha256"]].join(
        a1_predictions[["prediction", "input_sha256"]], how="outer", lsuffix="_I", rsuffix="_A1",
        validate="one_to_one")
    if not paired.input_sha256_I.eq(paired.input_sha256_A1).all():
        raise ValueError("A1 and I must use the same signal in every comparison")
    for policy in ("margin", "proposed"):
        rows = cases[(cases.model_id == "I") & (cases.policy == policy)].set_index(RECORD)
        pd.testing.assert_index_equal(rows.index.sort_values(), paired.index.sort_values())
        paired = paired.join(rows[["pass_matched", "pass_calibrated"]].add_prefix(policy + "_"),
                             validate="one_to_one")
    pd.testing.assert_series_equal(i_predictions.y_true.sort_index(), a1_predictions.y_true.sort_index())
    slag = paired.join(i_predictions[["y_true"]], validate="one_to_one").query("y_true == 1")
    records = []
    for scenario, rows in slag.groupby("scenario"):
        for (a1, i), label in PAIR_NAMES.items():
            part = rows[(rows.prediction_A1 == a1) & (rows.prediction_I == i)]
            for mode in ("matched", "calibrated"):
                records.append(dict(scenario=scenario, mode=mode, A1=a1, I=i, category=label,
                                    n=len(part), margin_held=int((~part["margin_pass_" + mode]).sum()),
                                    proposed_held=int((~part["proposed_pass_" + mode]).sum())))
    return pd.DataFrame(records)


def verify_replay(predictions, inner):
    """Independent raw-score check against the actual saved sklearn methods."""
    config = read_config()
    meta, raw, _ = load_dataset(ROOT / "data/raw")
    verified = []
    for split in config["folds"]:
        fold = split["fold_id"]
        bundle = joblib.load(DEST / f"models/fold_{fold}.joblib")
        assert set(bundle["train_groups"]) == set(split["train"])
        assert set(bundle["test_groups"]) == set(split["test"])
        assert set(split["train"]).isdisjoint(split["test"])
        test = meta[meta.group_id.isin(split["test"])]
        for model in ("A1", "I"):
            sub = inner[(inner.outer_fold == fold) & (inner.model_id == model)]
            assert set(sub.group_id) == set(split["train"]) and len(sub) == 360
            frozen = bundle["models"][model]["threshold"]
            assert sub.threshold.eq(frozen).all()
            assert select_cut(sub.y_true, sub.score, config["target_fp_rate"])["threshold"] == frozen
        for scenario in config["conditions"]:
            waves = [make_wave(raw[row.Index], bundle["scale"], row, scenario) for row in test.itertuples()]
            hashes = [hashlib.sha256(wave.tobytes()).hexdigest() for wave in waves]
            for model in ("I", "A1"):
                detector = bundle["models"][model]
                est = detector["estimator"]
                np.testing.assert_array_equal(est.classes_, [0, 1])
                if model == "I":
                    assert not est[-1].probability
                    matrix = np.stack([robust36(w, config["features_I"]) for w in waves])
                    scores = est.decision_function(matrix)
                else:
                    scores = est.predict_proba(np.stack([spectral36(w) for w in waves]))[:, 1]
                tab = predictions[(predictions.model_id == model) & (predictions.fold_id == fold)
                                  & (predictions.scenario == scenario)]
                tab = test[["file_id", "window_idx"]].merge(tab, on=["file_id", "window_idx"], validate="one_to_one")
                np.testing.assert_array_equal(tab.input_sha256, hashes)
                np.testing.assert_allclose(tab.score, scores, atol=1e-12, rtol=0)
                assert tab.threshold.eq(detector["threshold"]).all()
                np.testing.assert_array_equal(tab.prediction, scores >= detector["threshold"])
                verified.append(dict(scenario=scenario, model_id=model, fold_id=fold, n=len(scores),
                                     threshold=detector["threshold"], min_score=scores.min(), max_score=scores.max(),
                                     max_abs_score_error=np.max(np.abs(scores - tab.score))))
    return pd.DataFrame(verified)


def run():
    pred = pd.read_csv(DEST/"predictions.csv", float_precision="round_trip")
    inner = pd.read_csv(DEST/"inner_oof_clean.csv", float_precision="round_trip")
    saved = pd.read_csv(SAFETY/"cases.csv.gz", float_precision="round_trip")
    replay = verify_replay(pred, inner)
    rebuilt, _ = hi_cases()
    index = RECORD + ["policy"]
    cols = ["score", "threshold", "prediction", "risk_score", "cut", "pass_matched", "pass_calibrated"]
    old = saved[saved.model_id == "I"].set_index(index).sort_index()
    new = rebuilt[rebuilt.model_id == "I"].set_index(index).sort_index()
    pd.testing.assert_frame_equal(new[cols], old[cols], check_exact=True)
    breakdown = paired_slags(pred, saved)
    assert breakdown.groupby(["scenario", "mode"]).n.sum().eq(160).all()
    AUDIT.mkdir(exist_ok=True)
    replay.to_csv(AUDIT/"replay.csv", index=False)
    breakdown.to_csv(AUDIT/"paired_s.csv", index=False)
    inner_diagnostics(inner).to_csv(AUDIT/"inner_points.csv", index=False)
    print(f"Verified {replay.n.sum()} detector scores and {len(new)} policy rows in both modes.")
    print(breakdown.query("scenario == 'clean' and mode == 'matched'").to_string(index=False))


if __name__ == "__main__":
    run()
