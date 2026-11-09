"""Six frozen detectors; signal features never receive file names, labels or condition IDs."""
from collections import deque
from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from scipy.signal import welch

from .data import HZ, SIZE
from .hi_models import spectral36, robust36, predict_matrix
from .quality import quality

RUN = Path(__file__).resolve().parents[1] / "experiments/detector_v1"
REP = {"A1":"original", "H":"original", "I":"robust", "R":"robust", "J":"fine", "K":"history"}


def fine_spectrum(w):
    freq, psd = welch(w, fs=HZ, window="hann", nperseg=1024, noverlap=512,
                      detrend="constant", scaling="density", average="median", axis=0)
    energy = psd[1:].reshape(64, 8, 3).sum(axis=1) * (freq[1] - freq[0])
    log_energy = np.log(np.maximum(energy, 1e-12))
    return (log_energy - np.median(log_energy, axis=0)).T.ravel()


class Encoder:
    """Create a fresh encoder at every recording boundary; invalid input breaks history."""
    def __init__(self, representations, settings):
        self.reps = set(representations)
        self.settings = settings
        self.past = deque(maxlen=4)

    def step(self, w):
        invalid = "BAD_SHAPE" if w is None else quality(w)
        if invalid:
            self.past.clear()
            return {}, invalid
        values = {}
        if "original" in self.reps:
            values["original"] = spectral36(w)
        if self.reps & {"robust", "fine", "history"}:
            values["robust"] = robust36(w, self.settings)
        if self.reps & {"fine", "history"}:
            values["fine"] = np.r_[values["robust"], fine_spectrum(w)]
        if "history" in self.reps:
            self.past.append(values["fine"])
            values["history"] = np.median(np.stack(self.past), axis=0)
        return values, ""


def model_bundle(fold_bundle, model, budget):
    return dict(model_id=model, estimator=fold_bundle["estimators"][model],
                settings=fold_bundle["settings"], threshold=fold_bundle["cuts"][budget][model])


def predict_record(windows, bundle):
    """One end-of-window prediction at a time, from the start of ONE physical recording."""
    rep = REP[bundle["model_id"]]
    encoder = Encoder([rep], bundle["settings"])
    for window in windows:
        values, reason = encoder.step(window)
        if reason:
            yield dict(score=None, prediction=None, threshold=bundle["threshold"], input_status=reason)
            continue
        s, p = predict_matrix(bundle, values[rep][None, :])
        yield dict(score=float(s[0]), prediction=int(p[0]), threshold=bundle["threshold"], input_status="VALID")


def main():
    parser = argparse.ArgumentParser(description="Classify one 6400 Hz recording (time,x,y,z).")
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    raw = pd.read_csv(args.csv)
    if list(raw.columns) != ["time","x","y","z"] or not len(raw) or len(raw) % SIZE:
        parser.error("Expected time,x,y,z and a positive multiple of 3200 samples from one recording")
    x = raw[["x","y","z"]].to_numpy(dtype=float).reshape(-1,SIZE,3)
    model = joblib.load(RUN/"models/full.joblib")
    for i, result in enumerate(predict_record(x, model)):
        print(json.dumps(dict(end_s=(i+1)/2, model_id=model["model_id"], **result), allow_nan=False))


if __name__ == "__main__":
    main()
