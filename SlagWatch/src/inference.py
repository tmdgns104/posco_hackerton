import numpy as np
from .features import features, magnitude, band_powers
from .faults import inject
from .quality import quality


def classify_window(window, model_bundle, scenario, window_idx, seed):
    changed = inject(window, model_bundle["scale"], scenario, window_idx, seed)
    return changed, predict_window(changed, model_bundle)


def predict_window(changed, model_bundle):
    reason = quality(changed)
    if reason:
        return dict(input_status="INPUT_CHECK", reason_codes=reason,
                             rf_score=None, rf_pred=None, baseline_pred=None, spectral_pred=None, spectral_score=None)
    x = features(changed).reshape(1, -1)
    rf = model_bundle["rf"]
    positive = np.flatnonzero(rf.classes_ == 1).item()
    score = float(rf.predict_proba(x)[0, positive])
    spectral = model_bundle["spectral"]
    sx = np.concatenate((x.ravel(), band_powers(changed)))[None, :]
    spectral_score = float(spectral.predict_proba(sx)[0, np.flatnonzero(spectral.classes_ == 1).item()])
    return dict(spectral_score=spectral_score, spectral_pred=int(spectral_score >= 0.5),
                         input_status="SLAG" if score >= 0.5 else "NON_SLAG",
                         reason_codes="", rf_score=score, rf_pred=int(score >= 0.5),
                         baseline_pred=int(model_bundle["baseline"].predict(magnitude(x))[0]))
