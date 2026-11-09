"""Policies use input status and model outputs only. Truth is used by evaluation."""
from .inference import predict_window

MARGIN = 0.10
MARGINS = (0., .05, .10, .15, .20)


def apply_gate(reason, score, other_pred, policy="G2", margin=MARGIN):
    if policy not in ("G0","G1","G2","G3") or not 0 <= margin <= .5:
        raise ValueError("Invalid gate settings")
    if reason:
        code = "CONSTANT_AXIS" if reason == "CONST_AXIS" else "INVALID_INPUT"
        return ("NO_PREDICTION" if policy == "G0" else "HOLD"), [code], None
    pred = int(score >= .5)
    reasons = []
    if policy in ("G2","G3") and .5-margin < score < .5+margin:
        reasons.append("LOW_MARGIN")
    if policy == "G3" and other_pred != pred:
        reasons.append("MODEL_DISAGREEMENT")
    return ("HOLD", reasons, None) if reasons else ("PASS", [], pred)


def evaluate_window(window, bundle, model="rf", policy="G2", margin=MARGIN):
    result = predict_window(window,bundle)
    other = "spectral" if model=="rf" else "rf"
    score=result[model+"_score"]
    decision,reasons,forward=apply_gate(result["reason_codes"],score,result[other+"_pred"],policy,margin)
    return dict(pred=result[model+"_pred"],score=score,decision=decision,reasons=reasons,
                forwarded=forward,input_status=result["reason_codes"] or "VALID")
