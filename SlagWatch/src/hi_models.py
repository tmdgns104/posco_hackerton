"""Pure signal inference. No labels, condition names or file/group identifiers enter these functions."""
import numpy as np
from scipy.signal import welch
from .features import features,band_powers
from .quality import quality


def spectral36(w):
    return np.r_[features(w),band_powers(w)]


def robust36(w,settings):
    q05,q25,med,q75,q95=np.quantile(w,[.05,.25,.5,.75,.95],axis=0)
    mad=np.median(np.abs(w-med),axis=0)
    keep=(w>=q05)&(w<=q95)
    rms=np.sqrt(np.where(keep,w*w,0).sum(axis=0)/keep.sum(axis=0))
    freq,psd=welch(w,axis=0,**settings["welch"])
    power=np.add.reduceat(psd,np.searchsorted(freq,settings["band_starts_hz"]),axis=0)*(freq[1]-freq[0])
    total=power.sum(axis=0)
    rel=np.divide(power,total,out=np.zeros_like(power),where=total>0)
    eps=settings["eps"]
    by_axis=np.column_stack([mad,q75-q25,rms,np.log(np.maximum(rel,eps)).T,np.log(np.maximum(total,eps))])
    return by_axis.ravel()


def scores(estimator,matrix):
    if hasattr(estimator,"decision_function"):
        return estimator.decision_function(matrix)
    return estimator.predict_proba(matrix)[:,list(estimator.classes_).index(1)]


def predict_matrix(bundle,matrix):
    s=scores(bundle["estimator"],matrix)
    return s,(s>=bundle["threshold"]).astype(int)


def predict_signal(w,bundle,settings):
    reason=quality(w)
    if reason:return None,None,reason
    x=robust36(w,settings) if bundle["representation"]=="robust" else spectral36(w)
    s,p=predict_matrix(bundle,x[None,:])
    return float(s[0]),int(p[0]),""
