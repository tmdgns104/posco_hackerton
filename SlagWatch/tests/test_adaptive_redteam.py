import numpy as np
import pytest

from src.adaptive_redteam import (Setting, image_change, classify_objective,
                                  search_one, suggestion, seed_of, LOW, HIGH)
from src.adaptive_signal import signal_change


def test_image_transform_deterministic_and_safe():
    raw=np.full((200,200),.5,dtype=np.float32)
    z=np.random.default_rng(33).normal(size=raw.shape).astype(np.float32)
    p=np.array([.03,.02,1.1,.2])
    a,q=image_change(raw,p,z); b,q2=image_change(raw,p,z)
    assert np.array_equal(a,b) and q==q2
    assert a.shape==(200,200) and a.dtype==np.float32 and 0<=a.min()<=a.max()<=1
    with pytest.raises(ValueError): image_change(raw,np.array([.2,0,1,.5]),z)


def test_no_false_failure_when_correct_clean():
    raw=np.full((200,200),.45,dtype=np.float32)
    def predict(v):
        # A deterministic mock that flips class when brightness exceeds threshold.
        return dict(scores=[float(v.mean()-.49),float(.49-v.mean())],classes=['a','b'])
    # At .45 mean the correct label is b.
    r=search_one(Setting('mock_a',budget=12,warmup=4,strategy='adaptive'),raw,'b',predict)
    assert len(r['trials'])==12 and r['original_prediction']=='b'
    assert all('input_sha256' in t for t in r['trials'])
    assert 0<=r['failures']<=12
    assert r==search_one(Setting('mock_a',budget=12,warmup=4,strategy='adaptive'),raw,'b',predict)


def test_invalid_clean_rejected():
    raw=np.full((200,200),.5,dtype=np.float32)
    with pytest.raises(ValueError,match='already misclassified'):
        search_one(Setting('id',budget=8,warmup=4),raw,'a',lambda _:dict(classes=['a','b'],scores=[0,1]))


def test_adaptive_samples_within_domain():
    h=[];rng=np.random.default_rng(4)
    for i in range(12):
        p=suggestion(h,rng,'adaptive',warmup=5)
        assert np.all(p>=LOW) and np.all(p<=HIGH)
        h.append(dict(parameters_vector=p.tolist(),priority=float(p[0]-p[2])))
    assert len(h)==12


def test_signal_transform_validates_inputs():
    raw=np.random.default_rng(5).normal(0,.1,size=(3200,3))
    z=np.random.default_rng(2).normal(size=raw.shape)
    x,rmse=signal_change(raw,[.1,1.1,.4,1.0],z)
    assert x.shape==raw.shape and np.isfinite(x).all() and rmse>0
    with pytest.raises(ValueError):signal_change(raw,[2,1,.4,1],z)


def test_original_manifest_and_scope():
    from src.adaptive_audit import audit
    report=audit()
    assert report['n_failed']==0
    assert report['model_adoption']['G1'] and not report['model_adoption']['K1']
    from src.adaptive_redteam import run_surface
    from src.surface_data import ROOT,RUN
    import pandas as pd
    m=pd.read_csv(RUN/'split_manifest.csv')
    test_id=m[m.split=='test'].iloc[0].sample_id
    with pytest.raises(ValueError,match='validation'):
        run_surface(ROOT,[test_id],budget=4,warmup=2)
