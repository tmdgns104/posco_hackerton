import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src import slag_compare as m


def test_three_models_exact_240():
    f=m.aligned()
    assert all(len(t)==240 for t in f.values())
    assert len(set(tuple(x) for x in f['G6-B ExtraTrees'][['group','phase','start_s']].to_numpy()))==240


def test_stored_metrics_recomputed():
    s=m.summary().set_index('model')
    assert tuple(s.loc['G6-B ExtraTrees',['TP','FN','FP','TN']].tolist()) == (86,74,5,75)
    assert tuple(s.loc['G6 Deep MLP',['TP','FN','FP','TN']].tolist()) == (49,111,7,73)
    assert tuple(s.loc['G5-B SVM',['TP','FN','FP','TN']].tolist()) == (81,79,5,75)


def test_pairing_fixed_and_new_errors():
    f=m.aligned()
    changes=m.paired(f['G6-B ExtraTrees'],f['G6 Deep MLP']).change.value_counts()
    assert int(changes.get('fixed',0))==10
    assert int(changes.get('new_error',0))==49


def test_bootstrap_group_and_events():
    f=m.aligned();a=m.bootstrap_delta(f['G6-B ExtraTrees'],f['G6 Deep MLP'],n=50)
    assert a['n_independent_occurrences']==16
    assert all(len(a[k])==2 for k in ['delta_recall_95pct','delta_fpr_95pct'])
    ev=m.events(f['G6-B ExtraTrees'],2)
    assert len(ev)==32 and set(ev.phase)=={'B','S'}


def test_incompatible_csv_rejected(tmp_path):
    for n in m.MODEL_FILES.values():shutil.copy2(m.EVIDENCE/n,tmp_path/n)
    f=tmp_path/'G6Deep_oof.csv';t=pd.read_csv(f)
    t.loc[0,'group']=17;t.to_csv(f,index=False)
    with pytest.raises(ValueError):m.aligned(tmp_path)


def test_full_model_inference_matches_reference():
    raw=m.ROOT/'data/raw'
    ref=next(raw.rglob('B-1.csv')).read_bytes();target=next(raw.rglob('S-1.csv')).read_bytes()
    out=m.external_inference(ref,target)
    assert len(out)==10
    assert np.isfinite(out[['G5B_score','G6B_score']]).all().all()
    example=pd.read_csv(m.ROOT/'benchmarks/slag_b2/G6B_example.csv',encoding='utf-8-sig')
    assert np.allclose(out.G6B_score.to_numpy(),example.slag_score.to_numpy(),atol=1e-12)


def test_bad_reference_not_accepted():
    raw=m.ROOT/'data/raw';target=next(raw.rglob('S-1.csv')).read_bytes()
    ref=next(raw.rglob('B-1.csv')).read_bytes()
    short=b'\n'.join(ref.splitlines()[:1000])+b'\n'
    with pytest.raises(ValueError):m.external_inference(short,target)


def test_no_automatic_deployment():
    g=m.adoption_report()
    assert not g['industrial_approved'] and not g['research_deep_better_than_g6b']
    assert not g['independent_site_validation']


def test_checkpoint_inventory():
    p=m.EVIDENCE/'G6Deep_cv_checkpoints'
    assert len(list(p.glob('fold*_seed*.pt')))==20
    q=json.loads((m.EVIDENCE/'G6Deep_summary.json').read_text())
    assert q['cv_complete'] and q['slag_tp']==49
