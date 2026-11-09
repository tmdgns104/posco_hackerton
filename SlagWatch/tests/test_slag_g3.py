import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from src import slag_g3 as core
from src import slag_g3_aug as g3
from src.data import read_recording,recording_paths


def test_all_groups_and_reference_exclusion():
    paths=recording_paths(core.ROOT/'data/raw')
    assert len(paths)==48
    table=pd.read_csv(g3.OUT/'predictions.csv')
    assert set(table.scenario)=={'clean','gain_0p85','noise_0p05','noise_0p08','spike_x'}
    for _,s in table.groupby('scenario'):
        assert len(s)==400
        assert int((s.y_true==1).sum())==160
        assert int((s.y_true==0).sum())==240
        assert (s.loc[s.phase=='E','window_idx']>=5).all()
        assert len(s[['file_id','window_idx']].drop_duplicates())==400


def test_no_group_leakage_and_model_hashes():
    d=json.loads((g3.OUT/'run_manifest.json').read_text(encoding='utf8'))
    for f in d['model_files']:
        path=core.ROOT/f['path']
        assert path.is_file() and core.file_sha(path)==f['sha256']
    for k in range(4):
        b=joblib.load(g3.OUT/'models'/f'fold_{k}_G3R.joblib')
        assert set(b['train_groups']).isdisjoint(b['test_groups'])
        assert sorted(b['train_groups']+b['test_groups'])==list(range(1,17))
        assert b['reference_required'] is True


def test_saved_metrics_exact():
    p=pd.read_csv(g3.OUT/'predictions.csv')
    s=pd.read_csv(g3.OUT/'summary.csv')
    for row in s.itertuples():
        q=p[p.scenario==row.scenario]
        assert core.stats(q)['tp']==row.tp
        assert core.stats(q)['fp']==row.fp
    clean=s.set_index('scenario').loc['clean']
    assert int(clean.tp)==64 and int(clean.fp)==29
    assert int(s.set_index('scenario').loc['noise_0p05'].fp)==31


def test_reference_required_and_inference():
    ref=read_recording(core.ROOT/'data/raw/Steel Slag Flow Dataset (SSFD)/E-1.csv')
    sample=read_recording(core.ROOT/'data/raw/Steel Slag Flow Dataset (SSFD)/S-1.csv')[0]
    fold=joblib.load(g3.OUT/'models'/'fold_3_G3R.joblib')
    assert 1 in fold['test_groups']
    assert core.infer_with_reference(ref[:4],sample,fold)['status']=='NEEDS_REFERENCE'
    valid=core.infer_with_reference(ref,sample,fold)
    assert valid['status']=='VALID' and 0<=valid['score']<=1
    bad=sample.copy();bad[100,0]=np.nan
    assert core.infer_with_reference(ref,bad,fold)['status']=='INPUT_HOLD'


def test_deterministic_noise_no_alias():
    a=np.arange(9600,dtype=float).reshape(3200,3)
    x=g3.deterministic_noise(a,8,.05)
    assert not np.shares_memory(x,a)
    np.testing.assert_array_equal(x,g3.deterministic_noise(a,8,.05))
    assert not np.array_equal(x,g3.deterministic_noise(a,9,.05))


def test_two_consecutive_gate_is_advisory_only():
    p=pd.read_csv(g3.OUT/'paired_vs_J.csv')
    # Preserve grouping. A two-window confirmation loses true positive windows.
    def count(column):
        found=[]
        for _,group in p.sort_values('window_idx').groupby('file_id'):
            a=group[column].to_numpy(dtype=int)
            q=np.r_[0, a[1:]*a[:-1]]
            found.extend(zip(group.y_true_G3R.to_numpy(),q))
        values=np.asarray(found)
        return int(((values[:,0]==1)&(values[:,1]==1)).sum()),int(((values[:,0]==0)&(values[:,1]==1)).sum())
    assert count('prediction_G3R')==(43,15)
    assert count('prediction_J')==(29,24)
