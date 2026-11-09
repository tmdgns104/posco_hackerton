"""Read-only, testable evidence adapter for the simple SteelTrust dashboard.

No pretrained model is ever overwritten, and no file-uploaded pickle/joblib
is executed. Each comparison enforces identical sample IDs and true labels.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "experiments" / "inspection_binary"
SURFACE = ROOT / "experiments" / "surface_g1"


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _unique(t: pd.DataFrame, keys: list[str], source: str):
    if t.duplicated(keys).any():
        raise ValueError(f"{source}: 중복된 평가 표본이 있습니다.")
    if t[keys].isna().any().any():
        raise ValueError(f"{source}: 표본 식별자가 비어 있습니다.")


def _validate_pair(a: pd.DataFrame, b: pd.DataFrame, keys: list[str], truth: str):
    _unique(a, keys, '기존 모델')
    _unique(b, keys, '새 모델')
    paired = a.merge(b, on=keys, how='outer', validate='one_to_one',
                     suffixes=('_before', '_after'), indicator=True)
    if len(paired) != len(a) or len(paired) != len(b) or not paired['_merge'].eq('both').all():
        raise ValueError('기존·후보 모델의 시험 이미지 또는 구간이 서로 다릅니다.')
    if not paired[f'{truth}_before'].eq(paired[f'{truth}_after']).all():
        raise ValueError('기존·후보 모델의 정답 라벨이 서로 다릅니다.')
    return paired.drop(columns=['_merge'])


def binary_comparison(root: Path = BIN):
    """K0/K1: validation-only paired decisions; never touches held-out test."""
    key = ['sample_id', 'condition', 'seed']
    tables = {}
    for model in ('K0', 'K1'):
        path = Path(root) / f'{model}_validation_diagnostics.csv'
        t = pd.read_csv(path, dtype={'sample_id': str, 'condition': str})
        required = set(key + ['true_label','predicted_label','error'])
        if not required.issubset(t):
            raise ValueError(f'{model}: 평가 기록 형식이 다릅니다.')
        # 'error' records classification outcomes FP/FN, NOT runtime exceptions.
        observed = t['error'].fillna('')
        expected = np.select([t.true_label.eq('normal') & t.predicted_label.eq('defect'),
                              t.true_label.eq('defect') & t.predicted_label.eq('normal')],
                             ['FP','FN'], default='')
        if not np.array_equal(observed.to_numpy(), expected):
            raise ValueError(f'{model}: 저장된 오류 분류와 예측 결과가 다릅니다.')
        if not t.true_label.isin(['normal','defect']).all() or not t.predicted_label.isin(['normal','defect']).all():
            raise ValueError(f'{model}: 클래스 이름이 잘못됐습니다.')
        tables[model] = t
    c = _validate_pair(tables['K0'], tables['K1'], key, 'true_label')
    truth = c.true_label_before
    b = c.predicted_label_before
    a = c.predicted_label_after
    c['before_wrong'] = b.ne(truth)
    c['after_wrong'] = a.ne(truth)
    c['change'] = np.select([c.before_wrong & ~c.after_wrong, ~c.before_wrong & c.after_wrong],
                            ['fixed', 'new_error'], default='unchanged')
    c['new_miss'] = truth.eq('defect') & b.eq('defect') & a.eq('normal')
    c['new_false_alarm'] = truth.eq('normal') & b.eq('normal') & a.eq('defect')
    c['fixed_miss'] = truth.eq('defect') & b.eq('normal') & a.eq('defect')
    c['fixed_false_alarm'] = truth.eq('normal') & b.eq('defect') & a.eq('normal')
    return c.sort_values(key).reset_index(drop=True)


def binary_stats(c: pd.DataFrame):
    out = {}
    for m, field in [('K0','predicted_label_before'), ('K1','predicted_label_after')]:
        truth = c.true_label_before
        pred = c[field]
        defect = truth.eq('defect')
        normal = ~defect
        tp = int((defect & pred.eq('defect')).sum())
        fn = int((defect & pred.eq('normal')).sum())
        fp = int((normal & pred.eq('defect')).sum())
        tn = int((normal & pred.eq('normal')).sum())
        out[m] = dict(tp=tp, fn=fn, fp=fp, tn=tn,
                      recall=tp/(tp+fn), fpr=fp/(fp+tn), n=len(c))
    return out


def surface_comparison(root: Path = SURFACE):
    """Frozen G1 pair data: test scope only, not adaptive-development validation."""
    path = Path(root) / 'test_pairs.csv.gz'
    t = pd.read_csv(path)
    required = {'sample_id','sigma','seed','label','prediction_before','prediction_after','change'}
    if not required.issubset(t):
        raise ValueError('G1 평가 파일에 필수 열이 없습니다.')
    _unique(t, ['sample_id','sigma','seed'], 'G1')
    if not t.sigma.isin([0.0,0.02,0.05,0.10]).all():
        raise ValueError('G1 시험 조건 형식 오류')
    before = t.prediction_before.eq(t.label)
    after = t.prediction_after.eq(t.label)
    predicted_changes = np.select([~before & after, before & ~after,
                                   ~before & ~after],['fixed','new_error','both_wrong'],
                                  default='both_correct')
    if not (predicted_changes == t.change).all():
        raise ValueError('G1 저장된 오류 변화와 예측값이 일치하지 않습니다.')
    return t.copy()


def surface_stats(t: pd.DataFrame):
    b=t.prediction_before.eq(t.label)
    a=t.prediction_after.eq(t.label)
    return {
        'n':len(t), 'before_errors':int((~b).sum()), 'after_errors':int((~a).sum()),
        'fixed':int((~b & a).sum()), 'new_errors':int((b & ~a).sum()),
        'before_accuracy':float(b.mean()), 'after_accuracy':float(a.mean()),
    }


def slag_comparison():
    from . import slag_compare
    frames = slag_compare.aligned()
    stats = slag_compare.summary()
    return frames, stats


def slag_stats(t: pd.DataFrame):
    from .slag_compare import measures
    return measures(t)


def changes_for_slag(frames, before: str, after: str):
    from .slag_compare import paired
    if before == after:
        raise ValueError('서로 다른 모델 2개를 선택해 주세요.')
    return paired(frames[before], frames[after])


def model_registry():
    return [
        ('강판 결함 6종', 'V0, V1, V2, G1', 'ResNet18 특징 + 분류기', 'G1: 기존 검증 통과'),
        ('정상·불량', 'K0, K1', '이진 분류', 'K1: 새 미검출 증가, 보류'),
        ('슬래그 진동', 'A1, J, C0, C1', '진동 특징 / CNN', '연구 후보'),
        ('슬래그 E 기준', 'G3R', 'ExtraTrees', 'B 기준과 직접 비교 불가'),
        ('슬래그 B 기준', 'G5-B, G6-B, G6 Deep', 'SVM / ExtraTrees / MLP', 'G6-B 우세, 현장 승인 불가'),
    ]
