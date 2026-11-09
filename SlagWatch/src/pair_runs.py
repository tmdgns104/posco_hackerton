"""Supported paired experiments and their stored results, including legacy schema-1 cases."""
from pathlib import Path
from .rocket_detector import RUN as ROCKET, PAIRS as ROCKET_PAIRS

ROOT=Path(__file__).resolve().parents[1]
CNN=ROOT/"experiments/cnn_v1"
CNN_PAIRS=(("J","C0"),("J","C1"),("A1","C0"),("A1","C1"),("C0","C1"))
RUNS={p.name:p for p in (ROCKET,CNN)}


def pairs_for(run):
    if run.name not in RUNS:
        raise ValueError("알 수 없는 평가 실행 폴더")
    return ROCKET_PAIRS if run.name==ROCKET.name else CNN_PAIRS
