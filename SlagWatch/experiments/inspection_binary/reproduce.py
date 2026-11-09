"""Optional offline validation-only *retraining*, using the one fixed, shipped hypothesis.

Original, calibrated heads and the official test seal are never modified. This is
distinct from exact-input replay, which runs the frozen model and does not retrain.
"""
import argparse
from pathlib import Path

from . import study
from src.inspection_model import RUN


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True,help="new empty output folder")
    p.add_argument("--cache",type=Path,required=True,help="temporary encoder features")
    a=p.parse_args();out=a.out.resolve();cache=a.cache.resolve()
    if out == RUN or RUN in out.parents or (out.exists() and any(out.iterdir())):
        raise ValueError("Use a new, empty working directory outside the frozen study")
    if cache==RUN or RUN in cache.parents:
        raise ValueError("The feature cache belongs outside the frozen study")
    old=study.read(RUN/"selection.json")
    if any(study.SHA(RUN/n)!=v for n,v in old["frozen_files"].items()):
        raise ValueError("The reference experiment has changed")
    out.mkdir(parents=True,exist_ok=True)
    (out/"study.py").write_bytes(study.SCRIPT.read_bytes())
    study.run("baseline",out,cache)
    # Performance comparisons across environments can be informative, but only
    # recreate this failure-led study automatically if the same failure labels
    # are present (small floating-point score drift need not change labels).
    import csv
    def labels(path):
        with path.open(encoding="utf-8",newline="") as f:
            return [(r["sample_id"],r["condition"],r["seed"],r["true_label"],r["predicted_label"])
                    for r in csv.DictReader(f)]
    if labels(out/"K0_validation_diagnostics.csv")!=labels(RUN/"K0_validation_diagnostics.csv"):
        raise ValueError("Retraining produced different baseline validation decisions. Inspect the new results before using this hypothesis.")
    hyp=study.read(RUN/"hypothesis.json")
    hyp.update(created_utc=study.now(),protocol_sha256=study.SHA(out/"protocol.json"),
               baseline_diagnostics_sha256=study.SHA(out/"K0_validation_diagnostics.csv"),
               baseline_head_sha256=study.SHA(out/"models/K0.joblib"),
               reproduced_hypothesis_sha256=study.SHA(RUN/"hypothesis.json"))
    study.dump(out/"hypothesis.json",hyp)
    study.run("candidate",out,cache)
    print("New study:",out/"selection.json")


if __name__=="__main__":
    main()
