"""Two-stage, validation-only KSDD2 study; run from the project root with -m.

The baseline is saved before the failure hypothesis, which is saved before K1.
Use a NEW output directory to reproduce. Official test pixels never enter this script.
"""
import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import platform
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src import inspection_inputs, inspection_model as model
from src.inspection_inputs import BINARY_CONDITIONS, change_image

ROOT = model.ROOT
MANIFEST = ROOT / "data/ksdd2/manifest.csv"
TRAIN_SOURCE = ROOT / "data/ksdd2/raw/train"
SCRIPT = Path(__file__).resolve()
VERSIONS = {x:version(x) for x in ["torch","torchvision","numpy","opencv-python-headless","scikit-learn","Pillow","joblib"]}
SHA = model.sha


def dump(path, obj):
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_splits():
    with MANIFEST.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    dev = {s: sorted((r for r in rows if r["split"]==s), key=lambda r:r["sample_id"])
           for s in ("train","validation")}
    if not all(dev.values()):
        raise ValueError("Need complete train/validation manifest")
    # Defense in depth: only the official training image directory is readable.
    groups = {}
    for s, rr in dev.items():
        if {r["true_label"] for r in rr} != {"normal","defect"}:
            raise ValueError("Both development classes required")
        groups[s] = {r["group_id"] for r in rr}
        for r in rr:
            p = (ROOT/r["image_path"]).resolve()
            if p.parent != TRAIN_SOURCE.resolve() or p.name.endswith("_GT.png"):
                raise ValueError(f"Not a permitted official training image: {p}")
            if SHA(p) != r["file_sha256"]:
                raise ValueError(f"Source bytes changed: {p}")
    if groups["train"] & groups["validation"]:
        raise ValueError("Shared development group")
    return dev


def labels(rows):
    return np.asarray([int(r["true_label"]=="defect") for r in rows])


def feature_data(rows, policy, out, cache):
    # Caches are training scratch only; running the application never reads them.
    h = hashlib.sha256()
    for piece in [SHA(MANIFEST), SHA(model.__file__), SHA(inspection_inputs.__file__), model.ENCODER_SHA,
                  model.PREP, json.dumps(policy), json.dumps(VERSIONS)]:
        h.update(piece.encode())
    for r in rows:
        h.update(r["sample_id"].encode()); h.update(r["file_sha256"].encode())
    path = cache / (h.hexdigest()+".npy")
    if path.exists():
        a = np.load(path, allow_pickle=False)
        if a.shape != (len(rows),1536) or not np.isfinite(a).all():
            raise ValueError("Invalid training scratch feature cache")
        return a
    chunks = []
    t = time.monotonic()
    for i in range(0,len(rows),model.BATCH):
        raws = []
        for r in rows[i:i+model.BATCH]:
            with Image.open(ROOT/r["image_path"]) as im:
                raw = np.asarray(im.convert("RGB"),dtype=np.float32) / np.float32(255)
            name, seed = policy
            raws.append(change_image(raw, r["sample_id"], name, seed))
        chunks.append(model.extract(raws))
        if (i+model.BATCH)%256 == 0:
            print(f"{policy} {i+model.BATCH}/{len(rows)} ({time.monotonic()-t:.0f}s)",flush=True)
    a = np.concatenate(chunks)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.save(f,a)
    tmp.replace(path)
    return a


def choose_threshold(y, scores):
    # Sweep only distinct observed rounded scores plus the finite all-normal
    # cutoff one ULP above the maximum. Predictions use >= at the exact boundary.
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if y.shape != scores.shape or len(set(y)) != 2 or not np.isfinite(scores).all():
        raise ValueError("Both classes and finite validation-clean scores required")
    best = None
    for t in np.r_[np.unique(scores), np.nextafter(scores.max(),np.inf)]:
        p = scores >= t
        tp, fp = int(np.sum(p & (y==1))), int(np.sum(p & (y==0)))
        if 10*fp > int(sum(y==0)):
            continue
        key = (tp,-fp,float(t))
        if best is None or key > best:
            best = key
    return float(best[2])


def counts(y,p):
    tp,fp = int(np.sum((y==1)&p)), int(np.sum((y==0)&p))
    fn,tn = int(sum(y==1))-tp, int(sum(y==0))-fp
    return dict(TP=tp,FN=fn,FP=fp,TN=tn,defect_recall=tp/(tp+fn),
                normal_fpr=fp/(fp+tn), normal_specificity=tn/(tn+fp),
                normal_recall=tn/(tn+fp), defect_miss_rate=fn/(tp+fn))


def protocol(dev):
    return dict(schema="steeltrust.binary.protocol.v1", project_id="ksdd2", purpose="시제품 시험 기준; 제조 안전 기준 아님",
        source="KolektorSDD2 / official masks: any nonzero annotation = defect; missing or invalid masks rejected",
        split_manifest="data/ksdd2/manifest.csv",split_manifest_sha256=SHA(MANIFEST),
        source_scope="This study fits only training originals and evaluates only validation originals, both from official train. Test is independently locked.",
        development_class_counts={s:dict(Counter(r["true_label"] for r in rr)) for s,rr in dev.items()},
        evaluation_unit="one whole source image with one true label; no tile labels or independent views",
        positive_class="defect", classes=["normal","defect"], model_ids=["K0","K1"],baseline_model="K0",candidate_model="K1",
        encoder={"name":"frozen ImageNet-1K ResNet18","sha256":model.ENCODER_SHA,
                 "features":"concatenate full spatial mean and max per channel from stages 3 and 4 (1536 values)"},
        input={"version":model.PREP,"native":"RGB float32 HWC [0,1]; no mask, filename or annotation at inference",
               "geometry":"fit entire frame into 192w x 512h preserving aspect; rotate landscape 90 degrees clockwise; round resized axes to nearest int, min 1; cv2.INTER_AREA down / LINEAR up; centered BORDER_REFLECT_101 padding with extra pixels bottom/right; no tiling/crop",
               "normalization":"ImageNet RGB means .485/.456/.406, SD .229/.224/.225",
               "rationale":"Official train examples 10000 (229x645, empty mask) and 10021 (228x649, annotated defect in upper-left touching frame edges): retain full RGB view, localized responses, and borders."},
        head={"estimator":"StandardScaler + L2 logistic regression lbfgs, C=.1, class_weight=balanced, max_iter=2000; train split only",
              "K0_training":"one clean image per training original","K1_change":"one failure-led hypothesis must be saved after K0 validation and before K1 fitting; no other candidate"},
        conditions=[{"condition":n,"seed":s} for n,s in BINARY_CONDITIONS],
        transformations={"clean":"identity","noise_weak":"iid RGB Gaussian sigma .02","noise_strong":"iid RGB Gaussian sigma .08",
                         "dark":"multiply by .80","bright":"multiply by 1.20","range":"native float32 in [0,1]; clip after change",
                         "seed_rule":"SHA256(inspection-rgb-v1|sample_id|condition|seed), first 8 bytes big endian -> numpy default_rng",
                         "interpretation":"synthetic sensitivity probes, not representative of every installation or real camera contamination"},
        threshold={"selection":"validation clean only; budget FPR <= .10; maximize TP, then minimize FP, then highest threshold",
                   "sweep":"distinct rounded defect scores plus nextafter(max,+inf) for all-normal; score>=threshold is defect",
                   "score":"logistic decision_function, positive class defect, rounded to 6 decimals; normal=-defect; not probability",
                   "failure":"if best feasible recall is below .80, record absolute failure; no retuning on any condition/test"},
        score_atol=1e-5,minimum_recall=.80,maximum_fpr=.10,clean_drop_limit=.02,other_drop_limit=.05,
        gate={"require_all":"all seven cells meet absolute defect recall >=.80 and normal FPR <=.10; no missing, failed or deferred predictions",
              "relative_limits":"clean recall drop or FPR rise <=.02; each perturbed cell <=.05",
              "weights":"every original equal; each original supplies 7 equally weighted condition/seed cells (1/7 each); retain actual counts per cell",
              "relative_improvement":"sum over all 7 cells; fewer FN with no more FP OR fewer FP with no more FN",
              "comparison":"paired same original/condition/seed; no inferential claims from repeated trials",
              "selection":"only one K1, passes only if all criteria pass; if not, reject on validation and leave official test locked"},
        implementation={"model_source_sha256":SHA(model.__file__),"input_source_sha256":SHA(inspection_inputs.__file__)},
        versions=VERSIONS,created_utc=now())


def save_scores(out, name, head, t, val, xx):
    y = labels(val)
    rr=[]; cells=[]
    for (c,s),x in xx.items():
        z=model.decision(head,x)
        p=z>=t
        cells.append(dict(condition=c,seed=s,**counts(y,p)))
        for r, score, is_pos in zip(val,z,p,strict=True):
            pred="defect" if is_pos else "normal"
            rr.append(dict(sample_id=r["sample_id"],condition=c,seed=s,true_label=r["true_label"],
                 predicted_label=pred,defect_score=score,error=("FN" if r["true_label"]=="defect" else "FP") if pred!=r["true_label"] else ""))
    save_csv(out/f"{name}_validation_diagnostics.csv",rr)
    return cells


def train_head(x,y):
    head=make_pipeline(StandardScaler(),LogisticRegression(class_weight="balanced",C=.1,solver="lbfgs",max_iter=2000))
    with threadpool_limits(2,user_api="blas"):
        head.fit(x,y)
    if head[-1].n_iter_[0] >= 2000:
        raise RuntimeError("Head did not converge")
    return head


def run(stage, out, cache):
    out.mkdir(parents=True,exist_ok=True); cache.mkdir(parents=True,exist_ok=True)
    (out/"models").mkdir(exist_ok=True)
    if (out/"selection.json").exists():
        raise FileExistsError("Frozen study; use --out a fresh directory for retraining")
    dev=read_splits()
    tr,va=dev["train"],dev["validation"]
    y=labels(tr); vy=labels(va)
    if stage=="baseline":
        pr=protocol(dev);dump(out/"protocol.json",pr)
        dump(out/"protocol_receipt.json",dict(created_utc=now(),before="K0 fit or K1 candidate design",sha256=SHA(out/"protocol.json")))
        policies=[("clean",0)];name="K0"
    else:
        pr=read(out/"protocol.json");hyp=read(out/"hypothesis.json")
        if (pr["split_manifest_sha256"]!=SHA(MANIFEST) or pr["implementation"] !=
                {"model_source_sha256":SHA(model.__file__),"input_source_sha256":SHA(inspection_inputs.__file__)}
                or hyp["protocol_sha256"]!=SHA(out/"protocol.json")
                or hyp["baseline_diagnostics_sha256"]!=SHA(out/"K0_validation_diagnostics.csv")
                or hyp["baseline_head_sha256"]!=SHA(out/"models/K0.joblib")):
            raise ValueError("Development split, model/input code or failure hypothesis seal mismatch")
        policies=[(n,int(s)) for n,s in hyp["training_views"]];name="K1"
    if (out/"models"/f"{name}.joblib").exists():
        raise FileExistsError("Refusing to overwrite the trained head")
    features=np.concatenate([feature_data(tr,p,out,cache) for p in policies])
    head=train_head(features,np.tile(y,len(policies)))
    xv=feature_data(va,("clean",0),out,cache)
    t=choose_threshold(vy,model.decision(head,xv))
    joblib.dump(dict(model_id=name,head=head,threshold=t,encoder_sha256=model.ENCODER_SHA,
               preprocessing=model.PREP),out/"models"/f"{name}.joblib",compress=3)
    xx={p:(xv if p==("clean",0) else feature_data(va,p,out,cache)) for p in BINARY_CONDITIONS}
    cells=save_scores(out,name,head,t,va,xx)
    dump(out/f"{name}_study.json",dict(model_id=name,created_utc=now(),threshold=t,
            train_originals=len(tr), training_views=len(features),validation_originals=len(va),
            train_policies=policies,validation_cells=cells,minimum_clean_recall_met=cells[0]["defect_recall"]>=.8,
            naive_each_cell={"all_normal":counts(vy,np.zeros(len(vy),dtype=bool)),"all_defect":counts(vy,np.ones(len(vy),dtype=bool))}))
    print(json.dumps(cells),flush=True)
    if stage=="candidate":
        baseline=read(out/"K0_study.json")["validation_cells"]
        issues=[]
        for a,b in zip(baseline,cells,strict=True):
            c,s=b["condition"],b["seed"];margin=.02 if c=="clean" else .05
            for ok,reason in [(b["defect_recall"]>=.8,"minimum recall"),(b["normal_fpr"]<=.1,"maximum FPR"),
                (a["defect_recall"]-b["defect_recall"]<=margin+1e-12,"recall loss vs K0"),
                (b["normal_fpr"]-a["normal_fpr"]<=margin+1e-12,"FPR rise vs K0")]:
                if not ok: issues.append(dict(condition=c,seed=s,criterion=reason))
        d={k:sum(b[k]-a[k] for a,b in zip(baseline,cells)) for k in ["FN","FP"]}
        if not (max(d.values())<=0 and min(d.values())<0):
            issues.append(dict(condition="all",criterion="No strict FN/FP Pareto improvement"))
        files=["protocol.json","protocol_receipt.json","hypothesis.json","models/K0.joblib","models/K1.joblib",
               "K0_study.json","K1_study.json","K0_validation_diagnostics.csv","K1_validation_diagnostics.csv","study.py"]
        dump(out/"selection.json",dict(schema="steeltrust.binary.selection.v1",created_utc=now(),
                development_scope="train and validation only; no official test model access",
                test_eligible=not issues,eligible=not issues,validation_gate_reasons=issues,delta_total_K1_minus_K0=d,
                thresholds={"K0":read(out/"K0_study.json")["threshold"],"K1":t},
                frozen_files={n:SHA(SCRIPT if n=="study.py" else out/n) for n in files},
                encoder_sha256=model.ENCODER_SHA,split_manifest_sha256=SHA(MANIFEST),
                model_source_sha256=SHA(model.__file__),input_source_sha256=SHA(inspection_inputs.__file__),
                runtime={"python":platform.python_version(),"system":platform.system(),"device":"cpu",**VERSIONS}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["baseline","candidate"])
    parser.add_argument("--out",type=Path,default=model.RUN)
    parser.add_argument("--cache",type=Path,required=True,help="temporary feature directory outside release")
    a=parser.parse_args();run(a.stage,a.out.resolve(),a.cache.resolve())
