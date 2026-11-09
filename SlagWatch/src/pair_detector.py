"""One raw-window adapter for batch evaluation, replay, Streamlit and the CLI."""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from .data import SIZE
from .detector import Encoder, REP, RUN as OLD_RUN, model_bundle
from .hi_models import predict_matrix
from .quality import quality
from .pair_runs import CNN, ROCKET, RUNS
from .raw_cnn import CFG, RUN, load_net, score_batch, tensor_waves


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array,dtype="<f8").tobytes()).hexdigest()


def identifier(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(",",":")).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Detector:
    def __init__(self, name, fold, run=None, manifest_hash=None):
        run=Path(run) if run is not None else (ROCKET if name in ("R0","R1") else CNN)
        self.rocket_bundle=None
        self.name=name
        self.fold=str(fold)
        if name in ("R0","R1") and run.name==ROCKET.name:
            from .rocket_detector import representation_hash, versions
            path=run/f"models/{self.fold}.joblib"
            bundle_sha=file_hash(path)
            if manifest_hash is not None:
                manifest=run/"manifest.json"
                if file_hash(manifest)!=manifest_hash or json.loads(manifest.read_text())["checkpoints"][path.name]!=bundle_sha:
                    raise ValueError("MiniRocket 실행 또는 체크포인트 해시 불일치")
            b=joblib.load(path)
            if (b["fold_id"]!=self.fold or b["role"]!=("full" if self.fold=="full" else "outer") or
                    b["protocol_sha256"]!=file_hash(run/"protocol.json") or
                    representation_hash(b["rocket"],b["scaler"])!=b["feature_hash"]):
                raise ValueError("MiniRocket 분할·특징·프로토콜 불일치")
            installed=versions()
            for pkg in ("aeon","numpy","scikit-learn","numba"):
                if b["versions"][pkg]!=installed[pkg]:
                    raise ValueError(f"{pkg} 버전 불일치: {b['versions'][pkg]} 필요")
            self.rocket_bundle=b
            self.bundle={"threshold":b["thresholds"][name]}
            self.train,self.test=b["train_groups"],b["test_groups"]
            self.net=None
        elif name in ("A1","J") and self.fold!="full":
            path=OLD_RUN/f"models/fold_{int(fold)}.joblib"
            expected=json.loads((OLD_RUN/"manifest.json").read_text())["artifacts_sha256"][path.name]
            if file_hash(path)!=expected:
                raise ValueError("기존 모델 해시가 일치하지 않습니다.")
            self.legacy=joblib.load(path)
            self.bundle=model_bundle(self.legacy,name,.15)
            self.test=self.legacy["test_groups"]
            self.train=self.legacy["train_groups"]
            self.net=None
        elif name in ("C0","C1"):
            path=RUN/f"models/{fold}_{name}.pt"
            self.net,self.bundle=load_net(path)
            if self.bundle["fold_id"]!=self.fold or self.bundle["model_id"]!=name:
                raise ValueError("체크포인트의 모델 또는 fold가 일치하지 않습니다.")
            self.test=self.bundle["test_groups"];self.train=self.bundle["train_groups"]
        else:
            raise ValueError(f"지원하지 않는 모델/분할: {name}/{fold}")
        if set(self.test)&set(self.train):
            raise ValueError("평가 그룹이 학습에 포함됐습니다.")
        self.threshold=float(self.bundle["threshold"])
        self.sha=file_hash(path)
        if self.rocket_bundle is not None:
            head=self.rocket_bundle["heads"][name]
            self.sha=identifier([name,self.rocket_bundle["feature_hash"],digest(head.coef_),digest(head.intercept_),head.classes_.tolist()])
        self.run_id=identifier(dict(model=name,fold=self.fold,checkpoint=self.sha,threshold=self.threshold,
                                    pipeline="rocket-pair-v1" if self.rocket_bundle is not None else "raw-pair-v1"))
        self.identity=dict(model=name,fold=self.fold,model_sha256=self.sha,run_id=self.run_id,threshold=self.threshold)

        if self.rocket_bundle is not None:
            self.identity.update(bundle_sha256=bundle_sha,feature_sha256=b["feature_hash"])
    def infer(self, windows):
        valid=[]; indexed=[]
        for idx,w in enumerate(windows):
            status="BAD_SHAPE" if w is None else quality(np.asarray(w))
            indexed.append(dict(score=None,prediction=None,input_status=status or "VALID"))
            if not status:
                valid.append((idx,w))
        if valid:
            waves=[w for _,w in valid]
            if self.rocket_bundle is not None:
                from .rocket_detector import decisions, transform
                x=transform(self.rocket_bundle,waves)
                scores=decisions(self.rocket_bundle["heads"][self.name],x)
            elif self.net is not None:
                scores=score_batch(self.net,tensor_waves(waves))
            else:
                rep=REP[self.name]
                encoder=Encoder([rep],self.bundle["settings"])
                matrix=np.stack([encoder.step(w)[0][rep] for w in waves])
                scores,_=predict_matrix(self.bundle,matrix)
            for (idx,_),s in zip(valid,scores,strict=True):
                indexed[idx].update(score=float(s),prediction=int(float(s)>=self.threshold))
        return indexed


@lru_cache(maxsize=24)
def detector(name,fold,run=None,manifest_hash=None):
    torch.set_num_threads(3)
    return Detector(name,str(fold),run,manifest_hash)


def main():
    parser=argparse.ArgumentParser(description="Raw detector prediction or verified replay of a SteelTrust case.")
    file=parser.add_mutually_exclusive_group(required=True)
    file.add_argument("--csv");file.add_argument("--replay")
    parser.add_argument("--model",choices=["C0","C1","R0","R1"],help="Explicit experimental full-data checkpoint for --csv")
    args=parser.parse_args()
    if args.replay:
        from .pair_search import ReplaySet
        record=json.loads(Path(args.replay).read_text(encoding="utf8"))
        run=RUNS.get(record.get("study","cnn_v1"))
        if run is None:
            parser.error("Unknown replay study")
        _,_,actual=ReplaySet(run, file_hash(run/"manifest.json")).replay(record)
        print(json.dumps(actual,ensure_ascii=False));return
    data=pd.read_csv(args.csv)
    if list(data)!=["time","x","y","z"] or not len(data) or len(data)%SIZE:
        parser.error("Expected time,x,y,z and a positive multiple of 3200 rows")
    if args.model is None:
        parser.error("Select an experimental checkpoint explicitly with --model")
    run=ROCKET if args.model in ("R0","R1") else CNN
    d=detector(args.model,"full",run,file_hash(run/"manifest.json"))
    manifest=json.loads((run/"manifest.json").read_text(encoding="utf8"))
    if d.identity not in manifest["models"]:
        parser.error("전체 학습 모델과 배포 manifest가 일치하지 않습니다.")
    for i,result in enumerate(d.infer(data[["x","y","z"]].to_numpy(float).reshape(-1,SIZE,3))):
        print(json.dumps(dict(end_s=(i+1)/2,**d.identity,**result),allow_nan=False))


if __name__=="__main__":
    main()
