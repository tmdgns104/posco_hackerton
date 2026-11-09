"""Shared image preprocessing, frozen feature extraction and six-class inference."""
from functools import lru_cache
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image
import torch
from torchvision.models import resnet18
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .surface_data import CFG,CLASSES,DATA,RUN,TASK,array_sha,sha,stable_id

WEIGHTS=RUN/"models/resnet18-f37072fd.pth"
MEAN=torch.tensor([.485,.456,.406])[None,:,None,None]
STD=torch.tensor([.229,.224,.225])[None,:,None,None]
BATCH=32


def read_raw(row):
    data=(DATA/row.file_name).read_bytes()
    if sha(data)!=row.file_sha256:raise ValueError("원본 이미지 해시 불일치")
    import io
    with Image.open(io.BytesIO(data)) as image:
        pixels=np.array(image)
    if pixels.shape!=(200,200) or array_sha(pixels)!=row.pixels_sha256:raise ValueError("원본 픽셀 불일치")
    return pixels.astype(np.float32)/np.float32(255)


def case_seed(parent,split,sigma,seed):
    token=f"{TASK}|{split}|{parent}|{sigma:.2f}|{seed}".encode()
    return int(sha(token)[:16],16)


def disturb(raw,parent,split,sigma,seed):
    if sigma==0:return raw.copy()
    noise=np.random.default_rng(case_seed(parent,split,sigma,seed)).normal(0,sigma,raw.shape).astype(np.float32)
    return np.ascontiguousarray(np.clip(raw+noise,0,1),dtype="<f4")


def prepare(raw,policy):
    if raw.dtype!=np.float32 or raw.shape!=(200,200) or not np.isfinite(raw).all() or raw.min()<0 or raw.max()>1:
        raise ValueError("Expected finite float32 [0,1] native 200x200 grayscale")
    kernel=CFG["preprocessing"][policy]
    if kernel is None:return np.ascontiguousarray(raw)
    k,s=kernel
    return cv2.GaussianBlur(raw,(k,k),sigmaX=s,sigmaY=s,borderType=cv2.BORDER_REFLECT_101)


@lru_cache(maxsize=1)
def encoder():
    torch.set_num_threads(4);cv2.setNumThreads(1)
    if sha(WEIGHTS.read_bytes())!=CFG["weight_sha256"]:raise ValueError("ResNet18 가중치 불일치")
    net=resnet18(weights=None)
    net.load_state_dict(torch.load(WEIGHTS,map_location="cpu",weights_only=True))
    net.fc=torch.nn.Identity()
    return net.eval().requires_grad_(False).to(memory_format=torch.channels_last)


@torch.inference_mode()
def extract(raws,policy):
    net=encoder(); matrices=[];hashes=[]
    for start in range(0,len(raws),BATCH):
        processed=[prepare(x,policy) for x in raws[start:start+BATCH]]
        hashes.extend(array_sha(x) for x in processed)
        grey=torch.from_numpy(np.stack([cv2.resize(x,(224,224),interpolation=cv2.INTER_LINEAR) for x in processed]))[:,None]
        rgb=((grey-MEAN)/STD).contiguous(memory_format=torch.channels_last)
        matrices.append(net(rgb).cpu().numpy())
    return np.concatenate(matrices).astype(np.float64),hashes


def fit_head(raws,labels,policy,source_id):
    features,_=extract(raws,policy)
    head=make_pipeline(StandardScaler(),LogisticRegression(C=1,solver="lbfgs",max_iter=2000,class_weight="balanced"))
    with threadpool_limits(limits=2,user_api="blas"):
        head.fit(features,labels)
    if list(head.classes_)!=CLASSES or head[-1].n_iter_.max()>=2000:raise ValueError("Class mapping or convergence failure")
    joblib.dump(dict(head=head,policy=policy,train_id=source_id,encoder_sha=CFG["weight_sha256"],
                     input_policy=CFG["preprocessing"][policy],protocol_sha=sha((RUN/"protocol.json").read_bytes())),
                RUN/f"models/{policy}.joblib",compress=3)


class Detector:
    def __init__(self,policy):
        self.policy=policy;path=RUN/f"models/{policy}.joblib"
        self.head_sha=sha(path.read_bytes())
        b=joblib.load(path)
        protocol_sha=sha((RUN/"protocol.json").read_bytes())
        if (b["policy"]!=policy or b["encoder_sha"]!=CFG["weight_sha256"] or b["input_policy"]!=CFG["preprocessing"][policy]
                or b["protocol_sha"]!=protocol_sha or list(b["head"].classes_)!=CLASSES):
            raise ValueError("모델과 전처리 버전 불일치")
        self.head=b["head"]
        self.identity=dict(model_id=policy,head_sha256=self.head_sha,encoder_sha256=b["encoder_sha"],train_id=b["train_id"],
                           preprocessing_sha256=stable_id([protocol_sha,policy,b["input_policy"]]))
        self.run_id=stable_id(self.identity)

    def infer(self,raws,*,cache=None,case_keys=None):
        if cache is None:
            features,processed=extract(raws,self.policy)
        else:
            if case_keys is None or len(case_keys)!=len(raws):
                raise ValueError("Feature cache requires one complete case key per image")
            keys=[(k,self.run_id,array_sha(raw)) for k,raw in zip(case_keys,raws,strict=True)]
            entries=[cache.get(k) for k in keys]
            missing=[i for i,value in enumerate(entries) if value is None]
            if missing:
                matrix,hashes=extract([raws[i] for i in missing],self.policy)
                for i,vec,h in zip(missing,matrix,hashes,strict=True):
                    entries[i]=(vec,h);cache[keys[i]]=entries[i]
                    if len(cache)>4096:
                        cache.pop(next(iter(cache)),None)
            features=np.stack([v[0] for v in entries])
            processed=[v[1] for v in entries]
        with threadpool_limits(limits=2,user_api="blas"):
            logits=self.head.decision_function(features)
        return [dict(raw_input_hash=array_sha(raw),processed_input_hash=h,scores=z.tolist(),
                     prediction=CLASSES[int(np.argmax(z))]) for raw,h,z in zip(raws,processed,logits,strict=True)]
