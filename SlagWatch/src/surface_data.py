"""Source identity, parent grouping and fixed class-stratified split before model fitting."""
import hashlib
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_RUN=ROOT/"experiments/surface_v1"
RUN=Path(os.environ.get("STEELTRUST_SURFACE_RUN",DEFAULT_RUN)).resolve()
DATA=ROOT/"data/neu_cls"
CFG=json.loads((RUN/"protocol.json").read_text(encoding="utf8"))
TASK=CFG["task_id"]
UNIT=CFG["sample_unit"]
CLASSES=sorted(CFG["classes"].values())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def array_sha(x):
    return sha(np.ascontiguousarray(x).tobytes())


def write_json(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8")


def stable_id(obj):
    return sha(json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode())


def orientations(x):
    for k in range(4):
        a=np.rot90(x,k)
        yield a
        yield a[:,::-1]


def normalized(x):
    f=x.astype(np.float32).ravel()
    f-=f.mean()
    return f/max(np.linalg.norm(f),1e-8)


def build_manifest():
    prefix={k.lower():v for k,v in CFG["classes"].items()}
    rows=[]; images=[]; seen=set()
    for path in sorted(DATA.glob("*.bmp")):
        m=re.fullmatch(r"([A-Za-z]+)_(\d+)",path.stem)
        if m is None: raise ValueError(f"Unexpected NEU filename {path}")
        label=prefix[m[1].lower()];num=int(m[2])
        if not 1<=num<=300 or (label,num) in seen:raise ValueError("Missing unique original ID")
        seen.add((label,num))
        with Image.open(path) as im:
            if im.mode!="L" or im.size!=(200,200):raise ValueError("Non-native NEU image")
            x=np.array(im)
        images.append(x)
        rows.append(dict(sample_id=path.stem,file_name=path.name,label=label,file_sha256=sha(path.read_bytes()),
                         pixels_sha256=array_sha(x),canonical_sha256=min(array_sha(v) for v in orientations(x))))
    assert len(rows)==1800 and len(seen)==1800
    parent=list(range(len(rows)))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    links=[]; exact={}
    def merge(i,j,reason):
        a,b=root(i),root(j)
        if a!=b:
            parent[max(a,b)]=min(a,b)
            links.append(dict(first=rows[i]["sample_id"],second=rows[j]["sample_id"],basis=reason))
    for i,r in enumerate(rows):
        j=exact.setdefault(r["canonical_sha256"],i)
        if j!=i:merge(i,j,"exact pixel or D4 permutation")
    thumb=[cv2.resize(x,(32,32),interpolation=cv2.INTER_AREA) for x in images]
    flat=np.stack([normalized(v) for v in thumb])
    for k in range(8):
        vs=np.stack([normalized(list(orientations(v))[k]) for v in thumb])
        for i,j in zip(*np.where(np.triu(vs@flat.T>=.985,1))):
            if root(i)!=root(j) and normalized(list(orientations(images[i]))[k])@normalized(images[j])>=.99:
                merge(int(i),int(j),"D4 and linear brightness near-duplicate (NCC)")
    df=pd.DataFrame(rows)
    df["parent_image_id"]=[rows[root(i)]["sample_id"] for i in range(len(rows))]
    rng=np.random.default_rng(42)
    groups=list(df.groupby("parent_image_id",sort=True))
    groups=[groups[i] for i in rng.permutation(len(groups))]
    groups.sort(key=lambda v:-len(v[1]))
    remaining=np.tile(np.array([180,60,60])[:,None],(1,len(CLASSES)))
    assignments={}
    for key,g in groups:
        counts=g.label.value_counts().reindex(CLASSES,fill_value=0).to_numpy()
        choices=np.flatnonzero((remaining>=counts).all(axis=1))
        if not len(choices):raise ValueError("Parent groups cannot fit target split capacities")
        weights=remaining[choices][:,counts>0].sum(axis=1).astype(float)
        slot=rng.choice(choices,p=weights/weights.sum());remaining[slot]-=counts
        assignments[key]=("train","validation","test")[slot]
    df["split"]=df.parent_image_id.map(assignments)
    assert not remaining.any() and df.groupby("parent_image_id").split.nunique().max()==1
    df=df.assign(task_id=TASK,sample_unit=UNIT,population=1800,width=200,height=200)
    df.to_csv(RUN/"split_manifest.csv",index=False)
    write_json(RUN/"duplicates.json",dict(pixels_unique=df.pixels_sha256.nunique(),parents=df.parent_image_id.nunique(),
                                          retained_images=len(df),links=links,split_counts=df.split.value_counts().to_dict()))
    print(df.groupby(["split","label"]).size(), "parents",df.parent_image_id.nunique(),flush=True)

if __name__=="__main__":build_manifest()
