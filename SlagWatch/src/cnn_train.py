"""Training and inner selection only. This module never evaluates outer test windows."""
import argparse
import copy
import json
import statistics
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.nn import functional as F

from .data import load_dataset
from .hi_study import read_config, select_cut
from .raw_cnn import CFG, RUN, ROOT, PatchCNN, score_batch, tensor_waves


def fit(x,y,train,val,kind,epochs):
    torch.manual_seed(CFG["seed"])
    net=PatchCNN(kind)
    tr=x[train]
    net.axis_mean.copy_(tr.double().mean((0,1)).float())
    net.axis_std.copy_(tr.double().std((0,1),unbiased=False).clamp_min(1e-8).float())
    pos=(len(train)-y[train].sum())/y[train].sum()
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
    rng=torch.Generator().manual_seed(CFG["seed"])
    best,best_epoch,state=float("inf"),0,None
    losses=[]
    for epoch in range(1,epochs+1):
        net.train()
        for ids in torch.randperm(len(train),generator=rng).split(CFG["batch_size"]):
            batch=train[ids.numpy()]
            opt.zero_grad(set_to_none=True)
            loss=F.binary_cross_entropy_with_logits(net(x[batch]),y[batch],pos_weight=pos)
            loss.backward()
            opt.step()
        if val is None:
            continue
        net.eval()
        with torch.inference_mode():
            logits=torch.cat([net(b) for b in x[val].split(CFG["batch_size"])])
            metric=F.binary_cross_entropy_with_logits(logits,y[val],pos_weight=pos).item()
        losses.append(metric)
        if metric<best:
            best,best_epoch,state=metric,epoch,copy.deepcopy(net.state_dict())
        if epoch-best_epoch>=CFG["patience"]:
            break
    if val is not None:
        net.load_state_dict(state)
    return net.eval(), best_epoch if val is not None else epochs, losses


def inner_run(x,y,meta,split,kind):
    blocks=[]; history=[]
    for i,v in enumerate(split["inner"]):
        tr=np.flatnonzero(meta.group_id.isin(v["train"]))
        va=np.flatnonzero(meta.group_id.isin(v["val"]))
        assert set(v["train"]).isdisjoint(v["val"]) and set(v["train"]+v["val"])==set(split["train"])
        net,epoch,losses=fit(x,y,tr,va,kind,CFG["epochs"])
        s=score_batch(net,x[va])
        blocks.append(meta.iloc[va].assign(score=s.astype(np.float64),inner_id=i,model_id=kind,fold_id=split["fold_id"]))
        history.append(dict(train=v["train"],val=v["val"],epoch=epoch,validation_bce=losses))
        print(split["fold_id"],kind,"inner",i,"best",epoch,"/ ran",len(losses),flush=True)
    rows=pd.concat(blocks).sort_index()
    assert not rows.index.duplicated().any() and set(rows.index)==set(meta.index[meta.group_id.isin(split["train"])])
    cut=select_cut(rows.y_true,rows.score.to_numpy(dtype=float),.15)
    positives=rows.score>=cut["threshold"]
    metrics=dict(cut, fp_B=int(((rows.phase=="B")&positives).sum()),fp_E=int(((rows.phase=="E")&positives).sum()))
    for n in ("B","E"):
        pair=rows[rows.phase.isin(["S",n])]
        metrics["auc_S"+n]=float(roc_auc_score(pair.y_true,pair.score))
    return rows,history,metrics


def train_partition(x,y,meta,split,kinds):
    name=str(split["fold_id"]); records=[]; candidates={}
    for kind in kinds:
        rows,history,metrics=inner_run(x,y,meta,split,kind)
        epoch=int(statistics.median(v["epoch"] for v in history))
        tr=np.flatnonzero(meta.group_id.isin(split["train"]))
        net,_,_=fit(x,y,tr,None,kind,epoch)
        ck=dict(model_id=kind,fold_id=name,train_groups=split["train"],test_groups=split.get("test",[]),
                epoch=epoch,threshold=metrics["threshold"],state=net.state_dict())
        torch.save(ck,RUN/f"models/{name}_{kind}.pt")
        records.append(rows)
        candidates[kind]=dict(inner=metrics,history=history,refit_epochs=epoch)
    if kinds==["C1"]:
        previous=json.loads((RUN/f"fit_{name}.json").read_text())
        if previous["train"]!=split["train"] or "C0" not in previous["models"]:
            raise ValueError("C0 split missing or not aligned")
        candidates={"C0":previous["models"]["C0"],**candidates}
        old=pd.read_csv(RUN/f"inner_{name}.csv",float_precision="round_trip")
        records.insert(0,old[old.model_id=="C0"])
    winner=max(candidates,key=lambda k:(candidates[k]["inner"]["tp"],-candidates[k]["inner"]["fp"],k=="C0"))
    pd.concat(records).to_csv(RUN/f"inner_{name}.csv",index=False)
    result=dict(fold_id=name,train=split["train"],test=split.get("test",[]),models=candidates,choice=winner)
    (RUN/f"fit_{name}.json").write_text(json.dumps(result,indent=2))
    print("frozen",name,winner,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--c1-only",action="store_true",help="Reuse the frozen C0 inner run")
    args=parser.parse_args()
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    start=time.monotonic()
    meta,w,_=load_dataset(ROOT/"data/raw")
    x=tensor_waves(w); y=torch.tensor(meta.y_true.to_numpy(),dtype=torch.float32)
    folds=read_config()["folds"]
    full=[dict(train=sorted(map(int,meta.iloc[a].group_id.unique())),val=sorted(map(int,meta.iloc[b].group_id.unique())))
          for a,b in GroupKFold(3).split(meta,groups=meta.group_id)]
    for split in [*folds,dict(fold_id="full",train=list(range(1,17)),inner=full)]:
        train_partition(x,y,meta,split,["C1"] if args.c1_only else ["C0","C1"])
    print("training seconds",round(time.monotonic()-start),flush=True)


if __name__=="__main__":
    main()
