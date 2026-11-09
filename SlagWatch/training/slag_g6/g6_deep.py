"""Optional SteelTrust G6 supervised neural benchmark on B-relative spectral features.

Deep training is *experimental*: it is not certified to improve G6 ExtraTrees.
The B reference is independently confirmed and target-label independent.
Every held-out test group is excluded from scaler, training, early stop and cutoff.
"""
from __future__ import annotations
import argparse, json, os, random
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn import functional as F
from g6 import load_ssfd, dataset, threshold_for_target_fpr, stats, write_csv, write_json, FS, G6_VERSION

class Block(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.seq=nn.Sequential(nn.Linear(size,size),nn.LayerNorm(size),nn.GELU(),nn.Dropout(dropout),
                               nn.Linear(size,size),nn.Dropout(dropout))
    def forward(self,x):return F.gelu(x+self.seq(x))

class G6ContrastiveMLP(nn.Module):
    """Supervised deep detector on group-normalized 75-d log-spectral contrasts."""
    def __init__(self,dim=75,hidden=128,dropout=.25):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim,hidden),nn.LayerNorm(hidden),nn.GELU(),
                               Block(hidden,dropout),Block(hidden,dropout),
                               nn.Linear(hidden,64),nn.GELU(),nn.Dropout(dropout),nn.Linear(64,1))
    def forward(self,x):return self.net(x).squeeze(1)


def set_seed(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(s)

@torch.inference_mode()
def prob(model,x,device,batch=128):
    model.eval();p=[]
    for i in range(0,len(x),batch):
        logits=model(torch.from_numpy(x[i:i+batch].astype('float32')).to(device))
        p.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return np.asarray(p,dtype=float)


def train_one(X,y,groups,tr,va,device,seed,epochs=150,patience=22,batch=64,lr=.001,dropout=.25,out=None):
    set_seed(seed)
    if set(groups[tr]) & set(groups[va]):raise ValueError('Training/validation occurrence leakage')
    scaler=StandardScaler().fit(X[tr])
    a=scaler.transform(X[tr]).astype('float32');v=scaler.transform(X[va]).astype('float32')
    model=G6ContrastiveMLP(dim=X.shape[1],dropout=dropout).to(device)
    optim=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.03)
    lossfn=nn.BCEWithLogitsLoss()
    best=float('inf');best_state=None;bad=0;best_epoch=0
    for epoch in range(1,epochs+1):
        model.train();order=np.random.permutation(len(a))
        for batch_idx in range(0,len(order),batch):
            ids=order[batch_idx:batch_idx+batch]
            xx=torch.tensor(a[ids],device=device)
            yy=torch.tensor(y[tr][ids].astype('float32'),device=device)
            # Feature-space jitter is TRAIN-ONLY, no validation/test perturbation.
            xx=xx+torch.randn_like(xx)*.03
            optim.zero_grad(set_to_none=True)
            logits=model(xx)
            loss=lossfn(logits,yy)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3.0);optim.step()
        model.eval()
        val_logits=model(torch.from_numpy(v).to(device))
        val=float(lossfn(val_logits,torch.tensor(y[va].astype('float32'),device=device)).item())
        if val<best-1e-4:
            best=val;best_epoch=epoch;bad=0
            best_state={k:p.detach().cpu().clone() for k,p in model.state_dict().items()}
        else:bad+=1
        if bad>=patience:break
    model.load_state_dict(best_state)
    if out:
        out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'state_dict':best_state,'scaler_mean':scaler.mean_.astype('float32'),
                    'scaler_scale':scaler.scale_.astype('float32'),
                    'architecture':'G6ContrastiveMLP','dropout':dropout,'epoch':best_epoch,
                    'train_groups':sorted(set(groups[tr].tolist())),
                    'val_groups':sorted(set(groups[va].tolist()))},out)
    return model,scaler,best_epoch


def main(args):
    dev=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else args.device if args.device!='auto' else 'cpu')
    if dev.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    print('G6 deep',torch.__version__,'device',dev,'GPU',torch.cuda.get_device_name() if dev.type=='cuda' else 'none',flush=True)
    torch.set_num_threads(min(6,max(1,os.cpu_count() or 4)))
    records=load_ssfd(Path(args.data));X,y,groups,meta=dataset(records)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    rows=[];fold_metrics=[]
    for fold,(trva,te) in enumerate(GroupKFold(4).split(X,y,groups)):
        if args.fold is not None and fold!=args.fold:continue
        tr_groups=np.unique(groups[trva]);rng=np.random.default_rng(47+fold)
        val_groups=rng.permutation(tr_groups)[:max(2,len(tr_groups)//4)]
        va=trva[np.isin(groups[trva],val_groups)];tr=trva[~np.isin(groups[trva],val_groups)]
        assert not(set(groups[tr])&set(groups[va]) or set(groups[te])&set(groups[tr]) or set(groups[te])&set(groups[va]))
        vals=[];tests=[]
        for s in range(args.seeds):
            seed=47000+fold*100+s
            model,scaler,ep=train_one(X,y,groups,tr,va,dev,seed,args.epochs,args.patience,args.batch,args.lr,
                                      out=out/f'fold{fold}_seed{seed}.pt')
            vals.append(prob(model,scaler.transform(X[va]).astype('float32'),dev))
            tests.append(prob(model,scaler.transform(X[te]).astype('float32'),dev))
            print(f'  fold {fold} seed {seed}: best_epoch={ep}',flush=True)
        pv=np.mean(vals,axis=0);pt=np.mean(tests,axis=0)
        cut=threshold_for_target_fpr(y[va],pv,args.target_fpr)
        fold_metrics.append({'fold':fold,'train_groups':sorted(set(groups[tr].tolist())),
                             'val_groups':sorted(set(groups[va].tolist())),
                             'test_groups':sorted(set(groups[te].tolist())),
                             'threshold':cut,'test_auc_exploratory':roc_auc_score(y[te],pt)})
        for j,i in enumerate(te):
            rows.append({'index':int(i),'fold':fold,'group':int(groups[i]),'phase':meta[i]['phase'],
                         'start_s':meta[i]['start_s'],'true_slag':int(y[i]),'score':float(pt[j]),
                         'threshold':float(cut),'pred_slag':int(pt[j]>=cut)})
    rows.sort(key=lambda r:r['index'])
    write_csv(out/'oof_predictions.csv',rows)
    completed=args.fold is None and len(rows)==len(X)
    summary={'protocol':'B2','architecture':'G6 contrastive residual MLP','cv_complete':completed,
             'source_dataset':'SSFD v3','n_independent_occurrences':16,
             'notes':['Held-out groups excluded from scaler/training/threshold fitting',
                      'Validation used for both early stopping and cutoff; no claim of final unbiased production performance',
                      'Not automatically adopted unless held-out performance beats G5/G6 ExtraTrees',
                      'Original SSFD groups have been reused through multiple G1-G6 iterations']}
    if completed:summary.update(stats(rows))
    write_json(out/'summary.json',summary);write_json(out/'fold_metrics.json',fold_metrics)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description='G6 optional deep model candidate, group CV only')
    ap.add_argument('--data',default='data/SSFD.zip');ap.add_argument('--output',default='runs/g6_deep')
    ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    ap.add_argument('--epochs',type=int,default=150);ap.add_argument('--patience',type=int,default=22)
    ap.add_argument('--seeds',type=int,default=3);ap.add_argument('--fold',type=int,choices=[0,1,2,3]);ap.add_argument('--batch',type=int,default=64)
    ap.add_argument('--lr',type=float,default=.001);ap.add_argument('--target-fpr',type=float,default=.10)
    a=ap.parse_args()
    if a.epochs<1 or a.seeds<1 or a.patience<1 or not 0<a.target_fpr<1:ap.error('Invalid training config')
    main(a)
