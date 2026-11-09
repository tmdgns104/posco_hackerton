"""SteelTrust G6: paired B-reference spectral features and strict group CV.

SSFD has only 16 independent casting occurrences. Predictions are RESEARCH ONLY.
B/S classification requires the first 2.5s of the same occurrence to be
independently *confirmed* pre-slag (B). This is NOT a free-running B detector.
Do not make industrial interlock decisions from these models.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, math, re, zipfile
from pathlib import Path
import joblib
import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold

FS = 6400
WIDTH = 3200
REF_LEN = 16000
FEATURE_NAMES = [f'{a}_{i}' for a in ('x','y','z') for i in range(25)]  # 10+9+6/axis
G6_VERSION = 'G6B_refspec_v1'


def sha(path: Path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):d.update(b)
    return d.hexdigest()


def load_csv(raw: bytes, name: str):
    arr=np.genfromtxt(io.BytesIO(raw),delimiter=',',names=True,dtype=np.float64,encoding=None)
    if not arr.dtype.names or not {'x','y','z'}.issubset(arr.dtype.names):
        raise ValueError(f'{name}: CSV must have x, y, z columns')
    x=np.stack([arr[v] for v in ('x','y','z')],axis=-1)
    if x.ndim!=2 or len(x)<WIDTH or not np.isfinite(x).all():
        raise ValueError(f'{name}: need >=3200 rows of finite x,y,z')
    return x.astype('float32')


def load_ssfd(path: Path):
    pat=re.compile(r'^([EBS])-(\d+)\.csv$',re.I)
    p=Path(path)
    if p.is_file() and zipfile.is_zipfile(p):
        with zipfile.ZipFile(p) as z:
            content=[(Path(n).name,z.read(n)) for n in z.namelist() if pat.fullmatch(Path(n).name)]
    elif p.is_dir():
        content=[(f.name,f.read_bytes()) for f in p.rglob('*.csv') if pat.fullmatch(f.name)]
    else:raise FileNotFoundError(str(p))
    result={}
    for filename,raw in content:
        m=pat.fullmatch(filename);key=(m.group(1).upper(),int(m.group(2)))
        if key in result:raise ValueError('Duplicate SSFD stage/group: '+str(key))
        result[key]=load_csv(raw,filename)
    expected={(s,i) for i in range(1,17) for s in 'EBS'}
    if result.keys()!=expected or any(len(x)!=32000 for x in result.values()):
        raise ValueError('Expected exactly 16 groups x E/B/S with 32000 samples each')
    return result


def feature(window):
    """75-d per-window log power and robust distribution features.

    Learned group calibration is ONLY derived from confirmed reference (B),
    never S or from test-group labels; model-level normalization not fitted on test.
    """
    x=np.asarray(window,dtype=np.float64)
    if x.shape!=(WIDTH,3) or not np.isfinite(x).all():raise ValueError('window must be 3200 x 3 finite values')
    x=x-x.mean(axis=0)
    def bandpower(nperseg,bands):
        f,p=welch(x,fs=FS,nperseg=nperseg,noverlap=nperseg//2,axis=0)
        value=np.stack([np.log(np.maximum(p[(f>=a)&(f<b)].mean(axis=0),1e-15)) for a,b in bands],axis=0)
        return value.T
    band1=[(0,25),(25,50),(50,100),(100,200),(200,400),(400,800),(800,1200),(1200,1600),(1600,2200),(2200,3201)]
    band2=[(0,20),(20,40),(40,80),(80,160),(160,320),(320,640),(640,1280),(1280,2200),(2200,3201)]
    a=bandpower(512,band1); b=bandpower(1024,band2)
    d=np.diff(x,axis=0)
    v=np.stack([np.log(x.std(axis=0)+1e-9),np.log(d.std(axis=0)+1e-9),
                np.log(np.median(np.abs(x),axis=0)+1e-9),np.log(np.quantile(np.abs(x),.95,axis=0)+1e-9),
                np.clip(skew(x,axis=0),-20,20),np.clip(kurtosis(x,axis=0),-20,20)],axis=0).T
    z=np.r_[a.ravel(),b.ravel(),v.T.ravel()]
    if z.size!=75 or not np.isfinite(z).all():raise ValueError('Nonfinite output feature')
    return z.astype(np.float64)


def baseline(reference):
    ref=np.asarray(reference)
    if len(ref)<REF_LEN:raise ValueError('Need confirmed 2.5s pre-slag B reference at 6400Hz')
    return np.median(np.stack([feature(ref[i:i+WIDTH]) for i in range(0,REF_LEN,WIDTH)]),axis=0)


def dataset(records):
    xs=[];ys=[];groups=[];meta=[]
    for group in range(1,17):
        base=baseline(records['B',group])
        for phase in 'BS':
            recording=records[phase,group]
            for start in range(0,len(recording)-WIDTH+1,WIDTH):
                if phase=='B' and start<REF_LEN:continue
                xs.append(feature(recording[start:start+WIDTH])-base)
                ys.append(int(phase=='S'));groups.append(group)
                meta.append({'group':group,'phase':phase,'start_s':start/FS})
    return np.stack(xs),np.asarray(ys,np.int64),np.asarray(groups,np.int64),meta


def make_model():
    return ExtraTreesClassifier(n_estimators=130,min_samples_leaf=3,max_features=.8,
                                random_state=47,class_weight='balanced',n_jobs=2)


def threshold_for_target_fpr(y,prob,target_fpr=.10):
    """Max validation TP at FPR<=target, tied on fewer FP, then highest cutoff."""
    y=np.asarray(y,dtype=int);p=np.asarray(prob,dtype=float)
    if len(y)!=len(p) or set(np.unique(y))!={0,1} or not 0<target_fpr<1:
        raise ValueError('Binary validation truth and 0 < target_fpr < 1 required')
    if not np.isfinite(p).all():raise ValueError('Nonfinite validation score')
    budget=math.floor((y==0).sum()*target_fpr)
    best=None; cut=None
    for t in np.r_[np.unique(p),np.nextafter(p.max(),np.inf)]:
        d=p>=t;fp=int((d&(y==0)).sum());tp=int((d&(y==1)).sum())
        if fp<=budget:
            key=(tp,-fp,float(t))
            if best is None or key>best:best=key;cut=float(t)
    return cut


def predict_oof(x,y,group,target_fpr=.1,folds=4,inner_folds=3):
    rows=[];reports=[]
    for fold,(tr,te) in enumerate(GroupKFold(folds).split(x,y,group)):
        train_groups=set(group[tr]);test_groups=set(group[te]);assert train_groups.isdisjoint(test_groups)
        train_oof=np.full(len(tr),np.nan)
        for it,iv in GroupKFold(inner_folds).split(x[tr],y[tr],group[tr]):
            m=make_model().fit(x[tr[it]],y[tr[it]])
            train_oof[iv]=m.predict_proba(x[tr[iv]])[:,1]
        assert np.isfinite(train_oof).all()
        cut=threshold_for_target_fpr(y[tr],train_oof,target_fpr)
        model=make_model().fit(x[tr],y[tr]);p=model.predict_proba(x[te])[:,1]
        for j,i in enumerate(te):
            rows.append({'index':int(i),'fold':fold,'group':int(group[i]),'true_slag':int(y[i]),
                         'score':float(p[j]),'threshold':float(cut),'pred_slag':int(p[j]>=cut)})
        reports.append({'fold':fold,'train_groups':sorted(int(v) for v in train_groups),
                        'test_groups':sorted(int(v) for v in test_groups),'threshold':cut,
                        'inner_oof_auc':float(roc_auc_score(y[tr],train_oof)),
                        'test_auc_exploratory':float(roc_auc_score(y[te],p))})
    rows.sort(key=lambda r:r['index'])
    return rows,reports


def stats(rows):
    y=np.asarray([r['true_slag'] for r in rows]);p=np.asarray([r['pred_slag'] for r in rows]);score=np.asarray([r['score'] for r in rows])
    return {'n':int(len(y)),'n_slag':int(y.sum()),'n_non_slag':int((1-y).sum()),
            'slag_tp':int(((y==1)&(p==1)).sum()),'slag_fn':int(((y==1)&(p==0)).sum()),
            'non_slag_fp':int(((y==0)&(p==1)).sum()),'non_slag_tn':int(((y==0)&(p==0)).sum()),
            'slag_recall':float(p[y==1].mean()),'non_slag_fpr':float(p[y==0].mean()),
            'roc_auc':float(roc_auc_score(y,score)),'average_precision':float(average_precision_score(y,score))}


def write_csv(path,rows):
    if not rows:raise ValueError('No rows to save')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


def train(args):
    data=load_ssfd(Path(args.data));X,y,groups,meta=dataset(data)
    rows,folds=predict_oof(X,y,groups,args.target_fpr)
    for row in rows:row.update(meta[row['index']])
    result=stats(rows)
    ref_g5=None
    if Path(args.compare_csv).exists():
        with Path(args.compare_csv).open(encoding='utf-8-sig',newline='') as f:
            other=list(csv.DictReader(f))
        lookup={(int(r['group']),r['stage'],float(r['start_s'])):r for r in other}
        yold=[];pold=[];newidx=[];fixed=[];new_errors=[]
        for r in rows:
            key=(r['group'],r['phase'],r['start_s'])
            if key not in lookup:raise ValueError('Baseline mismatch in '+str(key))
            o=lookup[key];assert int(o['true_slag'])==r['true_slag']
            yold.append(int(o['true_slag']));pold.append(int(o['pred_slag']))
            if r['true_slag']==r['pred_slag'] and int(o['pred_slag'])!=r['true_slag']:fixed.append(key)
            if r['true_slag']!=r['pred_slag'] and int(o['pred_slag'])==r['true_slag']:new_errors.append(key)
        ref_g5={'existing_g5_tp':sum(int(t==1 and p==1) for t,p in zip(yold,pold)),
                'existing_g5_fp':sum(int(t==0 and p==1) for t,p in zip(yold,pold)),
                'fixed_previous_errors':len(fixed),'new_errors':len(new_errors),
                'fixed_examples':list(map(str,fixed[:12])), 'new_error_examples':list(map(str,new_errors[:12]))}
    o=Path(args.output);o.mkdir(parents=True,exist_ok=True)
    write_csv(o/'oof_predictions.csv',rows)
    write_json(o/'cv_metrics.json',{'model':G6_VERSION,'protocol':'B2',**result,'folds':folds,'comparison':ref_g5,
                                   'reporting_warning':'Repeat development on same 16 groups. NOT independent model certification.'})
    # Full fit uses all 16 groups; its in-sample scores must not be reported as test accuracy.
    calibration=np.full(len(y),np.nan)
    for tr,te in GroupKFold(4).split(X,y,groups):
        m=make_model().fit(X[tr],y[tr]);calibration[te]=m.predict_proba(X[te])[:,1]
    threshold=threshold_for_target_fpr(y,calibration,args.target_fpr)
    final=make_model().fit(X,y)
    archive={'estimator':final,'threshold':threshold,'fs':FS,'width':WIDTH,'ref_len':REF_LEN,
             'features':75,'name':G6_VERSION,'data_sha256':sha(Path(args.data)) if Path(args.data).is_file() else None,
             'training_groups':list(range(1,17)),'target_validation_fpr':args.target_fpr,
             'known_B_reference_required':True,
             'warning':'Research only: model fit on all 16 SSFD occurrences, no independent deployment test.'}
    joblib.dump(archive,o/'G6B_refspec_extratrees.joblib',compress=3)
    write_json(o/'final_metadata.json',{k:v for k,v in archive.items() if k!='estimator'})
    print(json.dumps({'validation':result,'G5_comparison':ref_g5,'output':str(o)},indent=2))


def infer(args):
    bundle=joblib.load(args.model) # Never load untrusted pickle-based files.
    if bundle.get('name')!=G6_VERSION or bundle.get('fs')!=FS or bundle.get('features')!=75:
        raise ValueError('Incompatible model metadata')
    ref=load_csv(Path(args.reference).read_bytes(),str(args.reference))
    tar=load_csv(Path(args.target).read_bytes(),str(args.target))
    if len(ref)<REF_LEN:raise ValueError('Confirmed B reference must have 16000 or more samples')
    base=baseline(ref);limit=len(tar)-WIDTH+1
    if limit<=0:raise ValueError('target must have >=3200 samples')
    rows=[];continuous=0
    for start in range(0,limit,WIDTH):
        feat=(feature(tar[start:start+WIDTH])-base)[None,:]
        score=float(bundle['estimator'].predict_proba(feat)[0,1]);flag=score>=bundle['threshold']
        continuous=continuous+1 if flag else 0
        rows.append({'start_s':start/FS,'slag_score':score,'model_score_cutoff':bundle['threshold'],
                     'raw_s_flag':int(flag),'persistent_alarm':int(continuous>=args.persistence),
                     'reference_provided_independently':True,'research_only':True})
    write_csv(Path(args.output),rows)
    print('Saved',args.output,'raw flags:',sum(r['raw_s_flag'] for r in rows),'persistent flags:',sum(r['persistent_alarm'] for r in rows))


def cli():
    p=argparse.ArgumentParser(description='SteelTrust G6B paired spectral defect detection - research only')
    sub=p.add_subparsers(dest='command',required=True)
    tr=sub.add_parser('train');tr.add_argument('--data',default='data/SSFD.zip');tr.add_argument('--output',default='runs/g6_extratrees')
    tr.add_argument('--target-fpr',type=float,default=.1);tr.add_argument('--compare-csv',default='reports/G5_B_reference_oof.csv')
    q=sub.add_parser('predict');q.add_argument('--model',default='trained_models/G6B_refspec_extratrees.joblib');q.add_argument('--reference',required=True)
    q.add_argument('--target',required=True);q.add_argument('--persistence',type=int,default=2);q.add_argument('--output',default='runs/g6_prediction.csv')
    a=p.parse_args()
    if a.command=='train':
        if not 0<a.target_fpr<1:p.error('target-fpr must be between 0 and 1')
        train(a)
    else:
        if a.persistence<1:p.error('persistence must be >= 1')
        infer(a)

if __name__=='__main__':cli()
