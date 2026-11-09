"""Native-resolution, lossless case export and replay in the same frozen classifiers."""
import argparse
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED,ZipFile

import numpy as np
import pandas as pd
from PIL import Image

from .surface_data import CFG,CLASSES,RUN,ROOT,TASK,UNIT,stable_id,sha,array_sha,write_json
from .surface_detector import Detector,WEIGHTS,read_raw,disturb,prepare,case_seed
from .surface_study import conditions,compare

CHANGES={"fixed":"고친 오분류","new_error":"새 오분류","both_wrong":"두 모델 모두 오분류","both_correct":"두 모델 모두 정분류"}
LABELS={"crazing":"망상 균열","inclusion":"개재물","patches":"반점","pitted_surface":"표면 패임",
        "rolled_in_scale":"압연 스케일","scratches":"긁힘"}


def finalize():
    from importlib.metadata import version
    import platform
    import torch
    status=json.loads((RUN/"test_status.json").read_text())
    files=["protocol.json","split_manifest.csv","duplicates.json","selection.json","test_status.json",
           "validation_discoveries.csv.gz","validation_discovered_inputs.zip"]
    splits=["validation","test"] if status["evaluated"] else ["validation"]
    files.extend(f"{s}_{name}" for s in splits for name in ("predictions.csv.gz","pairs.csv.gz","summary.csv","classes.csv","confusion.csv","changes.csv"))
    files.extend(f"models/{m}.joblib" for m in CFG["preprocessing"])
    for split in splits:
        pairs=pd.read_csv(RUN/f"{split}_pairs.csv.gz")
        keys=["task_id","sample_unit","evaluation","n_originals","after","parent_image_id","label","sigma","seed"]
        group=pairs.groupby(keys+["change"]).size().unstack(fill_value=0)
        group["n_group_images"]=group.sum(axis=1)
        name=f"{split}_by_parent.csv";group.reset_index().to_csv(RUN/name,index=False);files.append(name)
    seal=json.loads((RUN/"selection.json").read_text())
    hashes={f:sha((RUN/f).read_bytes()) for f in files}
    manifest=dict(task_id=TASK,sample_unit=UNIT,splits=splits,files=hashes,selected=seal["selected"],
                  techniques=CFG["preprocessing"],encoder_sha256=sha(WEIGHTS.read_bytes()),
                  n_originals=pd.read_csv(RUN/"split_manifest.csv").groupby("split").size().to_dict(),
                  versions={p:version(p) for p in ("torch","torchvision","opencv-python-headless","numpy","scikit-learn","Pillow","streamlit")},
                  os=platform.system(),python=platform.python_version(),device="cpu",cuda_available=torch.cuda.is_available())
    manifest["code_sha256"]={f:sha((ROOT/"src"/f).read_bytes()) for f in ("surface_data.py","surface_detector.py","surface_study.py")}
    manifest["run_id"]=stable_id(manifest)
    write_json(RUN/"manifest.json",manifest)


def load_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def key_for(run,row,sigma,seed):
    return (run,row.sample_id,row.parent_image_id,row.split,float(sigma),int(seed))


def png(x):
    buf=io.BytesIO();Image.fromarray(np.round(np.clip(x,0,1)*255).astype(np.uint8)).save(buf,format="PNG")
    return buf.getvalue()


def numpy_bytes(x):
    buf=io.BytesIO();np.save(buf,x,allow_pickle=False);return buf.getvalue()


def change_type(truth,old,new):
    if old==truth:return "both_correct" if new==truth else "new_error"
    return "fixed" if new==truth else "both_wrong"


class SurfaceRun:
    def __init__(self,expected_manifest=None):
        raw=(RUN/"manifest.json").read_bytes()
        if expected_manifest and sha(raw)!=expected_manifest:raise ValueError("실험 manifest 불일치")
        self.man=json.loads(raw);self.id=self.man["run_id"]
        if stable_id({k:v for k,v in self.man.items() if k!="run_id"})!=self.id:raise ValueError("실험 ID 불일치")
        for f,h in self.man["code_sha256"].items():
            if sha((ROOT/"src"/f).read_bytes())!=h:raise ValueError("실행 코드 해시 불일치 · "+f)
        self._checked=set()
        self.meta=pd.read_csv(self.checked("split_manifest.csv")).set_index("sample_id",drop=False)
        self.seal=load_json(self.checked("selection.json"))
        self.status=load_json(self.checked("test_status.json"))
        self.checked("protocol.json")
        self.models={};self.cache={}
        self.frames={};self._lookup={}

    def checked(self,relative):
        path=RUN/relative
        if relative not in self._checked:
            if sha(path.read_bytes())!=self.man["files"][relative]:raise ValueError(f"저장 결과 해시 불일치 · {relative}")
            self._checked.add(relative)
        return path

    def detector(self,name):
        if name not in CFG["preprocessing"]:raise ValueError("Unknown policy")
        if name not in self.models:
            self.checked(f"models/{name}.joblib")
            d=Detector(name)
            if d.identity!=self.seal["models"][name]:raise ValueError("평가된 모델과 불일치")
            self.models[name]=d
        return self.models[name]

    def table(self,split,name):
        if split not in self.man["splits"]:raise ValueError("미개방 시험 분할")
        key=(split,name)
        if key not in self.frames:
            ext=".csv.gz" if name in ("pairs","predictions") else ".csv"
            self.frames[key]=pd.read_csv(self.checked(f"{split}_{name}"+ext),float_precision="round_trip")
        return self.frames[key]

    def ensure_scope(self,split,after,sigma,seed):
        if split not in self.man["splits"] or after not in ("V1","V2") or (float(sigma),int(seed)) not in conditions(split):
            raise ValueError("분할·모델·교란 설정 불일치")
        if split=="test" and after!=self.seal["selected"]:raise ValueError("동결된 시험 후보 불일치")

    def pair_rows(self,split,after,sigma,seed):
        self.ensure_scope(split,after,sigma,seed)
        f=self.table(split,"pairs")
        return f[(f.after==after)&(f.sigma==sigma)&(f.seed==seed)].copy()

    def lookup(self,split):
        if split not in self._lookup:
            table=self.table(split,"predictions")
            self._lookup[split]=table.set_index(["sample_id","model_id","sigma","seed"],verify_integrity=True)
        return self._lookup[split]

    def verify_output(self,r,name,sigma,seed,out):
        ref=self.lookup(r.split).loc[(r.sample_id,name,sigma,seed)]
        if (ref.label!=r.label or ref.prediction!=out["prediction"] or ref.model_run_id!=self.detector(name).run_id
            or ref.raw_input_hash!=out["raw_input_hash"] or ref.processed_input_hash!=out["processed_input_hash"]
            or not np.allclose(json.loads(ref.scores),out["scores"],rtol=0,atol=CFG["score_atol"])):
            raise ValueError(f"평가 예측과 재추론 불일치 · {r.sample_id}/{name}")

    def live(self,after,sigma,seed,progress=None):
        self.ensure_scope("validation",after,sigma,seed)
        rows=self.meta[self.meta.split=="validation"]
        outputs=[]
        for start in range(0,len(rows),32):
            part=rows.iloc[start:start+32]; items=list(part.itertuples())
            raw=[read_raw(r) for r in items]
            changed=[disturb(x,r.parent_image_id,r.split,sigma,seed) for x,r in zip(raw,items)]
            # Each click executes inference. Case and feature caches are reserved for individual replay.
            clean=self.detector("V0").infer(raw)
            before=self.detector("V0").infer(changed);after_out=self.detector(after).infer(changed)
            for r,c,a,b in zip(items,clean,before,after_out,strict=True):
                self.verify_output(r,"V0",0.,0,c);self.verify_output(r,"V0",sigma,seed,a);self.verify_output(r,after,sigma,seed,b)
                outputs.append(dict(sample_id=r.sample_id,original_bracket=c["prediction"]==r.label,
                    induced=c["prediction"]==r.label and a["prediction"]!=r.label,
                    change=change_type(r.label,a["prediction"],b["prediction"])))
            if progress:progress(min(start+32,len(rows)),len(rows))
        return pd.DataFrame(outputs)

    def case(self,sample,after,sigma,seed,provided=None):
        r=self.meta.loc[sample];split=r.split
        self.ensure_scope(split,after,sigma,seed)
        original=read_raw(r)
        generated=disturb(original,r.parent_image_id,split,sigma,seed)
        raw=generated if provided is None else provided
        if raw.shape!=(200,200) or raw.dtype!=np.dtype("<f4") or array_sha(raw)!=array_sha(generated):
            raise ValueError("재현 배열·해시·난수 설정 불일치")
        outputs={}
        for name in ("V0",after):
            d=self.detector(name)
            if sigma==0:
                clean=noisy=d.infer([raw],cache=self.cache,case_keys=[key_for(self.id,r,sigma,seed)])[0]
            else:
                clean,noisy=d.infer([original,raw],cache=self.cache,case_keys=[key_for(self.id,r,0.,0),key_for(self.id,r,sigma,seed)])
            self.verify_output(r,name,0.,0,clean);self.verify_output(r,name,sigma,seed,noisy)
            outputs[name]=dict(identity=d.identity,model_run_id=d.run_id,original=clean,trial=noisy)
        entry=dict(task_id=TASK,sample_unit=UNIT,run_id=self.id,split=split,sample_id=sample,parent_image_id=r.parent_image_id,
                   original_file_sha256=r.file_sha256,original_input_hash=array_sha(original),raw_input_hash=array_sha(raw),
                   dtype="float32",shape=[200,200],sigma=float(sigma),seed=int(seed),derived_seed=str(case_seed(r.parent_image_id,split,sigma,seed)))
        case_id=stable_id(entry);execution_id=stable_id([case_id,outputs["V0"]["model_run_id"],outputs[after]["model_run_id"]])
        record=dict(**entry,schema=1,n_originals=self.man["n_originals"][split],case_id=case_id,execution_id=execution_id,
                    label=r.label,before="V0",after=after,class_order=CLASSES,outputs=outputs,
                    change=change_type(r.label,outputs["V0"]["trial"]["prediction"],outputs[after]["trial"]["prediction"]))
        return record,original,raw,prepare(raw,self.detector(after).policy)

    def export(self,case):
        r,original,raw,processed=case
        buf=io.BytesIO()
        with ZipFile(buf,"w",ZIP_DEFLATED) as z:
            z.writestr("case.json",json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False).encode())
            z.writestr("raw_input.npy",numpy_bytes(raw))
            z.writestr("original_preview.png",png(original));z.writestr("noise_preview.png",png(raw));z.writestr("processed_preview.png",png(processed))
        return buf.getvalue()

    def replay(self,data):
        with ZipFile(data) as archive:
            if len(set(archive.namelist()))!=len(archive.namelist()) or archive.getinfo("raw_input.npy").file_size>162048 or archive.getinfo("case.json").file_size>65536:
                raise ValueError("잘못된 사례 묶음")
            stored=json.loads(archive.read("case.json")); supplied=np.load(io.BytesIO(archive.read("raw_input.npy")),allow_pickle=False)
        actual,*_=self.case(stored["sample_id"],stored["after"],stored["sigma"],stored["seed"],provided=supplied)
        for key in actual.keys()-{"outputs"}:
            if actual[key]!=stored.get(key):raise ValueError("사례 불일치 · "+key)
        if set(actual["outputs"])!=set(stored["outputs"]):raise ValueError("모델 쌍 불일치")
        for name,x in actual["outputs"].items():
            other=stored["outputs"][name]
            if x["identity"]!=other["identity"] or x["model_run_id"]!=other["model_run_id"]:raise ValueError("모델 불일치")
            for part in ("original","trial"):
                a,b=x[part],other[part]
                if np.shape(b["scores"])!=np.shape(a["scores"]) or any(a[k]!=b[k] for k in a.keys()-{"scores"}) or not np.allclose(a["scores"],b["scores"],atol=CFG["score_atol"],rtol=0):
                    raise ValueError("점수, 입력 또는 예측 불일치")
        return actual


def examples(run):
    split="test" if run.status["evaluated"] else "validation"
    after=run.seal["selected"] or "V1"
    d=run.table(split,"pairs")
    out=RUN/"examples";out.mkdir(exist_ok=True)
    for kind in ("fixed","new_error"):
        g=d[(d.after==after)&(d.change==kind)&(d.sigma>0)]
        if len(g):
            r=g.sort_values(["sigma","seed","sample_id"]).iloc[0]
            path=out/(kind+".zip")
            path.write_bytes(run.export(run.case(r.sample_id,after,r.sigma,int(r.seed))))
            run.replay(path)
        else:(out/(kind+".zip")).unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser();op=parser.add_mutually_exclusive_group(required=True)
    op.add_argument("--replay",type=Path);op.add_argument("--finalize",action="store_true")
    args=parser.parse_args()
    if args.finalize:
        finalize(); examples(SurfaceRun()); return
    actual=SurfaceRun().replay(args.replay)
    print(json.dumps(actual,ensure_ascii=False,allow_nan=False))

if __name__=="__main__":main()
