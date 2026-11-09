"""Deterministic failure search and verified replay shared by the workbench and CLI."""
import argparse
from pathlib import Path
import itertools
import json
import time

import numpy as np
import pandas as pd

from .cnn_eval import paired
from .pair_runs import CNN, ROCKET, RUNS, pairs_for
from .data import load_dataset
from .faults import inject_fault
from .hi_study import ROOT, make_wave, read_config
from .pair_detector import detector, digest, identifier, file_hash
from .search import grid

CHANGES={"rescued_FN":"고친 미검출","new_FN":"새 미검출","removed_FP":"해소 오경보","new_FP":"새 오경보","unchanged":"판정 유지"}
METHODS={"R0":"MiniRocket · 원본 Ridge","R1":"같은 특징 · 증강 Ridge","J":"강건·세부 주파수 특징 + SVC","A1":"기존 주파수 특징 + RF",
         "C0":"3축 원시 파형 · 7패치 CNN","C1":"같은 CNN · 현재 구간 전체 크기 정규화"}


def explanation(before,after):
    if before=="C0":
        return ("내부 검증에서 진동 크기와 C0 점수의 상관 방향이 그룹에 따라 달라졌다.",
                "진동 크기에 의존하면 새로운 그룹에서 S와 비슬래그를 혼동할 수 있다.",
                "C0에 현재 0.5초 전체의 공통 RMS 정규화 한 단계만 추가해 재학습했다.")
    observed=("고정 교란 네 조건에서 J의 점수가 임계값을 넘은 구간은 0개였다." if before=="J" else
              "원본에서 A1은 실제 S 160개 중 23개를 검출했다.")
    return (observed,"요약 특징이 그룹을 넘어 유지되는 짧은 파형의 B/S 차이를 잃을 수 있다.",
            "요약 특징 분류기를 7패치 원시 파형 CNN으로 교체했다." +
            (" C1은 현재 구간의 공통 진동 크기도 정규화한다." if after=="C1" else ""))


def classify_change(y,a,b):
    if a==b:
        return "unchanged"
    if y:
        return "rescued_FN" if b else "new_FN"
    return "new_FP" if b else "removed_FP"


class ReplaySet:
    def __init__(self,run=CNN,manifest_hash=None):
        self.run=Path(run)
        self.pairs=pairs_for(self.run)
        self.manifest_hash=file_hash(self.run/"manifest.json")
        if manifest_hash is not None and manifest_hash!=self.manifest_hash:
            raise ValueError("평가 실행 ID 불일치")
        self.ready=set()
        self.meta,self.waves,_=load_dataset(ROOT/"data/raw")
        self.lookup={(r.file_id,r.window_idx):r for r in self.meta.itertuples()}
        self.dataset=file_hash(ROOT/"data/raw/manifest.json")
        self.fold_map={g:s["fold_id"] for s in read_config()["folds"] for g in s["test"]}
        self.conditions=tuple(read_config()["conditions"])
        self.manifest=json.loads((self.run/"manifest.json").read_text())
        if self.manifest["dataset_manifest"]!=self.dataset or file_hash(self.run/"predictions.csv")!=self.manifest["predictions_sha256"]:
            raise ValueError("데이터 manifest 또는 평가 CSV 해시 불일치")
        for name,sha in self.manifest["report_hashes"].items():
            if file_hash(self.run/name)!=sha:
                raise ValueError("전체 회귀 결과 해시 불일치: "+name)
        self.pred=pd.read_csv(self.run/"predictions.csv",float_precision="round_trip")
        self.identities={(r["model"],r["fold"]):r for r in self.manifest["models"]}
        self.regression=pd.read_csv(self.run/"changes.csv")
        self.evidence=json.loads((self.run/"evidence.json").read_text()) if self.run.name==ROCKET.name else None
        self.atol=json.loads((self.run/"protocol.json").read_text())["score_atol"]

    def model(self,name,fold):
        return detector(name,fold,self.run,self.manifest_hash)

    def warm(self,before,after):
        start=time.monotonic()
        for name in (before,after):
            for f in sorted(set(self.fold_map.values())):
                if (name,f) not in self.ready:
                    d=self.model(name,f)
                    if self.identities[name,str(f)]!=d.identity:
                        raise ValueError("저장 모델 ID 불일치")
                    row=next(r for r in self.lookup.values() if self.fold_map[r.group_id]==f)
                    d.infer([self.waves[row.Index]])
                    self.ready.add((name,f))
        return time.monotonic()-start

    def assess(self,before,after,file,idx,fault):
        if (before,after) not in self.pairs:
            raise ValueError("저장되지 않은 모델 쌍")
        row=self.lookup[file,idx]
        fold=self.fold_map[row.group_id]
        models=[self.model(name,fold) for name in (before,after)]
        for d in models:
            if self.identities[d.name,str(fold)]!=d.identity or row.group_id not in d.test or row.group_id in d.train:
                raise ValueError("저장 모델 ID, 임계값 또는 평가 그룹 불일치")
        raw=self.waves[row.Index]
        scale=self.model("J",fold).legacy["scale"]
        if fault["kind"]=="fixed" and fault["scenario"] in self.conditions:
            changed=make_wave(raw,scale,row,fault["scenario"])
        elif fault["kind"]=="grid":
            changed,positions,signs=inject_fault(raw,fault["type"],fault["axis"],fault["scale"],fault["count"],
                                          fault["seed"],scale,positions=fault.get("positions"),signs=fault.get("signs"))
            fault={**fault,"positions":positions,"signs":signs}
        else:
            raise ValueError("알 수 없는 교란 설정")
        data=dict(dataset_sha256=self.dataset,file_id=file,window_idx=int(idx),group_id=int(row.group_id),
                  fold_id=int(fold),phase=row.phase,y_true=int(row.y_true),source_sha256=digest(raw),
                  input_sha256=digest(changed),fault=fault)
        case_id=identifier(data)
        output={}
        for d in models:
            if data["source_sha256"]==data["input_sha256"]:
                clean=current=d.infer([raw])[0]
            else:
                clean,current=d.infer([raw,changed])
            output[d.name]=dict(**d.identity,clean=clean,test=current)
        a,b=[output[d.name]["test"]["prediction"] for d in models]
        if self.evidence is None:
            evidence,hypothesis,mod=explanation(before,after)
        elif before=="J":
            v=self.evidence[str(fold)]
            evidence=f"기존 J의 fold {fold} 원본 학습 AUC {v['j_train_auc']:.3f}, 학습 제외 그룹 AUC {v['j_outer_auc']:.3f}."
            hypothesis="다변량 파형의 국소 패턴을 변환하면 새로운 그룹의 S와 비슬래그를 더 잘 구분할 수 있다."
            mod="MiniRocket 특징과 Ridge 분류기를 원본 학습 그룹으로 적합했다."
        else:
            v=self.evidence[str(fold)]
            evidence=(f"이 fold의 학습 그룹 내부 교란 {v['inner_trials']}회에서 R0 정답이 오답으로 바뀐 경우 "
                      f"{v['induced']}회, 오답이 정답으로 바뀐 경우 {v['resolved']}회.")
            hypothesis="같은 특징에서 교란 입력을 Ridge 학습에 포함하면 유발 오판을 줄일 수 있다."
            mod="변환기와 scaler를 동결하고 원본·잡음·단축 충격을 0.5/0.25/0.25로 가중해 Ridge만 재학습했다."
        change=classify_change(data["y_true"],a,b)
        record=dict(schema=1,case_id=case_id,execution_id=identifier([case_id,*[d.run_id for d in models]]),
                    **data,before=before,after=after,outputs=output,change=change,
                    before_induced=(output[before]["clean"]["prediction"]==row.y_true and a!=row.y_true),
                    evidence=evidence,human_hypothesis=hypothesis,modification=mod,
                    regression={"table":f"experiments/{self.run.name}/changes.csv","before":before,"after":after,
                                "summary":f"experiments/{self.run.name}/summary.csv"})
        if self.evidence is not None:
            record.update(schema=2,study=self.run.name,manifest_sha256=self.manifest_hash,
                          execution_id=identifier([self.manifest_hash,record["execution_id"]]),
                          evaluation_origin="outer_held_out",inner_evidence=self.evidence[str(fold)]["inner_example"] if before=="R0" else None)
        return raw,changed,record

    def replay(self,stored):
        raw,changed,new=self.assess(stored["before"],stored["after"],stored["file_id"],stored["window_idx"],stored["fault"])
        for k in ("schema","case_id","execution_id","fault","dataset_sha256","source_sha256","input_sha256","group_id",
                  "phase","y_true","fold_id","before_induced","change"):
            if new[k]!=stored[k]:
                raise ValueError("사례 불일치: "+k)
        if new["schema"]==2:
            for k in ("study","manifest_sha256","evaluation_origin","inner_evidence","regression","evidence","modification","human_hypothesis"):
                if stored.get(k)!=new[k]:
                    raise ValueError("평가 실행 또는 근거 사례 불일치: "+k)
        if set(new["outputs"])!=set(stored["outputs"]):
            raise ValueError("모델 쌍 불일치")
        for name,actual in new["outputs"].items():
            saved=stored["outputs"][name]
            for k in actual.keys()-{"clean","test"}:
                if actual[k]!=saved[k]:
                    raise ValueError("모델 또는 임계값 불일치: "+name)
            for part in ("clean","test"):
                a,b=actual[part],saved[part]
                if a["prediction"]!=b["prediction"] or a["input_status"]!=b["input_status"] or not np.isclose(
                        a["score"],b["score"],rtol=0,atol=self.atol):
                    raise ValueError("재추론 결과 불일치: "+name)
        return raw,changed,new

    def fixed_case(self,before,after,row):
        raw,changed,record=self.assess(before,after,row.file_id,int(row.window_idx),dict(kind="fixed",scenario=row.scenario))
        if record["input_sha256"]!=row.input_sha256_before or record["change"]!=row.change:
            raise ValueError("전체 회귀 CSV와 사례 불일치")
        for side,name in (("before",before),("after",after)):
            out=record["outputs"][name]
            if (out["run_id"]!=row["model_run_id_"+side] or out["test"]["prediction"]!=row["prediction_"+side] or
                not np.isclose(out["test"]["score"],row["score_"+side],rtol=0,atol=self.atol)):
                raise ValueError("저장된 모델 실행과 재생 결과가 다릅니다.")
        return raw,changed,record

    def search(self,before,after,max_trials=160,max_seconds=90):
        warm_s=self.warm(before,after)
        start=time.monotonic()
        clean=self.pred[(self.pred.model_id==before)&(self.pred.scenario=="clean")]
        correct=clean[clean.prediction==clean.y_true]
        sample=[g.sort_values(["group_id","window_idx","file_id"]).sample(n=min(20,len(g)),random_state=42)
                for _,g in correct.groupby("y_true")]
        picked=[r for pair in itertools.zip_longest(*(s.itertuples() for s in sample)) for r in pair if r]
        records=[]
        for args,r in itertools.product(grid(("noise","spike")),picked):
            if len(records)>=max_trials or time.monotonic()-start>=max_seconds:
                break
            typ,axis,scale,count,seed=args
            fault=dict(kind="grid",type=typ,axis=axis,scale=scale,count=count,seed=seed)
            _,_,case=self.assess(before,after,r.file_id,r.window_idx,fault)
            records.append(case)
        return dict(study=self.run.name,manifest_sha256=self.manifest_hash,before=before,after=after,n_clean=len(clean),excluded_original_errors=len(clean)-len(correct),
                    eligible=len(correct),sample_size=len(picked),trials=len(records),
                    warmup_seconds=round(warm_s,3),elapsed_seconds=round(time.monotonic()-start,3),records=records)


def save_cases(run=CNN):
    run=Path(run)
    ctx=ReplaySet(run,file_hash(run/"manifest.json"))
    for before,after in ctx.pairs:
        result=ctx.search(before,after)
        (run/f"search_{before}_{after}.json").write_text(json.dumps(result,ensure_ascii=False,allow_nan=False),encoding="utf8")
        failures=[r for r in result["records"] if r["before_induced"]]
        if (before,after)==("J","C1") and run==CNN:
            examples=run/"examples"
            examples.mkdir(exist_ok=True)
            for label,kind in [("recovered","rescued_FN"),("still_missed","unchanged")]:
                dest=examples/(label+".json")
                first=next((r for r in failures if r["change"]==kind and r["y_true"]==1),None)
                if first is not None:
                    dest.write_text(json.dumps(first,ensure_ascii=False,indent=2),encoding="utf8")
                else:
                    dest.unlink(missing_ok=True)
        if run.name==ROCKET.name:
            examples=run/"examples"
            examples.mkdir(exist_ok=True)
            clean=paired(ctx.pred[ctx.pred.scenario=="clean"],before,after)
            for kind in ("rescued_FN","new_FN","removed_FP","new_FP"):
                rows=clean[clean.change==kind]
                target=examples/f"{before}_{after}_{kind}.json"
                if len(rows):
                    _,_,case=ctx.fixed_case(before,after,rows.iloc[0])
                    target.write_text(json.dumps(case,ensure_ascii=False,indent=2),encoding="utf8")
                else:
                    target.unlink(missing_ok=True)
            target=examples/f"{before}_{after}_search_failure.json"
            if failures:
                target.write_text(json.dumps(failures[0],ensure_ascii=False,indent=2),encoding="utf8")
            else:
                target.unlink(missing_ok=True)
        for case in failures[:2]:
            ctx.replay(case)
        print(before,after,result["trials"],"induced",len(failures),"fixed",sum(r["change"] in ("rescued_FN","removed_FP") for r in failures))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",choices=RUNS,default="cnn_v1")
    save_cases(RUNS[parser.parse_args().run])
