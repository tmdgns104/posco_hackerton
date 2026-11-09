"""Trusted inspector registry. Sampling and labels live outside the prediction adapters."""
import hashlib
import io
import re
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .inspection_data import MANIFEST_SHA as BINARY_INDEX_SHA
from .inspection_inputs import BINARY_CONDITIONS, change_image
from .surface_data import CLASSES, ROOT
from .surface_detector import disturb
from .surface_g1 import DEFAULT_G1, file_sha, read_json

RUNS = ROOT / "inspection_runs"
CASES = ROOT / "inspection_cases"
DATASETS = ROOT / "inspection_projects"
BINARY = ROOT / "experiments" / "inspection_binary"
BINARY_INDEX = ROOT / "data/ksdd2/manifest.csv"
BUILTIN = {"ksdd2":"KolektorSDD2 · 정상 / 불량 검사", "neu_cls6":"NEU-CLS · 강판 결함 6종"}
USER_ID = re.compile(r"user_[a-zA-Z0-9_-]{6,80}\Z")
SOURCES = {"ksdd2":"Kolektor Group / ViCoS Lab · KolektorSDD2 · CC BY-NC-SA 4.0",
           "neu_cls6":"NEU-CLS · Northeastern University · 공개 열연강판 결함 분류 데이터"}
META = ["sample_id","image_path","true_label","group_id","split","source_id","file_sha256"]


def register_zip(data, title, template_id):
    from .inspection_register import register_zip as register
    return register(data,title,template_id)


def register_folder(folder,csv_path,title,template_id):
    from .inspection_register import register_folder as register
    return register(folder,csv_path,title,template_id)


def list_projects():
    entries = dict(BUILTIN)
    if DATASETS.is_dir():
        for folder in sorted(DATASETS.iterdir()):
            if USER_ID.fullmatch(folder.name) and not folder.is_symlink() and (folder/"project.json").is_file():
                config = read_json(folder/"project.json")
                if config["id"] == folder.name:
                    entries[folder.name] = config["title"]
    return entries


class SurfaceAdapter:
    def __init__(self, detector):
        self.detector = detector
        self.identity = detector.identity

    def predict(self, raws):
        return [{"predicted_label":row["prediction"], "scores":dict(zip(CLASSES,row["scores"],strict=True)),
                 "score_type":"decision_function","status":"ok"} for row in self.detector.infer(raws)]


class InspectionProject:
    def __init__(self, project_id):
        self.id = project_id
        self.official = project_id in BUILTIN
        if self.official:
            self.template_id = project_id
            self.title = BUILTIN[project_id]
            self.config = {}
        else:
            if not USER_ID.fullmatch(project_id):
                raise ValueError("등록되지 않은 검사 프로젝트입니다.")
            folder = DATASETS / project_id
            if folder.is_symlink():
                raise ValueError("프로젝트 경로가 올바르지 않습니다.")
            self.config = read_json(folder/"project.json")
            if self.config["id"] != project_id or self.config["template_id"] not in BUILTIN:
                raise ValueError("프로젝트 등록 정보가 일치하지 않습니다.")
            self.template_id = self.config["template_id"]
            self.title = self.config["title"]
        self.kind = "binary" if self.template_id=="ksdd2" else "multiclass"
        self.classes = ["normal","defect"] if self.kind=="binary" else list(CLASSES)
        self.source = SOURCES[self.template_id] if self.official else "사용자 등록 정답 CSV · 개발용 검증"
        path = BINARY/"protocol.json" if self.kind=="binary" else DEFAULT_G1/"protocol.json"
        self.protocol_file_sha256 = file_sha(path)
        self.protocol = read_json(path)
        if self.kind == "binary":
            if not self.protocol["split_manifest_sha256"] == file_sha(BINARY_INDEX) == BINARY_INDEX_SHA:
                raise ValueError("KolektorSDD2 원본 분할과 학습 때 사용한 인덱스가 다릅니다.")
            if self.protocol["implementation"]["input_source_sha256"] != file_sha(Path(__file__).with_name("inspection_inputs.py")):
                raise ValueError("원시 입력 교란 코드가 학습 규약과 다릅니다.")
            if self.protocol["conditions"] != [{"condition":c,"seed":seed} for c,seed in BINARY_CONDITIONS]:
                raise ValueError("계획된 시험 조건이 학습 규약과 다릅니다.")
        self.frozen_status = read_json(DEFAULT_G1/"test_status.json") if self.kind=="multiclass" and self.official else None
        if self.official and self.kind=="multiclass":
            index = ROOT/"experiments/surface_v1/split_manifest.csv"
            if file_sha(index)!=self.protocol["split_sha256"]:
                raise ValueError("NEU 원본 분할 해시가 다릅니다.")
            df = pd.read_csv(index,dtype=str,keep_default_na=False)
            self.index = df.assign(image_path="data/neu_cls/"+df.file_name, true_label=df.label,
                                   group_id=df.parent_image_id,source_id="NEU-CLS")[META]
        else:
            index = BINARY_INDEX if self.official else DATASETS/project_id/"manifest.csv"
            if not self.official and file_sha(index)!=self.config["manifest_sha256"]:
                raise ValueError("등록한 데이터 목록이 바뀌었습니다.")
            self.index = pd.read_csv(index,dtype=str,keep_default_na=False)
        self.index_hash = file_sha(index)
        if not set(META).issubset(self.index) or self.index.sample_id.duplicated().any():
            raise ValueError("검사 목록의 필드 또는 고유 ID를 확인해 주세요.")

    @cached_property
    def _by_sample(self):
        return self.index.set_index("sample_id",drop=False,verify_integrity=True)

    @cached_property
    def models(self):
        if self.kind == "binary":
            from .inspection_model import load_binary
            return {name:load_binary(name) for name in ("K0","K1")}
        from .surface_g1_replay import G1Run
        frozen = G1Run(DEFAULT_G1)
        return {name:SurfaceAdapter(frozen.detector(name)) for name in ("V0","G1")}

    def available_splits(self):
        if self.official and self.kind=="binary" and (BINARY/"test_access.json").is_file():
            state = read_json(BINARY/"test_access.json")
            if state.get("allowed") is True:
                self.check_scope("test")
                return ["validation","test"]
        return ["validation"]

    def check_scope(self, split):
        if split=="validation":
            return
        if split!="test" or not self.official or self.kind!="binary":
            raise ValueError("이 실행기는 현재 프로젝트의 검증 분할만 허용합니다.")
        verify_test_access(self)

    def rows(self, split):
        self.check_scope(split)
        return self.index.loc[self.index.split==split, META].copy().reset_index(drop=True)

    def is_full_split(self,split):
        rows = self.rows(split)
        return len(rows)>0 and all((ROOT/p).is_file() for p in rows.image_path)

    def demo_ids(self,split):
        rows=self.rows(split)
        available=rows[rows.image_path.map(lambda name:(ROOT/name).is_file())].sort_values("sample_id")
        return available.groupby("true_label",sort=True).head(max(1,24//len(self.classes))).sample_id.tolist()

    def conditions(self,split):
        self.check_scope(split)
        if self.kind=="binary":
            return list(BINARY_CONDITIONS)
        return [("clean" if not sigma else f"noise_{sigma:.2f}",seed) for sigma,seed in self.protocol["conditions"][split]]

    def load(self,row):
        if row.split not in ("validation","test"):
            raise ValueError("허용되지 않은 추론 분할입니다.")
        self.check_scope(row.split)
        canonical = self._by_sample.loc[row.sample_id]
        if any(getattr(row,k)!=canonical[k] for k in META):
            raise ValueError("이 이미지의 분할·경로·정답이 원래 등록 기록과 다릅니다.")
        path=(ROOT/row.image_path).resolve()
        if self.official and self.kind=="binary":
            allowed_root=ROOT/"data/ksdd2/raw"/("train" if row.split=="validation" else "test")
        elif self.official:
            allowed_root=ROOT/"data/neu_cls"
        else:
            allowed_root=DATASETS/self.id
        if not path.is_relative_to(allowed_root.resolve()):
            raise ValueError("프로젝트 밖의 이미지 경로입니다.")
        payload=path.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=row.file_sha256:
            raise ValueError("원본 이미지 SHA-256이 일치하지 않습니다.")
        with Image.open(io.BytesIO(payload)) as im:
            if self.kind=="multiclass" and (im.size!=(200,200) or im.mode!="L"):
                raise ValueError("NEU 모델은 200×200 회색조 이미지를 사용합니다.")
            arr=np.asarray(im.convert("RGB") if self.kind=="binary" else im).astype(np.float32)
        return arr / np.float32(255)

    def change(self,raw,row,condition,seed):
        if self.kind=="binary":
            return change_image(raw,row.sample_id,condition,seed)
        actual = dict((("clean" if not scale else f"noise_{scale:.2f}",s),scale) for scale,s in self.protocol["conditions"][row.split])
        scale = actual[condition,seed]
        return disturb(raw,row.group_id,row.split,scale,seed)


def load_project(project_id):
    return InspectionProject(project_id)


def validation_seal(project, run_dir):
    from .inspection_runner import recompute_decision
    run_dir=Path(run_dir).resolve()
    if not run_dir.is_relative_to(RUNS.resolve()) or project.id!="ksdd2":
        raise ValueError("공식 정상·불량 검증 실행이 필요합니다.")
    manifest=read_json(run_dir/"manifest.json")
    if (manifest["project_id"]!=project.id or manifest["split"]!="validation"
            or manifest["scope"]!="full" or manifest["failed_pairs"]!=0
            or set(manifest["models"])!=set(project.models)
            or manifest["models"]!={name:m.identity for name,m in project.models.items()}):
        raise ValueError("전체 검증 분할과 동결된 두 모델의 기록이 필요합니다.")
    expected_roles=[project.protocol["baseline_model"], project.protocol["candidate_model"]]
    if (manifest.get("official") is not True or manifest["kind"]!=project.kind or manifest["classes"]!=project.classes
            or [manifest["before_model_id"],manifest["after_model_id"]]!=expected_roles
            or manifest["conditions"]!=[list(c) for c in project.conditions("validation")]
            or manifest["source_manifest_sha256"]!=project.index_hash
            or manifest["protocol_file_sha256"]!=project.protocol_file_sha256
            or manifest["protocol"]!=project.protocol):
        raise ValueError("실행 시점의 규약과 다릅니다.")
    recorded=pd.read_csv(run_dir/"data_manifest.csv",dtype=str,keep_default_na=False)
    expected=project.rows("validation")
    for frame in (recorded,expected):
        if frame.sample_id.duplicated().any():
            raise ValueError("중복된 검증 이미지 ID입니다.")
    if not recorded[META].sort_values("sample_id").reset_index(drop=True).equals(
            expected[META].sort_values("sample_id").reset_index(drop=True)):
        raise ValueError("검증 실행의 원본·정답 목록과 공식 검증 분할이 다릅니다.")
    verdict=recompute_decision(run_dir)
    selection=read_json(BINARY/"selection.json")
    return dict(allowed=verdict["status"]=="pass" and selection.get("eligible",False) is True,
                run_id=manifest["run_id"], relative_run=run_dir.relative_to(RUNS.resolve()).as_posix(),
                manifest_sha256=file_sha(run_dir/"manifest.json"), decision_sha256=file_sha(run_dir/"decision.json"),
                training_selection_sha256=file_sha(BINARY/"selection.json"), index_sha256=project.index_hash,
                protocol_sha256=file_sha(BINARY/"protocol.json"),models=manifest["models"])


def verify_test_access(project):
    seal_path = BINARY/"test_access.json"
    if not seal_path.is_file():
        raise ValueError("공식 시험은 전체 검증에서 사전 기준을 통과한 뒤에만 열립니다.")
    state=read_json(seal_path)
    if not state.get("allowed"):
        raise ValueError("검증 기준 미달로 공식 시험 모델 실행을 잠갔습니다.")
    manifest=RUNS/state["relative_run"]/"manifest.json"
    token=(file_sha(seal_path),manifest.stat().st_mtime_ns)
    if getattr(project,"_approved_test",None) != token:
        if validation_seal(project,manifest.parent)!=state:
            raise ValueError("검증 시점 이후 모델·규약·데이터 또는 실행 기록이 변경됐습니다.")
        project._approved_test=token
