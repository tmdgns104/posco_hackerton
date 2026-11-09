"""Run an image-inspection comparison or seal an official model-selection result."""
import argparse
import json
from pathlib import Path

from filelock import FileLock

from .inspection_projects import BINARY, RUNS, list_projects, load_project, register_folder, register_zip, validation_seal
from .surface_g1 import put_json, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cmd = parser.add_subparsers(dest="command", required=True)
    cmd.add_parser("list")
    run = cmd.add_parser("run")
    run.add_argument("--project",required=True)
    run.add_argument("--split",choices=["validation","test"],default="validation")
    run.add_argument("--demo",action="store_true",help="Use the deterministic, labelled demonstration subset")
    gate = cmd.add_parser("freeze-validation")
    gate.add_argument("--run",required=True,type=Path)
    register = cmd.add_parser("register")
    register.add_argument("--template",choices=["neu_cls6","ksdd2"],required=True)
    register.add_argument("--title",required=True)
    source = register.add_mutually_exclusive_group(required=True)
    source.add_argument("--zip",type=Path)
    source.add_argument("--folder",type=Path)
    register.add_argument("--labels",type=Path)
    args = parser.parse_args()
    if args.command=="list":
        print(json.dumps(list_projects(),ensure_ascii=False,indent=2))
    elif args.command=="register":
        if args.zip:
            from .inspection_register import MAX_ARCHIVE_BYTES
            with args.zip.open("rb") as stream:
                print(register_zip(stream.read(MAX_ARCHIVE_BYTES+1),args.title,args.template))
        else:
            print(register_folder(args.folder,args.labels or args.folder/"manifest.csv",args.title,args.template))
    elif args.command=="freeze-validation":
        with FileLock(str(BINARY/".freeze.lock"),timeout=0):
            state=validation_seal(load_project("ksdd2"),args.run)
            dest=BINARY/"test_access.json"
            if dest.exists() and read_json(dest)!=state:
                raise ValueError("이 실험의 공식 검증 판정은 이미 확정됐습니다.")
            if not dest.exists():
                put_json(dest,state)
            print(json.dumps(state,ensure_ascii=False))
    else:
        from .inspection_runner import run_project
        project=load_project(args.project)
        if not args.demo and not project.is_full_split(args.split):
            parser.error("전체 원본 복구가 필요합니다. 동봉 부분집합은 --demo로 실행하세요.")
        subset=project.demo_ids(args.split) if args.demo else None
        path=run_project(project,args.split,RUNS,sample_ids=subset)
        print(path)
        print((path/"decision.json").read_text(encoding="utf8"))

if __name__=="__main__":
    main()
