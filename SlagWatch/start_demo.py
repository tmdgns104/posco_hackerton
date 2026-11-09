"""Check the local release before opening Streamlit. Use only stdlib until imports pass."""
import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import logging
import os
import platform
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CPU_PACKAGES = ("torch==2.8.0", "torchvision==0.23.0")
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
CORE_MODULES = ("torch","torchvision","numpy","pandas","sklearn","cv2","streamlit","joblib","filelock",
                "PIL","scipy","plotly","aeon","numba")


def check_python(version, bits):
    if tuple(version[:2]) != (3,11) or bits != 64:
        raise RuntimeError(f"Python 3.11 64-bit required; found {version[0]}.{version[1]} {bits}-bit.")


def check_files(root):
    for name in ("app_demo.py","app.py","requirements.txt","requirements.lock.txt"):
        if not (root/name).is_file():
            raise FileNotFoundError(root/name)
    for name in ("surface_v1","surface_g1"):
        folder = root/"experiments"/name
        man = json.loads((folder/"manifest.json").read_text(encoding="utf8"))
        for path in [*(folder/n for n in man["files"]), *(root/"src"/n for n in man["code_sha256"])]:
            if not path.is_file():
                raise FileNotFoundError(path)
    binary = root/"experiments/inspection_binary"
    for name in ("protocol.json", "selection.json", "test_access.json", "models/K0.joblib", "models/K1.joblib"):
        if not (binary/name).is_file():
            raise FileNotFoundError(binary/name)
    with (root/"data/ksdd2/manifest.csv").open(encoding="utf8", newline="") as fp:
        for row in csv.DictReader(fp):
            if row["split"] == "validation":
                for column in ("image_path","mask_path"):
                    if not (root/row[column]).is_file():
                        raise FileNotFoundError(root/row[column])
    sources = root/"experiments/surface_v1"
    extras = [sources/"models/resnet18-f37072fd.pth",*(root/f"experiments/surface_g1/examples/{k}.zip" for k in ("fixed","new_error"))]
    with (sources/"split_manifest.csv").open(encoding="utf8",newline="") as fp:
        extras.extend(root/"data/neu_cls"/r["file_name"] for r in csv.DictReader(fp))
    for path in extras:
        if not path.is_file():
            raise FileNotFoundError(path)


def install_identity(root):
    return {"requirements":{n:hashlib.sha256((root/n).read_bytes()).hexdigest()
                            for n in ("requirements.txt","requirements.lock.txt")},
            "cpu_index":CPU_INDEX,"cpu_packages":list(CPU_PACKAGES), "install":"CPU wheels then requirements.lock.txt",
            "python":platform.python_version(),"architecture":platform.machine(),"bits":struct.calcsize("P")*8}


def pip(*args):
    proc = subprocess.run([sys.executable,"-m","pip","--disable-pip-version-check",*args],
                          stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors="replace")
    logging.info("%s",proc.stdout.rstrip())
    if proc.returncode:
        raise RuntimeError("pip " + args[0] + " failed (see startup.log)")


def check_dependencies(root):
    for ln in (root/"requirements.lock.txt").read_text(encoding="utf8").splitlines():
        line = ln.partition("#")[0].strip()
        if not line:
            continue
        name, wanted = line.split("==")
        got = importlib.metadata.version(name)
        if got.split("+")[0] != wanted:
            raise RuntimeError(f"{name}: expected {wanted}, found {got}")
    for name in CORE_MODULES:
        importlib.import_module(name)
    pip("check")


def write_stamp(path, identity):
    with tempfile.NamedTemporaryFile(mode="w",encoding="utf8",dir=path.parent,delete=False) as tmp:
        json.dump(identity,tmp,ensure_ascii=False,indent=2)
    os.replace(tmp.name,path)


def prepare(root, bootstrap):
    check_python(sys.version_info,struct.calcsize("P")*8)
    logging.info("Checking release files")
    check_files(root)
    identity = install_identity(root)
    in_venv = sys.prefix != sys.base_prefix
    stamp = Path(sys.prefix)/"steeltrust-demo-install.json"
    known = False
    if in_venv and stamp.is_file():
        try:
            known = json.loads(stamp.read_text(encoding="utf8")) == identity
        except (UnicodeError, json.JSONDecodeError):
            logging.info("Unreadable install record; verifying the installed environment")
    logging.info("Checking installed packages and imports (installation record %s)", "matches" if known else "new/changed")
    try:
        check_dependencies(root)
    except (ImportError,OSError,RuntimeError) as exc:
        if not bootstrap or not in_venv:
            raise
        logging.warning("Dependency check failed: %s. Installing release dependencies into this venv.",exc)
        pip("install","--index-url",CPU_INDEX,*CPU_PACKAGES)
        pip("install","-r",str(root/"requirements.lock.txt"))
        # A clean interpreter is needed when an imported dependency was repaired.
        subprocess.run([sys.executable,str(root/"start_demo.py"),"--check"],cwd=root,check=True)
        return
    if in_venv and not known:
        write_stamp(stamp,identity)
    logging.info("Local dependency checks passed; no installation needed")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bootstrap",action="store_true",help="Install pinned packages only if a venv fails its checks")
    parser.add_argument("--check",action="store_true",help="Check files and environment, without opening the browser")
    args=parser.parse_args()
    logs=ROOT/"logs"
    logs.mkdir(exist_ok=True)
    log=logs/"startup.log"
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),logging.FileHandler(log,encoding="utf8")])
    stage="environment / release preflight"
    try:
        prepare(ROOT,args.bootstrap)
        if not args.check:
            stage="Streamlit"
            logging.info("Opening SteelTrust inspection regression · app_demo.py")
            subprocess.run([sys.executable,"-m","streamlit","run",str(ROOT/"app_demo.py")],cwd=ROOT,check=True)
    except (OSError,ValueError,ImportError,KeyError,RuntimeError,subprocess.CalledProcessError) as exc:
        logging.exception("FAILED at %s: %s. Log: %s",stage,exc,log)
        return 1
    return 0


if __name__=="__main__":
    sys.exit(main())
