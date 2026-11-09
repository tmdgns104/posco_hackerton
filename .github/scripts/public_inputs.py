"""Materialize the published inputs used in the default checkout, without model training."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2] / "SlagWatch"
KSDD2 = "https://github.com/tmdgns104/posco_hackerton/releases/download/ksdd2-original-v1/KolektorSDD2.zip"
SSFD = "https://data.mendeley.com/public-api/zip/mjnpsbs6fs/download/3"
RESNET = "https://download.pytorch.org/models/resnet18-f37072fd.pth"


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def fetch(cache, name, url, expected):
    path = cache / name
    if not path.is_file() or sha(path) != expected:
        tmp = path.with_suffix(".part")
        try:
            with urlopen(url, timeout=120) as src, tmp.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            if sha(tmp) != expected:
                raise ValueError(f"Download hash mismatch for {name}")
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)
    return path


def put(z, name, dest, expected):
    data = z.read(name)
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Member hash mismatch for {name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)


def main(cache):
    cache.mkdir(parents=True, exist_ok=True)
    ksdd = ROOT / "data/ksdd2"
    audit = json.loads((ksdd / "audit.json").read_text(encoding="utf8"))
    archive = fetch(cache, "KolektorSDD2.zip", KSDD2, audit["archive_sha256"])
    with ZipFile(archive) as z, (ksdd / "manifest.csv").open(encoding="utf8", newline="") as f:
        validation = [r for r in csv.DictReader(f) if r["split"] == "validation"]
        if len(validation) != 466:
            raise ValueError("Expected 466 validation rows.")
        for row in validation:
            for path, digest in (("image_path", "file_sha256"), ("mask_path", "mask_sha256")):
                dest = ROOT / row[path]
                put(z, dest.relative_to(ksdd / "raw").as_posix(), dest, row[digest])
    raw = ROOT / "data/raw"
    manifest = json.loads((raw / "manifest.json").read_text(encoding="utf8"))
    with ZipFile(fetch(cache, "ssfd.zip", SSFD, manifest["original_zip_sha256"])) as z:
        for file in manifest["files"]:
            put(z, file["path"], raw / file["path"], file["sha256"])
    folder = ROOT / "experiments/surface_v1"
    protocol = json.loads((folder / "protocol.json").read_text(encoding="utf8"))
    weights = fetch(cache, "resnet18-f37072fd.pth", RESNET, protocol["weight_sha256"])
    (folder / "models").mkdir(exist_ok=True)
    shutil.copyfile(weights, folder / "models" / weights.name)
    print(f"Verified {len(validation)} image/mask pairs, {len(manifest['files'])} SSFD files and the encoder.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    main(parser.parse_args().cache)
