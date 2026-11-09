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


def matches(path, expected):
    return path.is_file() and sha(path) == expected


def restore_archive(cache, name, url, expected, members):
    """Repair only absent or changed members; a complete checkout stays offline."""
    missing = [(member, dest, digest) for member, dest, digest in members
               if not matches(dest, digest)]
    if missing:
        with ZipFile(fetch(cache, name, url, expected)) as z:
            for member, dest, digest in missing:
                put(z, member, dest, digest)


def main(cache):
    cache.mkdir(parents=True, exist_ok=True)
    ksdd = ROOT / "data/ksdd2"
    audit = json.loads((ksdd / "audit.json").read_text(encoding="utf8"))
    manifest_path = ksdd / "manifest.csv"
    if not matches(manifest_path, audit["manifest_sha256"]):
        raise ValueError("Registered KSDD2 manifest hash mismatch")
    with manifest_path.open(encoding="utf8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != audit["archive_png_pairs"]:
        raise ValueError("KSDD2 manifest pair count differs from the registered audit")
    members = []
    # Hash original bytes in every split, including the excluded duplicate.
    # Official test images/masks and split metadata are never evaluated or unpickled.
    for row in rows:
        for path, digest in (("image_path", "file_sha256"), ("mask_path", "mask_sha256")):
            dest = ROOT / row[path]
            members.append((dest.relative_to(ksdd / "raw").as_posix(), dest, row[digest]))
    metadata = audit["official_split_metadata_sha256"]
    members.extend((name, ksdd / "raw" / name, digest) for name, digest in metadata.items())
    restore_archive(cache, "KolektorSDD2.zip", KSDD2, audit["archive_sha256"], members)
    raw = ROOT / "data/raw"
    manifest = json.loads((raw / "manifest.json").read_text(encoding="utf8"))
    restore_archive(cache, "ssfd.zip", SSFD, manifest["original_zip_sha256"],
                    [(file["path"], raw / file["path"], file["sha256"])
                     for file in manifest["files"]])
    folder = ROOT / "experiments/surface_v1"
    protocol = json.loads((folder / "protocol.json").read_text(encoding="utf8"))
    dest = folder / "models/resnet18-f37072fd.pth"
    if not matches(dest, protocol["weight_sha256"]):
        weights = fetch(cache, dest.name, RESNET, protocol["weight_sha256"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(weights, dest)
    print(f"Verified {len(rows)} image/mask pairs, {len(metadata)} split metadata files, "
          f"{len(manifest['files'])} SSFD files and the encoder.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    main(parser.parse_args().cache)
