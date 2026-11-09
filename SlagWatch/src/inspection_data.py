"""Acquire and audit the pinned ViCoS KSDD2 release; no model or inference code."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import pickle
import pickletools
import re
import shutil
import tempfile
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("data/ksdd2")
SOURCE = "KolektorSDD2-ViCoS-2023-09-28"
URL = "https://data.vicos.si/datasets/KSDD/KolektorSDD2.zip"
ARCHIVE_SHA = "edcdb486809b24f1d17b785e30c52fafc5999554dd5fe18ddf77b61ceb6f36a8"
ARCHIVE_SIZE = 853126555
MANIFEST_SHA = "8c85620d838a4916afc94c365593bf351e09588d65b733f8bbc96e67d44faa68"
PNG = re.compile(r"(?:train|test)/[0-9]{5}(?:_GT)?(?: \(copy\))?\.png")
FLDS = ("sample_id image_path true_label group_id split source_id file_sha256 mask_path "
        "mask_sha256 pixel_sha256 mask_pixel_sha256 official_split part_id width height "
        "defect_pixels defect_bbox_xyxy exclusion_reason").split()
def by(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_archive(path):
    path = Path(path)
    if path.stat().st_size != ARCHIVE_SIZE or file_sha(path) != ARCHIVE_SHA:
        raise ValueError("Official KSDD2 archive size/SHA-256 mismatch; refusing to ingest")


def fetch(path):
    path = Path(path)
    if path.exists():
        verify_archive(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=path.parent) as tmp:
        stage = Path(tmp.name)
    try:
        with urllib.request.urlopen(URL, timeout=120) as src, stage.open("wb") as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)
        verify_archive(stage)
        os.replace(stage, path)
    finally:
        stage.unlink(missing_ok=True)


def unpack(archive, root):
    """Only regular files in this pinned release; never execute the official pickle."""
    verify_archive(archive)
    dest = Path(root) / DATA / "raw"
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            n = info.filename
            if n in ("train/", "test/"):
                continue
            if not (PNG.fullmatch(n) or re.fullmatch(r"split_weakly_(?:0|16|53|126|246)\.pyb", n)):
                raise ValueError(f"Unexpected archive entry: {n}")
            p = dest / n
            payload = z.read(info)  # ZIP validates CRC
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.exists() and file_sha(p) == by(payload):
                continue
            p.write_bytes(payload)


def pixel_hash(rgb):
    """SHA-256 over dimensions/mode and native row-major decoded RGB bytes."""
    h = hashlib.sha256(f"{rgb.width}x{rgb.height}:RGB:".encode("ascii"))
    h.update(rgb.tobytes())
    return h.hexdigest()


def image_group(rgb):
    native = pixel_hash(rgb)
    return native, "d4-" + min([native] + [pixel_hash(rgb.transpose(op)) for op in Image.Transpose])


def load_pair(image_path, mask_path):
    """A missing, damaged, mis-sized or nonbinary mask is a hard error, never 'normal'."""
    im_bytes, mask_bytes = Path(image_path).read_bytes(), Path(mask_path).read_bytes()
    with Image.open(io.BytesIO(im_bytes)) as im, Image.open(io.BytesIO(mask_bytes)) as mask:
        if im.format != "PNG" or im.mode != "RGB" or mask.format != "PNG" or mask.mode != "L":
            raise ValueError(f"Unexpected image/mask format: {image_path}")
        if im.size != mask.size:
            raise ValueError(f"Image/mask size mismatch: {image_path}")
        im.load()
        mask.load()
        hist = mask.histogram()
        if sum(hist[1:255]):
            raise ValueError(f"Nonbinary mask: {mask_path}")
        native, group = image_group(im)
        return dict(true_label="defect" if hist[255] else "normal", group_id=group,
                    file_sha256=by(im_bytes), mask_sha256=by(mask_bytes), pixel_sha256=native,
                    mask_pixel_sha256=pixel_hash(mask.convert("RGB")), width=im.width,
                    height=im.height, defect_pixels=hist[255],
                    defect_bbox_xyxy=",".join(map(str, mask.getbbox() or ())))


def official_ids(path):
    # This file is immutable, from the pinned release. Permit only literal primitive opcodes
    # in addition to disabling even the possibility of loading globals/reducers.
    buf = path.read_bytes()
    allowed = set("PROTO FRAME EMPTY_LIST MEMOIZE MARK BININT2 NEWTRUE TUPLE2 APPENDS STOP".split())
    if not all(op.name in allowed for op, arg, pos in pickletools.genops(buf)):
        raise ValueError("Unrecognized split metadata format")
    class NoGlobals(pickle.Unpickler):
        def find_class(self, module, name):
            raise ValueError("Unexpected global")
    raw = NoGlobals(io.BytesIO(buf)).load()
    if len(raw) != 2:
        raise ValueError("Invalid official partition")
    ids = {side: {str(i) for i, _ in samples} for side, samples in zip(("train", "test"), raw)}
    if any(len(ids[side]) != len(samples) for side, samples in zip(("train","test"),raw)) or ids["train"] & ids["test"]:
        raise ValueError("Duplicate part ID in official metadata")
    return ids


def indexed_rows(root, workers=4):
    raw = Path(root) / DATA / "raw"
    ids = official_ids(raw / "split_weakly_246.pyb")
    tasks = []
    for side in ("train", "test"):
        files = {p.name for p in (raw / side).iterdir() if p.is_file()}
        expected = {f"{i}{suffix}.png" for i in ids[side] for suffix in ("", "_GT")}
        extras = {"10301 (copy).png", "10301_GT (copy).png"} if side == "train" else set()
        if files != expected | extras:
            raise ValueError(f"{side} file list mismatch, missing or extra: {sorted(files ^ (expected | extras))[:8]}")
        for i in sorted(ids[side]):
            tasks.append((side, i, False))
        if extras:
            tasks.append((side, "10301", True))

    def make(n):
        side, i, extra = n
        suffix = " (copy)" if extra else ""
        base = DATA / "raw" / side
        ip = base / f"{i}{suffix}.png"
        mp = base / f"{i}_GT{suffix}.png"
        pair = load_pair(Path(root) / ip, Path(root) / mp)
        return dict(sample_id=f"ksdd2-{i}" + ("-copy" if extra else ""),
                    image_path=ip.as_posix(), mask_path=mp.as_posix(), source_id=SOURCE,
                    official_split=side, part_id=i, split="excluded" if extra else side,
                    exclusion_reason="nonofficial_archive_copy" if extra else "", **pair)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(make, tasks))
    return sorted(rows, key=lambda r: r["sample_id"])


def assign(rows, seed=42):
    """Decontaminate official train, then stratify distinct eligible source groups."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["group_id"]].append(r)
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate IDs")
    eligible = defaultdict(list)
    for gid, members in sorted(groups.items()):
        if len({r["true_label"] for r in members}) > 1:
            raise ValueError(f"Conflicting ground truth for {gid}")
        if any(r["official_split"] == "test" for r in members):
            for r in members:
                if r["official_split"] == "train":
                    r.update(split="excluded", exclusion_reason="duplicate_of_official_test")
        else:
            active = [r for r in members if r["split"] != "excluded"]
            if active:
                eligible[active[0]["true_label"]].append(gid)
    for label, gids in eligible.items():
        ranked = sorted(gids, key=lambda g: by(f"{seed}:{g}".encode("ascii")))
        n = int(len(ranked) * .20 + .5)
        for idx, group in enumerate(ranked):
            for r in groups[group]:
                if r["split"] != "excluded":
                    r["split"] = "validation" if idx < n else "train"
    return rows


def write_csv(path, rows):
    path = Path(path)
    with path.with_suffix(".tmp").open("w", encoding="utf-8", newline="") as f:
        csvw = csv.DictWriter(f, fieldnames=FLDS, lineterminator="\n")
        csvw.writeheader()
        csvw.writerows(rows)
    path.with_suffix(".tmp").replace(path)


def generate(root, workers=4):
    out = Path(root) / DATA
    rows = assign(indexed_rows(root, workers))
    canonical = [r for r in rows if not r["sample_id"].endswith("-copy")]
    counts = Counter((r["official_split"], r["true_label"]) for r in canonical)
    if counts != {("train","defect"):246,("train","normal"):2085,("test","defect"):110,("test","normal"):894}:
        raise ValueError(f"Counts differ from official release: {counts}")
    write_csv(out / "manifest.csv", rows)
    write_csv(out / "exclusions.csv", [r for r in rows if r["split"]=="excluded"])
    positives = [r for r in rows if r["split"]=="train" and r["true_label"]=="defect"]
    def quantiles(data):
        # Explicit nearest rank on the sorted observed values.
        import math
        return {str(q): sorted(data)[max(0,math.ceil(len(data)*q)-1)] for q in (0,.1,.25,.5,.75,.9,1)}
    def counts_by(key, records):
        return {s: dict(Counter(r["true_label"] for r in records if r[key]==s)) for s in sorted({r[key] for r in records})}
    info = dict(dataset=SOURCE, source_url="https://www.vicos.si/resources/kolektorsdd2/",
                archive_url=URL, archive_sha256=ARCHIVE_SHA, archive_bytes=ARCHIVE_SIZE,
                archive_files=6677, archive_png_pairs=len(rows), official_ids=len(canonical),
                archive_http_last_modified_utc="2023-09-28T10:40:04Z", archive_http_etag='"65155804-32d9ad9b"',
                release_paper_year=2021, version_note="No semantic version published; pinned publisher archive SHA-256.",
                official_split_metadata_sha256={p.name:file_sha(p) for p in sorted((out/"raw").glob("*.pyb"))},
                canonical_official_counts=counts_by("official_split",canonical), counts=counts_by("split",rows),
                unique_image_file_hashes=len({r["file_sha256"] for r in rows}),
                unique_image_pixel_hashes=len({r["pixel_sha256"] for r in rows}),
                unique_d4_groups=len({r["group_id"] for r in rows}), split_seed=42,
                manifest_sha256=file_sha(out / "manifest.csv"),
                width_minmax=[min(r["width"] for r in rows),max(r["width"] for r in rows)],
                height_minmax=[min(r["height"] for r in rows),max(r["height"] for r in rows)],
                train_defect_pixels_nearest_rank=quantiles([r["defect_pixels"] for r in positives]),
                train_defect_area_percent_nearest_rank=quantiles([100*r["defect_pixels"]/(r["width"]*r["height"]) for r in positives]),
                cross_split_train_excluded=sum(r["exclusion_reason"]=="duplicate_of_official_test" for r in rows),
                product_or_coil_group="not supplied", license="CC BY-NC-SA 4.0")
    (out / "audit.json").write_text(json.dumps(info, ensure_ascii=False, indent=2)+"\n",encoding="utf-8")
    return info


def verify_files(root, split="all"):
    """Verify untouched original image and ground truth bytes in a subset or full install."""
    root = Path(root)
    mf = root / DATA / "manifest.csv"
    if file_sha(mf) != MANIFEST_SHA:
        raise ValueError("Released KSDD2 manifest SHA-256 mismatch")
    with mf.open(encoding="utf-8",newline="") as f:
        rows = [r for r in csv.DictReader(f) if split == "all" or r["split"] == split]
    if not rows:
        raise ValueError(f"No records for split {split}")
    for r in rows:
        for k,h in (("image_path","file_sha256"),("mask_path","mask_sha256")):
            if file_sha(root / r[k]) != r[h]:
                raise ValueError(f"Changed {r[k]}")
    return {"split": split, "verified_image_mask_pairs": len(rows), "manifest_sha256": MANIFEST_SHA}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("fetch", "index", "verify"))
    p.add_argument("--archive", type=Path, help="Path to official KolektorSDD2.zip for fetch/index")
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--split", choices=("all", "train", "validation", "test", "excluded"),default="all",
                   help="Required local image/mask set for verify (does not run inference)")
    args = p.parse_args()
    if args.action == "verify":
        print(json.dumps(verify_files(args.root, args.split), indent=2))
        return
    if args.archive is None:
        p.error("--archive is required for fetch/index")
    if args.action == "fetch":
        fetch(args.archive)
        print(f"{ARCHIVE_SHA}  {args.archive}")
    else:
        unpack(args.archive, args.root)
        print(json.dumps(generate(args.root, args.workers), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
