"""Register labelled images for development evaluations, never for an independent test.

ZIPs contain manifest.csv and matching images. Both import routes copy original image
bytes into a new immutable project; caller-supplied paths never select models, output
locations, or official splits. The installation's complete indexes are required even
in a demo package that includes only selected validation images.
"""
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import tempfile
import unicodedata
import zipfile
import zlib

from PIL import Image

from .inspection_cases import MAX_ARRAY_PAYLOAD
from .inspection_data import MANIFEST_SHA as KSDD2_INDEX_SHA

ROOT = Path(__file__).resolve().parents[1]
MAX_ARCHIVE_BYTES = 100 << 20
MAX_EXPANDED_BYTES = 150 << 20
MAX_IMAGE_BYTES = 16 << 20
MAX_CSV_BYTES = 1 << 20
MAX_IMAGES = 500
MAX_ENTRIES = 2 * MAX_IMAGES + 32
MAX_DIMENSION = 4096
MAX_TOTAL_PIXELS = 128_000_000
LABELS = {
    "ksdd2": frozenset(("normal", "defect")),
    "neu_cls6": frozenset(("crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches")),
}
FORMATS = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".bmp": "BMP"}
EXT = {"PNG": ".png", "JPEG": ".jpg", "BMP": ".bmp"}
COLUMNS = ("sample_id", "image_path", "true_label", "group_id", "split", "source_id", "file_sha256")
REQUIRED = frozenset(("sample_id", "image_path", "true_label"))
OPTIONAL = frozenset(("group_id", "source_id"))
RESERVED = re.compile(r"^(?:CON|PRN|AUX|NUL|CLOCK\$|CONIN\$|CONOUT\$|(?:COM|LPT)[1-9¹²³])(?:\.|$)", re.I)
SAMPLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
DIGEST = re.compile(r"[0-9a-f]{64}\Z")
INDEX_SHA = {"ksdd2": KSDD2_INDEX_SHA, "neu_cls6":
             "139bef29ba80ab6d0de0d8972cf77924aa83515ebe9f12859c02a91eb3229676"}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _text(value, field, limit):
    if (not isinstance(value, str) or not value.strip() or len(value) > limit
            or any(unicodedata.category(ch).startswith("C") for ch in value)):
        raise ValueError(f"{field}: 비어 있거나 지원하지 않는 문자열입니다.")
    return value.strip()


def _options(title, template_id):
    if template_id not in LABELS:
        raise ValueError("등록된 검사 모델 종류를 선택하세요: ksdd2 또는 neu_cls6.")
    return _text(title, "프로젝트 이름", 120)


def _path(name):
    """Validate the raw spelling before a library can collapse dots or slashes."""
    if (not isinstance(name, str) or not name or len(name.encode("utf8")) > 512
            or any(c in name for c in '\\:<>"|?*') or
            any(unicodedata.category(c).startswith("C") for c in name)):
        raise ValueError("허용되지 않는 파일 경로입니다. 상대 경로와 /를 사용하세요.")
    for part in name.split("/"):
        if (part in ("", ".", "..") or part.endswith((".", " ")) or RESERVED.match(part)
                or len(part.encode("utf8")) > 200):
            raise ValueError(f"상대 경로 또는 파일 이름을 확인하세요: {name}")
    return name


def _key(name):
    return unicodedata.normalize("NFC", name).casefold()


def _format(name):
    ext = PurePosixPath(name).suffix.lower()
    if ext not in FORMATS:
        raise ValueError(f"지원하지 않는 파일입니다: {name}. PNG, JPG/JPEG, BMP 이미지와 정답 CSV만 사용하세요.")
    return FORMATS[ext]


def _table(blob, template_id):
    if not blob or len(blob) > MAX_CSV_BYTES:
        raise ValueError("정답 CSV가 비었거나 1 MiB를 초과했습니다. 정답 없이는 성능을 평가할 수 없습니다.")
    try:
        rd = csv.DictReader(io.StringIO(blob.decode("utf-8-sig"), newline=""), strict=True)
        fields = rd.fieldnames or []
        if not REQUIRED <= set(fields):
            raise ValueError("정답 CSV에 sample_id, image_path, true_label이 필요합니다. 정답 없이는 성능을 평가할 수 없습니다.")
        if set(fields) - REQUIRED - OPTIONAL or len(set(fields)) != len(fields):
            raise ValueError("정답 CSV에는 필수 열과 선택 열 group_id, source_id만 허용합니다. split은 직접 지정할 수 없습니다.")
        rows, ids, names = [], set(), set()
        for number, obj in enumerate(rd, 2):
            if None in obj or any(value is None for value in obj.values()):
                raise ValueError(f"CSV {number}행의 열 개수가 일치하지 않습니다.")
            sid = obj["sample_id"]
            if not SAMPLE_ID.fullmatch(sid) or RESERVED.match(sid) or sid.endswith("."):
                raise ValueError(f"CSV {number}행 sample_id: 영문·숫자로 시작하는 1~96자의 영문, 숫자, _, -, .만 허용합니다.")
            name = _path(obj["image_path"])
            _format(name)
            label = obj["true_label"].strip()
            if not label or label not in LABELS[template_id]:
                raise ValueError(f"{sid}: 정답이 없거나 클래스가 맞지 않아 평가할 수 없습니다. 허용: {', '.join(sorted(LABELS[template_id]))}")
            if sid.casefold() in ids or _key(name) in names:
                raise ValueError(f"{sid}: sample_id 또는 이미지 경로가 중복되었습니다.")
            ids.add(sid.casefold())
            names.add(_key(name))
            item = dict(sample_id=sid, image_path=name, true_label=label)
            for field in OPTIONAL:
                value = obj.get(field) or ""
                if value:
                    value = _text(value, field, 120)
                    if value[0] in "=+-@":
                        raise ValueError(f"{field}: 수식으로 시작하는 값은 허용하지 않습니다.")
                item[field] = value
            rows.append(item)
            if len(rows) > MAX_IMAGES:
                raise ValueError("프로젝트당 최대 500개 이미지를 등록할 수 있습니다.")
    except (csv.Error, UnicodeDecodeError) as e:
        raise ValueError("정답은 UTF-8 CSV로 제공하세요.") from e
    if not rows:
        raise ValueError("정답 CSV에 이미지 행이 없습니다. 정답 없이는 성능을 평가할 수 없습니다.")
    return sorted(rows, key=lambda r: r["sample_id"])


@dataclass(frozen=True)
class _Official:
    template: str
    sid: str
    group: str
    split: str
    label: str


def _guard():
    """Never read raw official test images for predictions. Only consult trusted index metadata."""
    entries, masks = {}, set()
    for template, path, columns in [
        ("ksdd2", ROOT / "data/ksdd2/manifest.csv",
         ("sample_id", "true_label", "group_id", "file_sha256", "pixel_sha256", "mask_sha256", "mask_pixel_sha256", "official_split", "split")),
        ("neu_cls6", ROOT / "experiments/surface_v1/split_manifest.csv",
         ("sample_id", "label", "parent_image_id", "file_sha256", "pixels_sha256", "canonical_sha256", "split")),
    ]:
        splits = set()
        try:
            raw = path.read_bytes()
            if _sha(raw) != INDEX_SHA[template]:
                raise ValueError("official split index checksum")
            with io.StringIO(raw.decode("utf-8-sig"), newline="") as f:
                rd = csv.DictReader(f, strict=True)
                if not set(columns).issubset(rd.fieldnames or ()):
                    raise KeyError("missing columns")
                for r in rd:
                    split = r["split"]
                    if split not in {"train", "validation", "test", "excluded"}:
                        raise ValueError("unknown split")
                    if template == "ksdd2":
                        if r["official_split"] not in {"test", "train"}:
                            raise ValueError("unknown official_split")
                        if r["official_split"] == "test":
                            split = "test"
                        obj = _Official(template, r["sample_id"], r["group_id"], split, r["true_label"])
                        hashes = (r["file_sha256"], r["pixel_sha256"])
                        mask_hashes = (r["mask_sha256"], r["mask_pixel_sha256"])
                        if not all(DIGEST.fullmatch(h) for h in mask_hashes):
                            raise ValueError("mask hashes")
                        masks.update(mask_hashes)
                    else:
                        obj = _Official(template, r["sample_id"], r["parent_image_id"], split, r["label"])
                        hashes = (r["file_sha256"], r["pixels_sha256"], r["canonical_sha256"])
                    if not all(DIGEST.fullmatch(h) for h in hashes) or not obj.group or obj.label not in LABELS[template]:
                        raise ValueError("invalid official record")
                    splits.add(split)
                    for h in hashes:
                        entries.setdefault(h, set()).add(obj)
                if not {"test", "train", "validation"} <= splits:
                    raise ValueError("incomplete splits")
        except (OSError, csv.Error, KeyError, TypeError, ValueError) as e:
            raise ValueError(f"학습·시험 원본의 중복을 확인할 수 없습니다. 공식 전체 인덱스를 복원하세요: {path.name}") from e
    return entries, masks


def _orientations(im):
    yield im
    for op in (Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM, Image.Transpose.ROTATE_90,
               Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270, Image.Transpose.TRANSPOSE, Image.Transpose.TRANSVERSE):
        yield im.transpose(op)


def _image(blob, name, template_id, remaining):
    if not blob or len(blob) > MAX_IMAGE_BYTES:
        raise ValueError(f"{name}: 이미지 크기는 1바이트~16 MiB여야 합니다.")
    fmt = _format(name)
    try:
        with Image.open(io.BytesIO(blob), formats=(fmt,)) as im:
            width, height = im.size
            count = width * height
            if max(im.size) > MAX_DIMENSION or count > remaining:
                raise ValueError(f"{name}: 한 변은 최대 {MAX_DIMENSION}픽셀, 프로젝트 합계는 최대 {MAX_TOTAL_PIXELS:,}픽셀입니다.")
            if getattr(im, "n_frames", 1) != 1 or im.mode not in {"RGB", "L"} or "transparency" in im.info:
                raise ValueError(f"{name}: 단일 프레임의 불투명 8비트 RGB 또는 회색조 이미지를 사용하세요.")
            if template_id == "neu_cls6" and (im.mode != "L" or im.size != (200, 200)):
                raise ValueError(f"{name}: NEU 모델에는 200×200 회색조 원본이 필요합니다.")
            # Both the original and the changed input are stored as separate float32 arrays.
            # The binary adapter expands even an L upload to three native RGB channels.
            channels = 3 if template_id == "ksdd2" else 1
            case_pixel_limit = MAX_ARRAY_PAYLOAD // (4 * channels)
            if count > case_pixel_limit:
                raise ValueError(f"{name}: 사례 저장 한도는 이미지당 {case_pixel_limit:,}픽셀입니다. 현재 {width}×{height}픽셀입니다.")
            im.verify()
        # verify() checks checksums for PNG; load() also forces full pixel decoding for JPEG and BMP.
        with Image.open(io.BytesIO(blob), formats=(fmt,)) as im:
            im.load()
            rgb = im.convert("RGB")
    except (OSError, SyntaxError, Image.DecompressionBombError) as e:
        raise ValueError(f"{name}: 이미지가 손상되었거나 파일 형식과 확장자가 맞지 않습니다.") from e
    keys = {_sha(blob)}
    colors = []
    for v in _orientations(rgb):
        h = hashlib.sha256(f"{v.width}x{v.height}:RGB:".encode("ascii"))
        h.update(v.tobytes())
        colors.append(h.hexdigest())
        if v.size == (200, 200):  # NEU legacy fingerprints were computed over native luminance bytes.
            keys.add(_sha(v.convert("L").tobytes()))
    keys.update(colors)
    return count, min(colors), keys


def _native_flag(st):
    return stat.S_ISLNK(st.st_mode) or bool(getattr(st, "st_file_attributes", 0) & 0x400)


def _real_path(path):
    path = Path(os.path.abspath(os.fspath(path)))
    for p in [*reversed(path.parents), path]:
        if _native_flag(p.lstat()):
            raise ValueError(f"심볼릭 링크·연결 경로는 사용할 수 없습니다: {p.name}")
    return path


def _read_file(path, limit):
    path = _real_path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit:
        raise ValueError(f"일반 파일이 아니거나 크기를 초과했습니다: {path.name}")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(fd, "rb") as f:
        current = os.fstat(fd)
        if not stat.S_ISREG(current.st_mode) or not os.path.samestat(current, info):
            raise ValueError(f"읽는 동안 파일이 변경되었습니다: {path.name}")
        data = f.read(limit + 1)
    if len(data) != info.st_size or len(data) > limit:
        raise ValueError(f"읽는 동안 파일 크기가 변경되었습니다: {path.name}")
    return data


def _manifest_bytes(rows):
    sio = io.StringIO(newline="")
    wr = csv.DictWriter(sio, fieldnames=COLUMNS, lineterminator="\n")
    wr.writeheader()
    wr.writerows(rows)
    return sio.getvalue().encode("utf8")


def _existing(project, expected, data_sha, rows):
    if not project.is_dir() or _real_path(project) != project:
        raise ValueError("등록 프로젝트 위치가 손상되었습니다.")
    cfg = json.loads(_read_file(project / "project.json", MAX_CSV_BYTES))
    if (cfg.get("manifest_sha256") != _sha(expected) or cfg.get("dataset_sha256") != data_sha
            or cfg.get("id") != project.name or _read_file(project / "manifest.csv", MAX_CSV_BYTES) != expected):
        raise ValueError("기존 등록 정보의 무결성 확인에 실패했습니다.")
    for r in rows:
        if _sha(_read_file(ROOT / r["image_path"], MAX_IMAGE_BYTES)) != r["file_sha256"]:
            raise ValueError("기존 등록 이미지의 해시가 달라졌습니다.")


def _publish(blob, image_names, read, title, template_id):
    rows = _table(blob, template_id)
    if {r["image_path"] for r in rows} != image_names:
        raise ValueError("정답 목록과 실제 이미지 목록이 다릅니다. 누락·미라벨 이미지와 경로를 확인하세요.")
    known, masks = _guard()
    directory = ROOT / "inspection_projects"
    directory.mkdir(exist_ok=True)
    _real_path(directory)
    with tempfile.TemporaryDirectory(dir=directory, prefix=".register-") as scratch:
        stage = Path(scratch) / "project"
        (stage / "images").mkdir(parents=True)
        output, seen = [], set()
        room = MAX_TOTAL_PIXELS
        for r in rows:
            name = r["image_path"]
            data = read(name)
            digest = _sha(data)
            count, canonical, keys = _image(data, name, template_id, room)
            room -= count
            if keys & masks:
                raise ValueError(f"{r['sample_id']}: 공식 정답 마스크를 추론 이미지로 등록할 수 없습니다.")
            matches = set().union(*(known[k] for k in keys if k in known))
            if any(m.split != "validation" for m in matches):
                raise ValueError(f"{r['sample_id']}: 공식 학습·시험·제외 원본과 중복되어 개발 평가에 등록할 수 없습니다.")
            if any(m.template != template_id or m.label != r["true_label"] for m in matches):
                raise ValueError(f"{r['sample_id']}: 공식 검증 원본과 모델 종류 또는 정답이 다릅니다.")
            if canonical in seen:
                raise ValueError(f"{r['sample_id']}: 같은 원본 픽셀이 중복되었습니다.")
            seen.add(canonical)
            group = r["group_id"] or f"unverified-{canonical[:24]}"
            source = "user_csv" + (":" + r["source_id"] if r["source_id"] else "")
            if matches:
                ref = min(matches, key=lambda m: m.sid)
                group = ref.group
                source = f"{ref.template}:validation:{ref.sid}"
            local = "images/" + digest + EXT[_format(name)]
            (stage / local).write_bytes(data)
            output.append(dict(sample_id=r["sample_id"], image_path=local, true_label=r["true_label"],
                               split="validation", group_id=group, source_id=source, file_sha256=digest))
        semantic = json.dumps([template_id, output], sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        data_sha = _sha(semantic.encode("ascii"))
        pid = "user_" + data_sha[:24]
        rel = PurePosixPath("inspection_projects", pid)
        for r in output:
            r["image_path"] = (rel / r["image_path"]).as_posix()
        manifest = _manifest_bytes(output)
        cfg = dict(id=pid, title=title, template_id=template_id, manifest_sha256=_sha(manifest),
                   dataset_sha256=data_sha, role="user_development_validation", sample_count=len(output))
        (stage / "manifest.csv").write_bytes(manifest)
        (stage / "project.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
        dest = directory / pid
        if not os.path.lexists(dest):
            try:
                stage.rename(dest)  # same filesystem: readers see all the verified content or nothing.
                return pid
            except OSError:
                if not os.path.lexists(dest):
                    raise
        _existing(dest, manifest, data_sha, output)
        return pid


def _zip_directory_limit(data):
    # ZipFile creates one Python object per central entry at construction. Bound the
    # directory *before* opening it, including archives that lie about their count.
    eocd = data.rfind(b"PK\x05\x06", max(0, len(data) - (65535 + 22)))
    if eocd < 0 or len(data) - eocd < 22:
        raise ValueError("완전한 ZIP 파일을 선택하세요.")
    disk, home, entries, count, size, start, comment = struct.unpack_from("<4H2IH", data, eocd + 4)
    if (disk or home or entries != count or not 0 < count <= MAX_ENTRIES or size > 2 << 20
            or start + size != eocd or eocd + 22 + comment != len(data)):
        raise ValueError("단일 ZIP 파일을 선택하세요. 분할·ZIP64·과도한 항목 수는 지원하지 않습니다.")
    return count


def register_zip(data: bytes, title: str, template_id: str) -> str:
    title = _options(title, template_id)
    if not isinstance(data, bytes) or not 0 < len(data) <= MAX_ARCHIVE_BYTES:
        raise ValueError("비어 있지 않은 100 MiB 이하 ZIP을 선택하세요.")
    try:
        count = _zip_directory_limit(data)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            infos = z.infolist()
            if len(infos) != count or sum(x.file_size for x in infos) > MAX_EXPANDED_BYTES:
                raise ValueError("ZIP 항목 수 또는 압축 해제 크기(150 MiB)를 초과했습니다.")
            members, names, total = {}, set(), set()
            for info in infos:
                isdir = info.is_dir()
                # orig_filename retains the suffix after NUL which ZipInfo.filename strips.
                name = _path(info.orig_filename.removesuffix("/") if isdir else info.orig_filename)
                key = _key(name)
                mode = stat.S_IFMT(info.external_attr >> 16)
                if (key in total or mode not in (0, stat.S_IFDIR if isdir else stat.S_IFREG)
                        or info.external_attr & 0x400 or info.flag_bits & 1
                        or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
                    raise ValueError(f"중복 경로·링크·암호 또는 지원하지 않는 ZIP 항목입니다: {name}")
                total.add(key)
                if isdir:
                    if info.file_size:
                        raise ValueError("잘못된 ZIP 디렉터리 항목입니다.")
                    continue
                limit = MAX_CSV_BYTES if name == "manifest.csv" else MAX_IMAGE_BYTES
                if not 0 < info.file_size <= limit:
                    raise ValueError(f"{name}: 파일 크기를 확인하세요.")
                if name != "manifest.csv":
                    _format(name)
                    names.add(name)
                members[name] = info
            files = {_key(n) for n in members}
            if any(_key(p.as_posix()) in files for n in total for p in PurePosixPath(n).parents):
                raise ValueError("파일과 폴더 경로가 충돌합니다.")
            if "manifest.csv" not in members:
                raise ValueError("ZIP 맨 위에 정답 manifest.csv가 필요합니다. 정답 없이는 성능을 평가할 수 없습니다.")
            def read(name):
                entry = members[name]
                with z.open(entry) as stream:
                    blob = stream.read(entry.file_size + 1)
                if len(blob) != entry.file_size:
                    raise ValueError(f"ZIP 항목 크기가 일치하지 않습니다: {name}")
                return blob
            return _publish(read("manifest.csv"), names, read, title, template_id)
    except (zipfile.BadZipFile, OSError, EOFError, UnicodeError, zlib.error) as e:
        raise ValueError("ZIP이 손상되었거나 등록 파일을 읽고 저장할 수 없습니다.") from e


def register_folder(folder: Path, csv_path: Path, title: str, template_id: str) -> str:
    """Local/CLI import. CSV may be a separately selected file outside the image folder."""
    title = _options(title, template_id)
    try:
        folder, csv_path = _real_path(folder), _real_path(csv_path)
        if not folder.is_dir() or csv_path.suffix.lower() != ".csv":
            raise ValueError("이미지 폴더와 정답 CSV 파일을 선택하세요.")
        csv_bytes = _read_file(csv_path, MAX_CSV_BYTES)
        paths, keys = {}, set()
        count, total = 0, len(csv_bytes)
        pending = [folder]
        while pending:
            current = pending.pop()
            with os.scandir(current) as items:
                for item in items:
                    count += 1
                    path = Path(item.path)
                    name = _path(path.relative_to(folder).as_posix())
                    key = _key(name)
                    info = item.stat(follow_symlinks=False)
                    if count > MAX_ENTRIES or _native_flag(info) or key in keys:
                        raise ValueError("연결 경로, 중복 이름 또는 너무 많은 파일은 등록할 수 없습니다.")
                    keys.add(key)
                    if stat.S_ISDIR(info.st_mode):
                        pending.append(path)
                    elif path != csv_path:
                        _format(name)
                        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_IMAGE_BYTES:
                            raise ValueError(f"{name}: 16 MiB 이하 일반 이미지 파일만 허용합니다.")
                        total += info.st_size
                        paths[name] = path
                    if total > MAX_EXPANDED_BYTES or len(paths) > MAX_IMAGES:
                        raise ValueError("이미지 수 또는 폴더 크기(150 MiB) 제한을 초과했습니다.")
        return _publish(csv_bytes, set(paths), lambda n: _read_file(paths[n], MAX_IMAGE_BYTES), title, template_id)
    except OSError as e:
        raise ValueError("이미지 폴더와 정답 CSV의 위치·권한을 확인하세요.") from e
