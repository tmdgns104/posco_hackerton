"""Validate a bounded in-memory G1 case archive, then call the original replay."""
import io
import json
import zipfile
import zlib

import numpy as np

from .surface_detector import read_raw

MAX_ZIP = 1024 * 1024
LIMITS = {"case.json":65536, "raw_input.npy":162048, "original_preview.png":150000, "noise_preview.png":150000}


def replay_upload(run, data):
    if not 0 < len(data) <= MAX_ZIP:
        raise ValueError("사례 ZIP은 1 MiB 이하여야 합니다.")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = archive.namelist()
            if not {"case.json","raw_input.npy"}.issubset(names):
                raise ValueError("필수 case.json 또는 raw_input.npy가 없습니다.")
            if len(set(names)) != len(names) or any(n not in LIMITS for n in names):
                raise ValueError("허용되지 않는 항목이나 중복된 이름이 있습니다.")
            for info in archive.infolist():
                if (info.file_size > LIMITS[info.filename] or info.flag_bits & 1
                        or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
                    raise ValueError("지원하지 않는 압축 방식 또는 항목 크기입니다.")
            meta = json.loads(archive.read("case.json"))
            if not isinstance(meta,dict) or meta.get("experiment") != "surface_g1" or meta.get("run_id") != run.id:
                raise ValueError("이 실행본에서 생성한 G1 사례가 아닙니다.")
            buf = io.BytesIO(archive.read("raw_input.npy"))
            if np.lib.format.read_magic(buf) != (1, 0):
                raise ValueError("G1 내보내기 형식의 NPY가 필요합니다.")
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(buf, max_header_size=1024)
            if shape != (200,200) or dtype != np.dtype("<f4") or fortran or buf.getbuffer().nbytes-buf.tell() != 160000:
                raise ValueError("원시 입력은 200×200 float32 배열이어야 합니다.")
            buf.seek(0)
            raw = np.load(buf, allow_pickle=False)
        # Fresh per-upload feature cache forces real inference even after viewing the same case.
        from copy import copy
        verifier = copy(run)
        verifier.cache = {}
        record = verifier.replay(io.BytesIO(data))
        original = read_raw(run.meta.loc[record["sample_id"]])
        return record, original, raw
    except (zipfile.BadZipFile, zlib.error, EOFError, KeyError, TypeError, IndexError, UnicodeError, OverflowError, RecursionError) as exc:
        raise ValueError("ZIP 형식, 필수 항목 또는 모델·입력 정보가 올바르지 않습니다.") from exc
