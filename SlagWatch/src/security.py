"""Local accounts and fail-closed external audit for model mutations."""
from datetime import datetime, timezone
from contextlib import closing
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener
import uuid

from filelock import FileLock, Timeout

from .surface_data import ROOT

SECURITY_ROOT = ROOT / "steeltrust_security"
AUDIT_ROOT = ROOT / "steeltrust_audit_local"
USERNAME = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{2,63}\Z")
ROLES = {"admin", "operator", "viewer"}
SESSION_SECONDS = 8 * 60 * 60


class SecurityError(ValueError):
    pass


def _db():
    root = Path(SECURITY_ROOT)
    if root.is_symlink():
        raise SecurityError("보안 저장 경로가 심볼릭 링크입니다.")
    root.mkdir(parents=True, exist_ok=True)
    path = root / "accounts.sqlite"
    if path.is_symlink():
        raise SecurityError("계정 저장 파일이 심볼릭 링크입니다.")
    db = sqlite3.connect(path, timeout=10)
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY, role TEXT NOT NULL, salt BLOB NOT NULL,
            password_hash BLOB NOT NULL, failures INTEGER NOT NULL DEFAULT 0,
            locked_until REAL NOT NULL DEFAULT 0, enabled INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS sessions (
            token_hash TEXT PRIMARY KEY, username TEXT NOT NULL REFERENCES users(username),
            expires REAL NOT NULL
        );
    """)
    return db


def _hash_password(password, salt):
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=16384, r=8, p=1, dklen=32)


def bootstrap_admin(username, password):
    if not isinstance(username, str) or not USERNAME.fullmatch(username):
        raise SecurityError("계정은 영문으로 시작하는 3~64자의 영문·숫자·._-이어야 합니다.")
    if not isinstance(password, str) or not 12 <= len(password) <= 1024:
        raise SecurityError("비밀번호는 12~1024자여야 합니다.")
    salt = secrets.token_bytes(16)
    digest = _hash_password(password, salt)
    with closing(_db()) as db, db:
        db.execute("BEGIN IMMEDIATE")
        if db.execute("SELECT COUNT(*) FROM users").fetchone()[0]:
            raise SecurityError("초기 관리자가 이미 존재합니다.")
        db.execute("INSERT INTO users(username,role,salt,password_hash) VALUES(?,?,?,?)",
                   (username, "admin", salt, digest))


def login(username, password):
    if not isinstance(username, str) or not isinstance(password, str) or len(password) > 1024:
        raise SecurityError("계정 또는 비밀번호가 올바르지 않습니다.")
    with closing(_db()) as db, db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT role,salt,password_hash,failures,locked_until,enabled FROM users WHERE username=?",
                         (username,)).fetchone()
        if row is None:
            _hash_password(password, b"\0" * 16)
            raise SecurityError("계정 또는 비밀번호가 올바르지 않습니다.")
        role, salt, digest, failures, locked_until, enabled = row
        valid = hmac.compare_digest(_hash_password(password, salt), digest)
        if not enabled or locked_until > time.time() or not valid:
            if enabled and locked_until <= time.time():
                failures += 1
                db.execute("UPDATE users SET failures=?,locked_until=? WHERE username=?",
                           (failures, time.time() + 600 if failures >= 5 else 0, username))
                db.commit()
            raise SecurityError("계정 또는 비밀번호가 올바르지 않거나 계정이 잠겼습니다.")
        token = secrets.token_urlsafe(32)
        db.execute("UPDATE users SET failures=0,locked_until=0 WHERE username=?", (username,))
        db.execute("INSERT INTO sessions(token_hash,username,expires) VALUES(?,?,?)",
                   (hashlib.sha256(token.encode()).hexdigest(), username, time.time()+SESSION_SECONDS))
        return token


def authorize(token, *, roles=("admin", "operator")):
    if not isinstance(token, str) or not token or len(token) > 256:
        raise SecurityError("인증된 사용자 로그인이 필요합니다.")
    with closing(_db()) as db, db:
        row = db.execute("""SELECT users.username,users.role,users.enabled,sessions.expires
                            FROM sessions JOIN users ON users.username=sessions.username
                            WHERE sessions.token_hash=?""",
                         (hashlib.sha256(token.encode()).hexdigest(),)).fetchone()
    if not row or not row[2] or row[3] <= time.time() or row[1] not in roles:
        raise SecurityError("유효한 로그인과 작업 권한이 필요합니다.")
    return {"username": row[0], "role": row[1]}


def logout(token):
    if isinstance(token, str) and token:
        with closing(_db()) as db, db:
            db.execute("DELETE FROM sessions WHERE token_hash=?", (hashlib.sha256(token.encode()).hexdigest(),))


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _verified_events(db, key):
    previous = "0" * 64
    result = []
    for stored_payload, stored_mac in db.execute("SELECT payload,mac FROM events ORDER BY seq"):
        expected = hmac.new(key, (previous + stored_payload).encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, stored_mac):
            raise SecurityError("로컬 감사 기록 무결성이 일치하지 않습니다.")
        previous = stored_mac
        result.append(json.loads(stored_payload))
    return previous, result


def _local_audit(event):
    root = Path(AUDIT_ROOT)
    if root.is_symlink():
        raise SecurityError("로컬 감사 저장 경로가 심볼릭 링크입니다.")
    try:
        root.mkdir(parents=True, exist_ok=True)
        with FileLock(str(root / ".audit.lock"), timeout=10):
            key_path = root / "audit.key"
            db_path = root / "audit.sqlite"
            if key_path.is_symlink() or db_path.is_symlink():
                raise SecurityError("로컬 감사 파일이 심볼릭 링크입니다.")
            if key_path.exists() != db_path.exists():
                raise SecurityError("로컬 감사 저장소 또는 무결성 키가 누락되었습니다.")
            if not key_path.exists():
                with key_path.open("xb") as stream:
                    stream.write(secrets.token_bytes(32))
                    stream.flush()
                    os.fsync(stream.fileno())
            key = key_path.read_bytes()
            if len(key) != 32:
                raise SecurityError("로컬 감사 무결성 키가 올바르지 않습니다.")
            payload = json.dumps(event, sort_keys=True, ensure_ascii=False, allow_nan=False)
            with closing(sqlite3.connect(db_path, timeout=10)) as db, db:
                db.execute("PRAGMA synchronous=FULL")
                db.execute("CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY AUTOINCREMENT, payload TEXT NOT NULL, mac TEXT NOT NULL)")
                db.execute("BEGIN IMMEDIATE")
                previous, _ = _verified_events(db, key)
                mac = hmac.new(key, (previous + payload).encode("utf-8"), hashlib.sha256).hexdigest()
                db.execute("INSERT INTO events(payload,mac) VALUES(?,?)", (payload, mac))
    except (OSError, sqlite3.Error, Timeout) as exc:
        raise SecurityError("로컬 감사 저장소에 기록할 수 없어 작업이 차단되었습니다.") from exc


def read_local_audit(token):
    authorize(token, roles=tuple(ROLES))
    root = Path(AUDIT_ROOT)
    if not root.exists():
        return []
    if root.is_symlink() or (root / "audit.key").is_symlink() or (root / "audit.sqlite").is_symlink():
        raise SecurityError("로컬 감사 경로가 심볼릭 링크입니다.")
    try:
        with FileLock(str(root / ".audit.lock"), timeout=10):
            key = (root / "audit.key").read_bytes()
            if len(key) != 32:
                raise SecurityError("로컬 감사 무결성 키가 올바르지 않습니다.")
            if not (root / "audit.sqlite").is_file():
                raise SecurityError("로컬 감사 저장소가 없습니다.")
            with closing(sqlite3.connect(root / "audit.sqlite", timeout=10)) as db, db:
                _, events = _verified_events(db, key)
                return events
    except (OSError, sqlite3.Error, Timeout, json.JSONDecodeError) as exc:
        raise SecurityError("로컬 감사 기록을 읽을 수 없습니다.") from exc


def send_audit(event):
    url = os.environ.get("STEELTRUST_AUDIT_URL", "").strip()
    token = os.environ.get("STEELTRUST_AUDIT_BEARER", "").strip()
    if not url and not token:
        _local_audit(event)
        return
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.fragment or not token:
        raise SecurityError("외부 감사 HTTPS 주소와 인증 설정이 없어 중요 작업이 차단되었습니다.")
    payload = json.dumps(event, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    request = Request(url, data=payload, method="POST",
                      headers={"Content-Type": "application/json", "Authorization": "Bearer " + token})
    try:
        with build_opener(_NoRedirect(), HTTPSHandler()).open(request, timeout=5) as response:
            if not 200 <= response.status < 300:
                raise SecurityError(f"외부 감사 서버가 HTTP {response.status}을 반환했습니다.")
    except (HTTPError, URLError, OSError, TimeoutError) as exc:
        raise SecurityError("외부 감사 서버에 기록할 수 없어 작업이 차단되었습니다.") from exc


def begin_action(token, action, details, *, roles=("admin", "operator")):
    principal = authorize(token, roles=roles)
    event = {"event_id": uuid.uuid4().hex, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
             "actor": principal["username"], "role": principal["role"], "action": action,
             "phase": "intent", "details": details}
    send_audit(event)
    return event


def finish_action(event, outcome):
    try:
        send_audit({**event, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "phase": "result", "outcome": outcome})
    except (SecurityError, ValueError) as exc:
        raise SecurityError("작업이 이미 저장되었을 수 있지만 감사 결과 기록을 확인하지 못했습니다. 현재 상태를 확인하세요.") from exc


def create_user(token, username, password, role):
    if not isinstance(username, str) or not USERNAME.fullmatch(username):
        raise SecurityError("계정 이름이 올바르지 않습니다.")
    if not isinstance(password, str) or not 12 <= len(password) <= 1024 or role not in ROLES:
        raise SecurityError("비밀번호 또는 역할이 올바르지 않습니다.")
    event = begin_action(token, "account_creation", {"username": username, "role": role}, roles=("admin",))
    salt = secrets.token_bytes(16)
    digest = _hash_password(password, salt)
    try:
        with closing(_db()) as db, db:
            db.execute("INSERT INTO users(username,role,salt,password_hash) VALUES(?,?,?,?)",
                       (username, role, salt, digest))
    except sqlite3.IntegrityError as exc:
        raise SecurityError("이미 존재하는 계정입니다.") from exc
    finish_action(event, "completed")
