import sqlite3
from contextlib import closing
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from src import security


class SecurityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        for item in (patch.object(security, "SECURITY_ROOT", root / "security"),
                     patch.object(security, "AUDIT_ROOT", root / "audit"),
                     patch.dict("os.environ", {"STEELTRUST_AUDIT_URL": "", "STEELTRUST_AUDIT_BEARER": ""})):
            item.start()
            self.addCleanup(item.stop)
        security.bootstrap_admin("admin", "correct horse battery staple")
        self.token = security.login("admin", "correct horse battery staple")

    def test_authentication_roles_expiry_and_logout(self):
        with self.assertRaises(security.SecurityError):
            security.login("admin", "wrong password")
        with self.assertRaises(security.SecurityError):
            security.bootstrap_admin("other", "correct horse battery staple")
        security.create_user(self.token, "viewer", "another sufficiently long password", "viewer")
        viewer = security.login("viewer", "another sufficiently long password")
        self.assertEqual(security.authorize(viewer, roles=("viewer",))["role"], "viewer")
        with self.assertRaises(security.SecurityError):
            security.begin_action(viewer, "change", {})
        with self.assertRaises(security.SecurityError):
            security.create_user(viewer, "another", "another sufficiently long password", "operator")
        with closing(sqlite3.connect(security.SECURITY_ROOT / "accounts.sqlite")) as db, db:
            db.execute("UPDATE sessions SET expires=?", (time.time() - 1,))
        with self.assertRaises(security.SecurityError):
            security.authorize(self.token)
        self.token = security.login("admin", "correct horse battery staple")
        security.logout(self.token)
        with self.assertRaises(security.SecurityError):
            security.authorize(self.token)

    def test_audit_chain_and_tampering_blocks_action(self):
        event = security.begin_action(self.token, "model_registration", {"model_id": "M1"})
        security.finish_action(event, "completed")
        records = security.read_local_audit(self.token)
        self.assertEqual([x["phase"] for x in records], ["intent", "result"])
        self.assertEqual(records[0]["actor"], "admin")
        with closing(sqlite3.connect(security.AUDIT_ROOT / "audit.sqlite")) as db, db:
            db.execute("UPDATE events SET payload='{}' WHERE seq=1")
        with self.assertRaisesRegex(security.SecurityError, "무결성"):
            security.begin_action(self.token, "model_registration", {"model_id": "M2"})

    def test_concurrent_events_and_configured_external_failure(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda number: security.begin_action(self.token, "change", {"number": number}), range(8)))
        self.assertEqual(len(security.read_local_audit(self.token)), 8)
        with patch.dict("os.environ", {"STEELTRUST_AUDIT_URL": "https://example.invalid/audit",
                                       "STEELTRUST_AUDIT_BEARER": ""}):
            with self.assertRaises(security.SecurityError):
                security.begin_action(self.token, "change", {})
        self.assertEqual(len(security.read_local_audit(self.token)), 8)

    def test_sidebar_login_and_logout(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app_demo.py"), default_timeout=120).run()
        next(x for x in app.text_input if x.label == "계정").set_value("admin")
        next(x for x in app.text_input if x.label == "비밀번호").set_value("correct horse battery staple")
        next(x for x in app.button if x.label == "로그인").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("로그인: admin" in x.value for x in app.markdown))
        token = app.session_state["security-token"]
        next(x for x in app.button if x.label == "로그아웃").click().run()
        self.assertFalse(app.exception)
        with self.assertRaises(security.SecurityError):
            security.authorize(token)


if __name__ == "__main__":
    unittest.main()
