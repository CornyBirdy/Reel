"""Updater tests: real git repos for the git path, a fake GitHub server for the ZIP path.

    python -m unittest discover tests
"""
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reel import config, updater  # noqa: E402

GIT = shutil.which("git")
ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, env=ENV, check=True, capture_output=True, text=True).stdout.strip()


def write(root: Path, rel: str, text: str):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


class Patched(unittest.TestCase):
    """Point the updater at a throwaway install folder."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "install"
        self.root.mkdir()
        self._saved = (updater.ROOT, config.DATA_DIR, config.SETTINGS_FILE, updater.API, updater.REPO, updater.BRANCH)
        updater.ROOT = self.root
        config.DATA_DIR = self.root / "data"
        config.SETTINGS_FILE = config.DATA_DIR / "settings.json"
        updater.BRANCH = "main"
        updater._invalidate()

    def tearDown(self):
        updater.ROOT, config.DATA_DIR, config.SETTINGS_FILE, updater.API, updater.REPO, updater.BRANCH = self._saved
        updater._invalidate()
        shutil.rmtree(self.tmp, ignore_errors=True)


@unittest.skipUnless(GIT, "git not installed")
class GitMode(Patched):
    def setUp(self):
        super().setUp()
        self.origin = self.tmp / "origin.git"
        self.src = self.tmp / "src"
        git(self.tmp, "init", "-q", "--bare", "-b", "main", str(self.origin))
        git(self.tmp, "clone", "-q", str(self.origin), str(self.src))
        git(self.src, "checkout", "-q", "-b", "main")
        write(self.src, "reel/__init__.py", "")
        write(self.src, "start.bat", "v1")
        write(self.src, "requirements.txt", "fastapi\n")
        write(self.src, "old.txt", "will be deleted")
        write(self.src, ".gitignore", "data/\n")
        git(self.src, "add", "-A"); git(self.src, "commit", "-qm", "first"); git(self.src, "push", "-q", "origin", "main")
        shutil.rmtree(self.root)
        git(self.tmp, "clone", "-q", "-b", "main", str(self.origin), str(self.root))
        write(self.root, "data/models/m/keep.bin", "weights")   # ignored user data
        os.environ["GIT_TERMINAL_PROMPT"] = "0"
        self.old_url = git(self.root, "remote", "get-url", "origin")

    def upstream_commit(self, msg="second"):
        write(self.src, "requirements.txt", "fastapi\nuvicorn\n")
        write(self.src, "reel/new.py", "x = 1")
        (self.src / "old.txt").unlink()
        git(self.src, "add", "-A"); git(self.src, "commit", "-qm", msg); git(self.src, "push", "-q", "origin", "main")

    def test_up_to_date(self):
        self.assertEqual(updater.mode(), "git")
        self.assertEqual(updater.check(force=True)["state"], "up_to_date")

    def test_update_fast_forwards_and_keeps_user_data(self):
        self.upstream_commit("Add a feature")
        c = updater.check(force=True)
        self.assertEqual((c["state"], c["behind"]), ("available", 1))
        self.assertEqual(c["commits"][0]["message"], "Add a feature")
        r = updater.apply()
        self.assertTrue(r["changed"] and r["restart"] and r["requirements_changed"])
        self.assertTrue((self.root / "reel/new.py").exists())
        self.assertFalse((self.root / "old.txt").exists())
        self.assertEqual((self.root / "data/models/m/keep.bin").read_text(), "weights")
        self.assertEqual(updater.check(force=True)["state"], "up_to_date")

    def test_local_changes_block_the_update_with_a_clear_message(self):
        self.upstream_commit()
        write(self.root, "start.bat", "my edit")
        self.assertIn("start.bat", updater.check(force=True)["dirty_files"])
        with self.assertRaisesRegex(updater.UpdateError, "local changes"):
            updater.apply()
        self.assertEqual((self.root / "start.bat").read_text(), "my edit")  # untouched

    def test_diverged_history_is_refused(self):
        self.upstream_commit()
        write(self.root, "mine.txt", "x"); git(self.root, "add", "-A"); git(self.root, "commit", "-qm", "local")
        self.assertEqual(updater.check(force=True)["state"], "diverged")
        with self.assertRaises(updater.UpdateError):
            updater.apply()

    def test_unreachable_remote_gives_a_friendly_error(self):
        git(self.root, "remote", "set-url", "origin", str(self.tmp / "nope.git"))
        c = updater.check(force=True)
        self.assertEqual(c["state"], "error")
        self.assertIn("Git could not reach GitHub", c["error"])


# ---------------------------------------------------------------- ZIP mode with a fake GitHub
OLD, NEW = "a" * 40, "b" * 40


def make_zip(files: dict[str, str], top="o-r-bbbbbbb") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, text in files.items():
            z.writestr(f"{top}/{name}" if top else name, text)
    return buf.getvalue()


class FakeGitHub:
    def __init__(self, zip_bytes: bytes):
        outer = self
        self.zip_bytes, self.auth_headers = zip_bytes, []

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def reply(self, code, body, ctype="application/json"):
                self.send_response(code); self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

            def do_GET(self):
                outer.auth_headers.append(self.headers.get("Authorization"))
                commit = {"sha": NEW, "commit": {"message": "Add updater\n\nlong body", "committer": {"date": "2026-10-30T01:00:00Z"}}}
                if self.path.startswith("/repos/o/r/commits/main"):
                    self.reply(200, json.dumps(commit).encode())
                elif self.path.startswith(f"/repos/o/r/compare/{OLD}...{NEW}"):
                    self.reply(200, json.dumps({"status": "ahead", "ahead_by": 1, "behind_by": 0, "commits": [commit]}).encode())
                elif self.path.startswith(f"/repos/o/r/zipball/{NEW}"):
                    self.reply(200, outer.zip_bytes, "application/zip")
                else:
                    self.reply(404, b"{}")

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


GOOD = {"reel/__init__.py": "", "start.bat": "new bat", "requirements.txt": "fastapi\nuvicorn\n", "reel/feature.py": "ok = True",
        "data/should_be_ignored.txt": "upstream data dir must never be written"}


class ZipMode(Patched):
    def start_fake(self, zip_bytes):
        self.gh = FakeGitHub(zip_bytes)
        updater.API, updater.REPO = self.gh.url, "o/r"
        self.addCleanup(self.gh.close)

    def installed_copy(self, version_line=f"{OLD} 2026-10-01"):
        write(self.root, "reel/__init__.py", "old")
        write(self.root, "start.bat", "old bat")
        write(self.root, "gone_upstream.txt", "stale")
        write(self.root, "data/models/m/keep.bin", "weights")
        write(self.root, "data/installed-files.json", json.dumps(["reel/__init__.py", "start.bat", "gone_upstream.txt"]))
        if version_line:
            write(self.root, "reel/VERSION", version_line + "\n")

    def test_reads_the_version_that_git_archive_expanded(self):
        self.installed_copy()
        v = updater.local_version()
        self.assertEqual((v["sha"], v["date"], v["source"]), (OLD, "2026-10-01", "zip"))

    def test_unexpanded_version_marker_means_unknown(self):
        self.installed_copy("$Format:%H %cs$")
        self.assertEqual(updater.local_version()["source"], "unknown")

    def test_check_lists_whats_new(self):
        self.installed_copy(); self.start_fake(make_zip(GOOD))
        c = updater.check(force=True)
        self.assertEqual((c["mode"], c["state"], c["behind"]), ("zip", "available", 1))
        self.assertEqual(c["commits"][0]["message"], "Add updater")

    def test_update_overlays_files_removes_stale_ones_and_protects_data(self):
        self.installed_copy(); self.start_fake(make_zip(GOOD))
        r = updater.apply()
        self.assertTrue(r["changed"] and r["requirements_changed"])
        self.assertEqual((self.root / "start.bat").read_text(), "new bat")
        self.assertEqual((self.root / "reel/feature.py").read_text(), "ok = True")
        self.assertFalse((self.root / "gone_upstream.txt").exists())                       # removed upstream
        self.assertEqual((self.root / "data/models/m/keep.bin").read_text(), "weights")    # user data intact
        self.assertFalse((self.root / "data/should_be_ignored.txt").exists())              # data/ never written
        self.assertEqual(json.loads((self.root / "data/installed-version.json").read_text())["sha"], NEW)
        self.assertEqual(updater.check(force=True)["state"], "up_to_date")

    def test_unknown_local_version_still_offers_an_update(self):
        self.installed_copy(None); self.start_fake(make_zip(GOOD))
        self.assertEqual(updater.check(force=True)["state"], "unknown")
        self.assertTrue(updater.apply()["changed"])
        self.assertEqual(updater.local_version()["sha"], NEW)  # tracked from now on

    def test_unsafe_paths_are_rejected_and_nothing_is_written(self):
        self.installed_copy(); self.start_fake(make_zip({**GOOD, "../../evil.txt": "pwn"}))
        with self.assertRaisesRegex(updater.UpdateError, "unsafe"):
            updater.apply()
        self.assertEqual((self.root / "start.bat").read_text(), "old bat")
        self.assertFalse((self.tmp / "evil.txt").exists())

    def test_a_zip_that_is_not_reel_is_rejected(self):
        self.installed_copy(); self.start_fake(make_zip({"readme.md": "hi"}))
        with self.assertRaisesRegex(updater.UpdateError, "does not look like Reel"):
            updater.apply()
        self.assertEqual((self.root / "start.bat").read_text(), "old bat")

    def test_saved_token_is_sent_to_github(self):
        self.installed_copy(); self.start_fake(make_zip(GOOD))
        write(self.root, "data/settings.json", json.dumps({"github_token": "tok123"}))
        updater.check(force=True)
        self.assertIn("Bearer tok123", self.gh.auth_headers)

    def test_private_repo_error_explains_the_token(self):
        self.installed_copy(); self.start_fake(make_zip(GOOD))
        updater.REPO = "o/private"
        c = updater.check(force=True)
        self.assertEqual(c["state"], "error")
        self.assertIn("access token", c["error"])


# ---------------------------------------------------------------- restart contract with start.bat
@unittest.skipUnless(shutil.which("python3") or sys.executable, "no python")
class RestartExitCode(unittest.TestCase):
    def test_restart_endpoint_exits_with_75_when_launched_by_start_bat(self):
        try:
            import fastapi, uvicorn  # noqa: F401
        except ImportError:
            self.skipTest("fastapi/uvicorn not installed")
        import socket
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
        env = {**os.environ, "REEL_NO_BROWSER": "1", "REEL_PORT": str(port), "REEL_LAUNCHER": "bat"}
        proc = subprocess.Popen([sys.executable, "-m", "reel"], cwd=Path(__file__).resolve().parent.parent, env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(60):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1); break
                except OSError:
                    import time; time.sleep(0.3)
            req = urllib.request.Request(f"http://127.0.0.1:{port}/api/update/restart", method="POST", data=b"{}",
                                         headers={"Content-Type": "application/json"})
            self.assertTrue(json.load(urllib.request.urlopen(req, timeout=5))["restarting"])
            self.assertEqual(proc.wait(timeout=10), 75)
        finally:
            if proc.poll() is None:
                proc.kill()


if __name__ == "__main__":
    unittest.main()
