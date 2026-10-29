"""Update Reel in place from GitHub.

Two mechanisms, chosen automatically:
  git  - this folder is a git clone and git is installed: `git fetch` + fast-forward merge.
  zip  - this folder came from "Download ZIP": download the branch snapshot from GitHub and
         overlay it. A manifest of installed files lets files that were deleted upstream
         be removed again.
Neither touches data/ (models, history, settings, outputs) or .venv/.

Environment overrides (mainly for tests): REEL_UPDATE_REPO, REEL_UPDATE_BRANCH, REEL_GITHUB_API.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from . import config

REPO = os.environ.get("REEL_UPDATE_REPO", "CornyBirdy/Reel")
BRANCH = os.environ.get("REEL_UPDATE_BRANCH", "main")
API = os.environ.get("REEL_GITHUB_API", "https://api.github.com").rstrip("/")
ROOT = config.ROOT
PROTECTED = ("data/", ".venv/", ".git/")      # never overwritten or deleted by an update
MAX_ZIP_BYTES = 80 * 1024 * 1024
CACHE_SECONDS = 30 * 60
SHA_RE = re.compile(r"^[0-9a-f]{40}$")

_lock = threading.Lock()
_cache: dict = {"at": 0.0, "result": None}


class UpdateError(Exception):
    """A problem worth showing to the user as-is."""


# --------------------------------------------------------------------------- helpers
def _git_ok() -> bool:
    return (ROOT / ".git").exists() and shutil.which("git") is not None


def _git(*args: str, timeout: int = 90, strip: bool = True) -> str:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"}
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout, env=env,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise UpdateError(f"git did not run: {exc}") from exc
    if out.returncode != 0:
        raise UpdateError((out.stderr or out.stdout).strip() or f"git {args[0]} failed")
    return out.stdout.strip() if strip else out.stdout


def mode() -> str:
    return "git" if _git_ok() else "zip"


def _short(sha: str | None) -> str | None:
    return sha[:7] if sha else None


def _headers() -> dict:
    h = {"Accept": "application/vnd.github+json", "User-Agent": "reel-updater", "X-GitHub-Api-Version": "2022-11-28"}
    token = config.load_settings().get("github_token")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


class _DropAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlparse(newurl).netloc != urllib.parse.urlparse(req.full_url).netloc:
            new.headers.pop("Authorization", None)
        return new


_opener = urllib.request.build_opener(_DropAuthOnRedirect)


def _open(url: str, timeout: int = 30):
    try:
        return _opener.open(urllib.request.Request(url, headers=_headers()), timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403, 404):
            has_token = bool(config.load_settings().get("github_token"))
            raise UpdateError(
                "GitHub would not show the Reel repository" + (" with the saved token." if has_token else
                ". If the repository is private, save a GitHub access token below, or make the repository public.")
                + (" (rate limit reached; try again later)" if exc.code == 403 and not has_token else "")) from exc
        raise UpdateError(f"GitHub returned HTTP {exc.code}.") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise UpdateError(f"Could not reach GitHub: {getattr(exc, 'reason', exc)}") from exc


def _api_json(path: str):
    with _open(f"{API}{path}") as r:
        return json.load(r)


# --------------------------------------------------------------------------- local version
def local_version() -> dict:
    """{sha, short, date, source} - source says how we know. sha is None if it cannot be determined."""
    if _git_ok():
        try:
            sha = _git("rev-parse", "HEAD")
            date = _git("log", "-1", "--format=%cs")
            return {"sha": sha, "short": _short(sha), "date": date, "source": "git"}
        except UpdateError:
            pass
    rec = config.DATA_DIR / "installed-version.json"          # written by the ZIP updater
    try:
        d = json.loads(rec.read_text(encoding="utf-8"))
        if SHA_RE.match(d.get("sha", "")):
            return {"sha": d["sha"], "short": _short(d["sha"]), "date": d.get("date"), "source": "update"}
    except (OSError, json.JSONDecodeError):
        pass
    try:                                                         # expanded by `git archive` in GitHub's ZIP
        first, _, date = (ROOT / "reel" / "VERSION").read_text(encoding="utf-8").strip().partition(" ")
        if SHA_RE.match(first):
            return {"sha": first, "short": _short(first), "date": date or None, "source": "zip"}
    except OSError:
        pass
    return {"sha": None, "short": None, "date": None, "source": "unknown"}


# --------------------------------------------------------------------------- check
def _check_git() -> dict:
    try:
        _git("fetch", "--quiet", "origin", BRANCH, timeout=120)
    except UpdateError as exc:
        raise UpdateError(f"Git could not reach GitHub: {exc}. If the repository is private, sign in with Git "
                          "Credential Manager (or run `git fetch` once in the Reel folder).") from exc
    remote = _git("rev-parse", "FETCH_HEAD")
    local = _git("rev-parse", "HEAD")
    behind = int(_git("rev-list", "--count", "HEAD..FETCH_HEAD"))
    ahead = int(_git("rev-list", "--count", "FETCH_HEAD..HEAD"))
    commits = []
    for line in _git("log", "--format=%H%x09%s%x09%cs", "-n", "30", "HEAD..FETCH_HEAD").splitlines():
        sha, subject, date = line.split("\t")
        commits.append({"sha": _short(sha), "message": subject, "date": date})
    dirty = [l[3:] for l in _git("status", "--porcelain", "--untracked-files=no", strip=False).splitlines() if l.strip()]
    remote_info = {"sha": remote, "short": _short(remote), "date": _git("log", "-1", "--format=%cs", remote),
                   "message": _git("log", "-1", "--format=%s", remote)}
    if behind and ahead:
        state = "diverged"
    elif behind:
        state = "available"
    elif ahead:
        state = "ahead"
    else:
        state = "up_to_date"
    return {"state": state, "behind": behind, "ahead": ahead, "commits": commits, "remote": remote_info, "dirty_files": dirty}


def _check_zip() -> dict:
    local = local_version()
    head = _api_json(f"/repos/{REPO}/commits/{urllib.parse.quote(BRANCH)}")
    remote = {"sha": head["sha"], "short": _short(head["sha"]), "message": head["commit"]["message"].splitlines()[0],
              "date": (head["commit"]["committer"]["date"] or "")[:10]}
    if local["sha"] == head["sha"]:
        return {"state": "up_to_date", "behind": 0, "ahead": 0, "commits": [], "remote": remote, "dirty_files": []}
    if not local["sha"]:
        return {"state": "unknown", "behind": None, "ahead": 0, "commits": [], "remote": remote, "dirty_files": []}
    try:
        cmp = _api_json(f"/repos/{REPO}/compare/{local['sha']}...{head['sha']}")
    except UpdateError:  # e.g. the installed commit no longer exists upstream
        return {"state": "unknown", "behind": None, "ahead": 0, "commits": [], "remote": remote, "dirty_files": []}
    commits = [{"sha": _short(c["sha"]), "message": c["commit"]["message"].splitlines()[0],
                "date": (c["commit"]["committer"]["date"] or "")[:10]} for c in reversed(cmp.get("commits", []))][:30]
    status = cmp.get("status")
    state = {"ahead": "available", "identical": "up_to_date", "behind": "ahead", "diverged": "diverged"}.get(status, "unknown")
    return {"state": state, "behind": cmp.get("ahead_by"), "ahead": cmp.get("behind_by", 0), "commits": commits,
            "remote": remote, "dirty_files": []}


def check(force: bool = False) -> dict:
    """Compare this install with GitHub. Cached for 30 minutes unless force."""
    with _lock:
        if not force and _cache["result"] and time.time() - _cache["at"] < CACHE_SECONDS:
            return _cache["result"]
    m = mode()
    result = {"mode": m, "repo": REPO, "branch": BRANCH, "local": local_version(), "error": None,
              "checked_at": time.time(), "can_restart": os.environ.get("REEL_LAUNCHER") == "bat"}
    try:
        result.update(_check_git() if m == "git" else _check_zip())
    except UpdateError as exc:
        result.update(state="error", error=str(exc), behind=None, ahead=0, commits=[], remote=None, dirty_files=[])
    with _lock:
        _cache.update(at=time.time(), result=result)
    return result


def _invalidate() -> None:
    with _lock:
        _cache.update(at=0.0, result=None)


# --------------------------------------------------------------------------- apply
def apply() -> dict:
    status = check(force=True)
    if status["state"] == "error":
        raise UpdateError(status["error"])
    if status["state"] == "up_to_date":
        return {"changed": False, "message": "Already up to date.", "restart": False}
    if status["state"] == "diverged":
        raise UpdateError("This copy has local commits that GitHub does not have, so it cannot be fast-forwarded. "
                          "Merge or reset it with git yourself.")
    if status["state"] == "ahead":
        raise UpdateError("This copy is ahead of GitHub; there is nothing to update.")
    result = _apply_git(status) if status["mode"] == "git" else _apply_zip(status)
    _invalidate()
    return result


def _apply_git(status: dict) -> dict:
    if status["dirty_files"]:
        raise UpdateError("You have local changes to tracked files (" + ", ".join(status["dirty_files"][:5]) +
                          "). Commit, stash or discard them first, then update.")
    before = _git("rev-parse", "HEAD")
    _git("merge", "--ff-only", "FETCH_HEAD")
    changed = _git("diff", "--name-only", before, "HEAD").splitlines()
    return {"changed": True, "files": len(changed), "restart": True, "new_version": _short(_git("rev-parse", "HEAD")),
            "requirements_changed": "requirements.txt" in changed,
            "message": f"Updated {len(changed)} file{'s' if len(changed) != 1 else ''}."}


def _safe_member(name: str) -> bool:
    parts = Path(name).parts
    return bool(parts) and not Path(name).is_absolute() and ".." not in parts and ":" not in name


def _apply_zip(status: dict) -> dict:
    sha = status["remote"]["sha"]
    tmp = Path(tempfile.mkdtemp(prefix="update-", dir=_tmp_root()))
    try:
        zpath = tmp / "reel.zip"
        size = 0
        with _open(f"{API}/repos/{REPO}/zipball/{sha}", timeout=120) as r, zpath.open("wb") as out:
            while chunk := r.read(1 << 20):
                size += len(chunk)
                if size > MAX_ZIP_BYTES:
                    raise UpdateError("The download is larger than expected; refusing to unpack it.")
                out.write(chunk)
        try:
            zf = zipfile.ZipFile(zpath)
        except zipfile.BadZipFile as exc:
            raise UpdateError("The downloaded update was not a valid ZIP file.") from exc
        with zf:
            names = [n for n in zf.namelist() if n and not n.endswith("/")]
            if not names or not all(_safe_member(n) for n in names):
                raise UpdateError("The update contains unsafe file paths; refusing to unpack it.")
            top = {Path(n).parts[0] for n in names}
            if len(top) != 1:
                raise UpdateError("The update has an unexpected layout.")
            prefix = top.pop() + "/"
            files = {n[len(prefix):]: n for n in names if n.startswith(prefix) and len(n) > len(prefix)}
            if "reel/__init__.py" not in files or "start.bat" not in files:
                raise UpdateError("The download does not look like Reel; refusing to unpack it.")
            written = []
            for rel, member in sorted(files.items()):
                if rel.startswith(PROTECTED):
                    continue
                dest = ROOT / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                staged = dest.with_name(dest.name + ".reel-new")
                with zf.open(member) as src, staged.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                os.replace(staged, dest)
                written.append(rel)
        removed = _remove_stale(written)
        (config.DATA_DIR / "installed-files.json").write_text(json.dumps(sorted(written)), encoding="utf-8")
        (config.DATA_DIR / "installed-version.json").write_text(json.dumps(
            {"sha": sha, "date": status["remote"].get("date"), "message": status["remote"].get("message"),
             "installed_at": time.time()}), encoding="utf-8")
        return {"changed": True, "files": len(written), "removed": removed, "restart": True,
                "new_version": _short(sha), "requirements_changed": "requirements.txt" in written,
                "message": f"Updated {len(written)} files."}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _tmp_root() -> Path:
    p = config.DATA_DIR / "tmp"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _remove_stale(new_files: list[str]) -> int:
    """Delete files that an earlier update installed but the new version no longer has."""
    manifest = config.DATA_DIR / "installed-files.json"
    try:
        old = set(json.loads(manifest.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return 0
    removed = 0
    for rel in old - set(new_files):
        if rel.startswith(PROTECTED) or not _safe_member(rel):
            continue
        try:
            (ROOT / rel).unlink()
            removed += 1
        except OSError:
            pass
    return removed
