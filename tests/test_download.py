"""Downloading weights: a stopped download continues instead of starting again (no network, no GPU).

    python -m unittest discover tests
"""
import http.server
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "adapters"))
import download  # noqa: E402

DATA = os.urandom(300_000)


class Server:
    """A file server with Range support that records every request and can be told to misbehave."""

    def __init__(self, data=DATA, cut_first_at=None, ignore_range=False, status=None, redirect_to=None):
        self.data, self.cut_first_at, self.ignore_range, self.status, self.redirect_to = data, cut_first_at, ignore_range, status, redirect_to
        self.requests: list[dict] = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_GET(self):
                outer.requests.append({"range": self.headers.get("Range"), "auth": self.headers.get("Authorization"), "path": self.path})
                if outer.status:
                    self.send_response(outer.status)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if outer.redirect_to:
                    self.send_response(302)
                    self.send_header("Location", outer.redirect_to)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body, start = outer.data, 0
                rng = self.headers.get("Range")
                if rng and not outer.ignore_range:
                    start = int(rng.split("=")[1].rstrip("-"))
                    if start >= len(body):
                        self.send_response(416)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
                else:
                    self.send_response(200)
                chunk = body[start:]
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                if outer.cut_first_at is not None and len(outer.requests) == 1:
                    self.wfile.write(chunk[:outer.cut_first_at])
                    self.wfile.flush()
                    self.close_connection = True     # the connection dies half way
                    return
                self.wfile.write(chunk)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/file.bin"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class Fetching(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.dest, self.part = root / "file.bin", root / ".cache" / "reel" / "file.bin.part"

    def serve(self, **kw):
        server = Server(**kw)
        self.addCleanup(server.close)
        return server

    def fetch(self, server, size=len(DATA), token=None, **kw):
        download.fetch_file(server.url, self.dest, self.part, size, token, threading.Event(), sleep=lambda s: None, **kw)

    def test_a_download_is_saved_and_leaves_no_partial_file(self):
        s = self.serve()
        self.fetch(s)
        self.assertEqual(self.dest.read_bytes(), DATA)
        self.assertFalse(self.part.exists())

    def test_a_dropped_connection_continues_where_it_stopped(self):
        s = self.serve(cut_first_at=120_000)
        self.fetch(s)
        self.assertEqual(self.dest.read_bytes(), DATA)
        self.assertEqual(len(s.requests), 2)
        self.assertIsNone(s.requests[0]["range"])
        self.assertEqual(s.requests[1]["range"], "bytes=120000-")          # not asked for again from the start

    def test_a_partial_file_from_an_earlier_run_is_continued(self):
        self.part.parent.mkdir(parents=True)
        self.part.write_bytes(DATA[:200_000])
        s = self.serve()
        self.fetch(s)
        self.assertEqual(self.dest.read_bytes(), DATA)
        self.assertEqual([r["range"] for r in s.requests], ["bytes=200000-"])

    def test_a_server_that_ignores_the_range_gets_the_file_from_the_start(self):
        self.part.parent.mkdir(parents=True)
        self.part.write_bytes(DATA[:50_000])
        s = self.serve(ignore_range=True)
        self.fetch(s)
        self.assertEqual(self.dest.read_bytes(), DATA)

    def test_a_partial_file_that_is_too_big_starts_again(self):
        self.part.parent.mkdir(parents=True)
        self.part.write_bytes(DATA + b"extra")
        s = self.serve()
        self.fetch(s)
        self.assertEqual(self.dest.read_bytes(), DATA)

    def test_a_finished_file_is_not_downloaded_again(self):
        self.dest.write_bytes(DATA)
        s = self.serve()
        self.fetch(s)
        self.assertEqual(s.requests, [])

    def test_a_file_of_the_wrong_size_is_replaced(self):
        self.dest.write_bytes(DATA[:10])
        s = self.serve()
        self.fetch(s)
        self.assertEqual(self.dest.read_bytes(), DATA)

    def test_the_token_goes_to_the_hub_but_not_to_the_cdn_it_redirects_to(self):
        cdn = self.serve()
        hub = self.serve(redirect_to=cdn.url)
        self.fetch(hub, token="hf_secret")
        self.assertEqual(self.dest.read_bytes(), DATA)
        self.assertEqual(hub.requests[0]["auth"], "Bearer hf_secret")
        self.assertIsNone(cdn.requests[0]["auth"])

    def test_a_refused_download_says_why_and_is_not_retried(self):
        s = self.serve(status=403)
        with self.assertRaisesRegex(download.Fatal, "gated"):
            self.fetch(s)
        self.assertEqual(len(s.requests), 1)

    def test_a_server_that_keeps_failing_is_given_up_on(self):
        s = self.serve(status=503)
        with self.assertRaisesRegex(download.Fatal, "Press Install again"):
            self.fetch(s)
        self.assertEqual(len(s.requests), download.MAX_FAILURES)

    def test_a_connection_that_always_dies_early_ends_in_an_error_not_a_loop(self):
        s = self.serve(data=DATA[:100_000])          # the server has less than the Hub said the file has
        with self.assertRaises(download.Fatal):
            self.fetch(s, size=len(DATA))


class MovingIntoPlace(unittest.TestCase):
    """Windows refuses the rename while a virus scanner (or anything else) has the finished file open."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.part, self.dest = Path(self.tmp.name) / "a.part", Path(self.tmp.name) / "a.bin"
        self.part.write_bytes(b"done")

    def test_it_waits_while_the_file_is_in_use_and_then_succeeds(self):
        real, calls, sleeps = os.replace, [], []

        def flaky(src, dst):
            calls.append(1)
            if len(calls) < 4:
                raise PermissionError(32, "being used by another process")
            real(src, dst)

        with mock.patch.object(download.os, "replace", flaky):
            download.move_into_place(self.part, self.dest, threading.Event(), sleeps.append)
        self.assertEqual(self.dest.read_bytes(), b"done")
        self.assertEqual(len(sleeps), 3)

    def test_a_file_that_stays_locked_is_copied_instead(self):
        real, calls = os.replace, []

        def locked(src, dst):
            calls.append(src)
            if Path(src) == self.part:
                raise PermissionError(32, "being used by another process")
            real(src, dst)

        with mock.patch.object(download.os, "replace", locked):
            download.move_into_place(self.part, self.dest, threading.Event(), lambda s: None)
        self.assertEqual(self.dest.read_bytes(), b"done")
        self.assertEqual(len(calls), download.REPLACE_ATTEMPTS + 1)       # tried the rename, then renamed the copy
        self.assertFalse(Path(str(self.dest) + ".copying").exists())

    def test_a_file_that_cannot_even_be_read_keeps_its_partial_and_says_what_to_do(self):
        def locked(src, dst):
            raise PermissionError(32, "being used by another process")

        with mock.patch.object(download.os, "replace", locked), \
                mock.patch.object(download.shutil, "copyfile", side_effect=PermissionError(32, "in use")):
            with self.assertRaisesRegex(download.Fatal, "press Install"):
                download.move_into_place(self.part, self.dest, threading.Event(), lambda s: None)
        self.assertTrue(self.part.exists())
        self.assertFalse(self.dest.exists())

    def test_the_next_install_only_renames_a_complete_partial_file(self):
        server = Server()
        self.addCleanup(server.close)
        self.part.write_bytes(DATA)
        download.fetch_file(server.url, self.dest, self.part, len(DATA), None, threading.Event(), sleep=lambda s: None)
        self.assertEqual(self.dest.read_bytes(), DATA)
        self.assertEqual(server.requests, [])


class Partials(unittest.TestCase):
    def test_the_hub_clients_leftovers_go_but_reels_partial_files_stay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / ".cache" / "huggingface" / "download" / "a.safetensors.abc.incomplete"
            mine = root / ".cache" / "reel" / "a.safetensors.part"
            for p in (old, mine):
                p.parent.mkdir(parents=True)
                p.write_bytes(b"x" * 10)
            self.assertEqual(download.drop_partials(root), 1)
            self.assertFalse(old.exists())
            self.assertTrue(mine.exists())
            self.assertEqual(download.bytes_on_disk(root), 10)       # and its bytes count towards the progress bar


if __name__ == "__main__":
    unittest.main()


class ThrottleTests(unittest.TestCase):
    """The speed limit: read from a file the app rewrites, shared by every thread (no network, no sleeping)."""

    def make(self, cfg=None):
        self.clock = [0.0]
        self.slept: list[float] = []

        def sleep(s):
            self.slept.append(s)
            self.clock[0] += s

        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        path = Path(d.name) / "t.json"
        if cfg is not None:
            import json as _json
            path.write_text(_json.dumps(cfg))
        return download.Throttle(path, clock=lambda: self.clock[0], sleep=sleep)

    def test_no_file_means_no_limit(self):
        t = self.make()
        t.wait(10_000_000)
        self.assertEqual(self.slept, [])

    def test_manual_limit_paces_the_writes(self):
        t = self.make({"mode": "manual", "mb_per_s": 2})
        for _ in range(4):
            t.wait(1_000_000)                      # 4 MB at 2 MB/s: the last block may go at 1.5 s
        self.assertAlmostEqual(sum(self.slept), 1.5, places=1)

    def test_auto_is_unlimited_until_the_pc_is_busy(self):
        t = self.make({"mode": "auto", "mb_per_s": 0, "busy": False, "busy_mb_per_s": 10})
        self.assertIsNone(t.limit())
        t = self.make({"mode": "auto", "mb_per_s": 0, "busy": True, "busy_mb_per_s": 10})
        self.assertEqual(t.limit(), 10_000_000)

    def test_auto_ceiling_applies_even_when_idle(self):
        t = self.make({"mode": "auto", "mb_per_s": 50, "busy": False, "busy_mb_per_s": 10})
        self.assertEqual(t.limit(), 50_000_000)
        t = self.make({"mode": "auto", "mb_per_s": 5, "busy": True, "busy_mb_per_s": 10})
        self.assertEqual(t.limit(), 5_000_000)

    def test_off_ignores_the_numbers(self):
        self.assertIsNone(self.make({"mode": "off", "mb_per_s": 5, "busy": True}).limit())

    def test_a_change_is_picked_up_while_running(self):
        t = self.make({"mode": "manual", "mb_per_s": 1})
        self.assertEqual(t.limit(), 1_000_000)
        t.path.write_text('{"mode": "off"}')
        self.clock[0] += 2
        self.assertIsNone(t.limit())

    def test_a_cancelled_download_stops_waiting(self):
        t = self.make({"mode": "manual", "mb_per_s": 1})
        t.wait(1_000_000)
        abort = threading.Event()
        abort.set()
        with self.assertRaises(download.Aborted):
            t.wait(5_000_000, abort)

    def test_a_throttled_download_still_arrives_whole(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        dest = Path(d.name) / "f.bin"
        part = Path(d.name) / ".cache" / "f.part"
        slept = []
        t = download.Throttle(None)
        t.limit = lambda: 100_000_000.0
        t.sleep = slept.append
        s = Server()
        self.addCleanup(s.close)
        download.fetch_file(s.url, dest, part, len(DATA), None, threading.Event(), throttle=t)
        self.assertEqual(dest.read_bytes(), DATA)
