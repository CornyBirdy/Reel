"""The page the app serves must never mix old and new static files (a browser keeps its cache across updates).

    python -m unittest discover tests
"""
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reel import app, config  # noqa: E402


class Index(unittest.TestCase):
    def test_every_script_and_stylesheet_is_versioned(self):
        html = app.render_index()
        v = app.static_version()
        self.assertIn(f'<script type="module" src="/static/js/app.js?v={v}"></script>', html)
        for css in re.findall(r'<link rel="stylesheet" href="([^"]+)"', html):
            self.assertTrue(css.endswith(f"?v={v}"), css)
        imap = json.loads(re.search(r'<script type="importmap">(.*?)</script>', html, re.S).group(1))["imports"]
        self.assertEqual(imap["/static/js/ui.js"], f"/static/js/ui.js?v={v}")           # reached by relative imports
        self.assertEqual(imap["/static/js/views/library.js"], f"/static/js/views/library.js?v={v}")   # reached by import()
        self.assertLess(html.index('type="importmap"'), html.index('type="module"'))     # an import map must come first

    def test_the_version_changes_when_a_file_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "js").mkdir()
            (root / "index.html").write_text("<html><head></head></html>")
            (root / "js" / "a.js").write_text("1")
            with mock.patch.object(config, "STATIC_DIR", root):
                before = app.static_version()
                (root / "js" / "a.js").write_text("22")
                self.assertNotEqual(before, app.static_version())


if __name__ == "__main__":
    unittest.main()
