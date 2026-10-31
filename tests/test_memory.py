"""Memory management: the offload plan, "auto" offload, what the estimate says about the card, and adding a missing package.

No GPU and no torch needed.

    python -m unittest discover tests
"""
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "adapters"))
import base  # noqa: E402
import memplan  # noqa: E402
from reel import estimate, installer, recipes, verify  # noqa: E402

VACE = {"transformer": 7.1, "vae": 0.5}      # the parts the VACE pipeline holds; the 11.4 GB text encoder is handled apart


class ChooseOffload(unittest.TestCase):
    def test_everything_fits_with_room_to_work(self):
        mode, why = memplan.choose_offload({"unet": 5.1, "text_encoder": 0.5, "text_encoder_2": 2.6, "vae": 0.3}, free_gb=14.7, total_gb=15.9)   # SDXL
        self.assertEqual(mode, "none")
        self.assertIn("keep everything on the GPU", why)

    def test_vace_without_its_text_encoder_stays_on_a_16gb_card(self):
        self.assertEqual(memplan.choose_offload(VACE, free_gb=14.7, total_gb=15.9)[0], "none")

    def test_the_same_model_with_a_browser_and_games_open_has_less_free(self):
        self.assertEqual(memplan.choose_offload(VACE, free_gb=11.0, total_gb=15.9)[0], "model")
        self.assertEqual(memplan.choose_offload(VACE, free_gb=9.0, total_gb=15.9)[0], "sequential")

    def test_whole_pipeline_of_wan_22_needs_one_part_at_a_time(self):
        mode, why = memplan.choose_offload({"transformer": 7.1, "text_encoder": 11.4, "vae": 0.5}, free_gb=14.7, total_gb=15.9)
        self.assertEqual(mode, "model")
        self.assertIn("text_encoder", why)

    def test_when_even_the_biggest_part_does_not_fit_it_goes_layer_by_layer(self):
        self.assertEqual(memplan.choose_offload({"transformer": 24.0, "vae": 0.5}, free_gb=14.7, total_gb=15.9)[0], "sequential")

    def test_headroom_grows_with_the_card_but_never_below_the_floor(self):
        self.assertEqual(memplan.headroom(8, memplan.NONE_HEADROOM), 3.0)
        self.assertAlmostEqual(memplan.headroom(24, memplan.NONE_HEADROOM), 7.2)

    def test_the_text_encoder_is_kept_in_ram_only_when_plenty_is_left(self):
        self.assertFalse(memplan.keep_in_ram(11.4, 15.0))       # 3.6 GB would be left
        self.assertTrue(memplan.keep_in_ram(11.4, 40.0))
        self.assertFalse(memplan.keep_in_ram(11.4, None))       # unknown RAM: be careful


class FakeTorch:
    """Just enough of torch for the planner: Module, and a card with this much memory."""
    class Tensor:
        def __init__(self, gb):
            self.gb = gb

        def numel(self):
            return int(self.gb * 1024**3)

        def element_size(self):
            return 1

    class Module:
        def __init__(self, gb):
            self.gb = gb

        def parameters(self):
            return [FakeTorch.Tensor(self.gb)]

        def buffers(self):
            return []

    def __init__(self, free_gb, total_gb):
        self.nn = types.SimpleNamespace(Module=FakeTorch.Module)
        self.cuda = types.SimpleNamespace(mem_get_info=lambda: (int(free_gb * 1024**3), int(total_gb * 1024**3)))


class AutoOffload(unittest.TestCase):
    def adapter(self, offload, free, total=15.9):
        a = base.BaseAdapter.__new__(base.BaseAdapter)
        a.torch, a.req, a.device = FakeTorch(free, total), {"offload": offload}, "cuda"
        a.offload_active = offload
        return a

    def pipe(self, **parts):
        return types.SimpleNamespace(components={**{n: FakeTorch.Module(g) for n, g in parts.items()}, "tokenizer": object(), "scheduler": object()})

    def test_auto_measures_the_loaded_parts_and_the_free_vram(self):
        a = self.adapter("auto", free=14.7)
        with mock.patch.object(base, "emit") as emit:
            self.assertEqual(a.resolve_offload(self.pipe(transformer=7.1, vae=0.5)), "none")
        self.assertEqual(a.offload_active, "none")
        self.assertIn("'auto' chose none", emit.call_args.kwargs["message"])

    def test_auto_follows_what_is_really_free(self):
        parts = dict(transformer=7.1, text_encoder=11.4, vae=0.5)
        with mock.patch.object(base, "emit"):
            self.assertEqual(self.adapter("auto", free=14.7).resolve_offload(self.pipe(**parts)), "model")
            self.assertEqual(self.adapter("auto", free=30.0, total=32.0).resolve_offload(self.pipe(**parts)), "none")
            self.assertEqual(self.adapter("auto", free=6.0).resolve_offload(self.pipe(**parts)), "sequential")

    def test_a_fixed_mode_is_left_alone(self):
        a = self.adapter("model", free=14.7)
        with mock.patch.object(base, "emit") as emit:
            self.assertEqual(a.resolve_offload(self.pipe(transformer=1.0)), "model")
        emit.assert_not_called()

    def test_sizes_ignore_things_that_are_not_modules(self):
        self.assertEqual(set(memplan.component_gb(self.pipe(transformer=2.0, vae=0.5), FakeTorch(10, 16))), {"transformer", "vae"})


class OfferAuto(unittest.TestCase):
    def test_every_model_that_can_offload_gets_auto_first(self):
        found, problems = recipes.load_all()
        self.assertFalse(problems)
        for r in found.values():
            options = (r.get("memory") or {}).get("offload_options")
            if options and "model" in options:
                self.assertEqual(options[0], "auto", r["id"])
                self.assertEqual(options.count("auto"), 1, r["id"])

    def test_it_is_not_added_twice(self):
        r = {"memory": {"offload_options": ["auto", "none", "model"]}}
        self.assertEqual(recipes.offer_auto_offload(r)["memory"]["offload_options"], ["auto", "none", "model"])

    def test_a_model_that_cannot_offload_is_left_alone(self):
        r = {"memory": {"offload_options": ["none"]}}
        self.assertEqual(recipes.offer_auto_offload(r)["memory"]["offload_options"], ["none"])


class VaceRecipe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recipe = recipes.load_all()[0]["wan-vace-1.3b"]

    def test_scipy_is_installed_and_verified(self):
        """Mask2Former (finding the person) cannot even be constructed without scipy."""
        self.assertIn("scipy", [installer.package_name(p) for p in self.recipe["python"]["packages"]])
        self.assertIn("scipy", self.recipe["verify_modules"])

    def test_the_text_encoder_is_estimated_as_transient(self):
        self.assertEqual(self.recipe["memory"]["estimate"]["transient"], ["text_encoder"])

    def test_vace_defaults_to_auto_offload(self):
        self.assertEqual(self.recipe["defaults"]["offload"], "auto")

    def test_a_720p_clip_fits_a_32gb_pc_with_room(self):
        e = estimate.estimate(self.recipe, {"width": 1280, "height": 720, "frames": 81, "offload": "auto", "quantization": "none", "vae_tiling": True},
                              {"total_gb": 15.9, "free_gb": 13.7}, {"total_gb": 31.9, "available_gb": 23.6})
        self.assertEqual(e["ram"], "ok", e)


class EstimateSaysWhereTheVramWent(unittest.TestCase):
    """'13.7 GB free of a 16 GB card': Windows and other programs hold the rest, and the estimate now says so."""

    def est(self, free, worker=None):
        r = recipes.load_all()[0]["wan-vace-1.3b"]
        return estimate.estimate(r, {"width": 832, "height": 480, "frames": 81, "offload": "model", "quantization": "none", "vae_tiling": True},
                                 {"total_gb": 15.9, "free_gb": free}, {"total_gb": 31.9, "available_gb": 23.6}, worker=worker)

    def test_other_programs_share_is_reported(self):
        e = self.est(13.7)
        self.assertEqual((e["vram_total_gb"], e["vram_cap_gb"], e["vram_other_gb"]), (15.9, 13.7, 2.2))

    def test_reels_own_loaded_model_is_not_blamed_on_other_programs(self):
        e = self.est(6.0, worker={"vram_gb": 7.7, "ram_gb": 1.0})
        self.assertEqual(e["vram_cap_gb"], 13.7)
        self.assertEqual(e["vram_other_gb"], 2.2)

    def test_without_a_gpu_there_is_nothing_to_report(self):
        r = recipes.load_all()[0]["wan-vace-1.3b"]
        e = estimate.estimate(r, {"width": 832, "height": 480, "frames": 81, "offload": "model", "quantization": "none", "vae_tiling": True}, None, None)
        self.assertIsNone(e["vram_other_gb"])


class PackageNames(unittest.TestCase):
    def test_version_pins_extras_and_markers_are_dropped(self):
        for spec, name in [("diffusers>=0.35", "diffusers"), ("imageio-ffmpeg", "imageio-ffmpeg"), ("torch==2.7.0+cu128", "torch"),
                           ("pywebview>=5.3; sys_platform == 'win32'", "pywebview"), ("transformers[sentencepiece]>=4", "transformers"),
                           ("numpy ", "numpy"), ("Pillow<12", "Pillow")]:
            self.assertEqual(installer.package_name(spec), name, spec)


@unittest.skipIf(sys.platform == "win32", "the test environment's python is a symlink")
class TopUp(unittest.TestCase):
    """An install from before the recipe listed scipy: the missing package is found and installed alone."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.mdir = Path(self.tmp.name)
        (self.mdir / "venv" / "bin").mkdir(parents=True)
        (self.mdir / "venv" / "bin" / "python").symlink_to(sys.executable)
        self.recipe = {"id": "m", "python": {"packages": ["unittest-xmlrunner-not-real>=1", "pip"]}}
        patch = mock.patch.object(installer, "model_dir", lambda _m: self.mdir)
        patch.start()
        self.addCleanup(patch.stop)

    def test_only_what_is_missing_is_reported(self):
        self.assertEqual(installer.missing_packages(self.recipe), ["unittest-xmlrunner-not-real"])

    def test_nothing_is_missing_in_a_complete_environment(self):
        self.assertEqual(installer.missing_packages({"id": "m", "python": {"packages": ["pip"]}}), [])

    def test_no_environment_means_nothing_to_ask(self):
        (self.mdir / "venv" / "bin" / "python").unlink()
        self.assertEqual(installer.missing_packages(self.recipe), [])

    def test_top_up_installs_just_the_missing_ones_and_keeps_torch(self):
        said, calls = [], []

        def run(cmd, **kw):
            calls.append(cmd)
            return types.SimpleNamespace(returncode=0, stdout="ok", stderr="")

        with mock.patch.object(installer, "missing_packages", return_value=["unittest-xmlrunner-not-real"]), \
                mock.patch.object(installer, "torch_constraints", return_value=["-c", "constraints.txt"]), \
                mock.patch.object(installer.subprocess, "run", side_effect=run):
            added = installer.top_up_packages(self.recipe, said.append)
        self.assertEqual(added, ["unittest-xmlrunner-not-real"])
        pip = calls[0]
        self.assertEqual(pip[-1], "unittest-xmlrunner-not-real>=1")      # the recipe's own spec, version and all
        self.assertNotIn("pip", pip[pip.index("install") + 1:])          # packages that are already there are not touched
        self.assertEqual(pip[pip.index("-c"):pip.index("-c") + 2], ["-c", "constraints.txt"])
        self.assertTrue(said and "unittest-xmlrunner-not-real" in said[0])

    def test_a_failing_pip_is_reported_not_swallowed(self):
        failed = types.SimpleNamespace(returncode=1, stdout="", stderr="ERROR: No matching distribution found")
        with mock.patch.object(installer, "missing_packages", return_value=["x"]), \
                mock.patch.object(installer, "torch_constraints", return_value=[]), \
                mock.patch.object(installer.subprocess, "run", return_value=failed):
            with self.assertRaises(installer.InstallError) as ctx:
                installer.top_up_packages({"id": "m", "python": {"packages": ["x"]}})
        self.assertIn("No matching distribution", str(ctx.exception))

    def test_nothing_to_do_runs_no_pip(self):
        with mock.patch.object(installer.subprocess, "run") as run, mock.patch.object(installer, "missing_packages", return_value=[]):
            self.assertEqual(installer.top_up_packages(self.recipe), [])
        run.assert_not_called()


@unittest.skipIf(sys.platform == "win32", "the test environment's python is a symlink")
class WarmUp(unittest.TestCase):
    """The first import after packages change can sit silent for minutes on Windows: it is done once, before the model process starts."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.mdir = Path(self.tmp.name)
        (self.mdir / "venv" / "bin").mkdir(parents=True)
        (self.mdir / "venv" / "bin" / "python").symlink_to(sys.executable)
        self.recipe = {"id": "m", "python": {"packages": ["pip"]}, "adapter_config": {}}
        for patch in (mock.patch.object(installer, "model_dir", lambda _m: self.mdir), mock.patch.object(installer, "_WARM", "print('imported')\n")):
            patch.start()
            self.addCleanup(patch.stop)

    def test_it_runs_once_and_says_why(self):
        said, lines = [], []
        self.assertTrue(installer.warm_up(self.recipe, said.append, lines.append))
        self.assertEqual(said, [installer.WARM_MESSAGE])
        self.assertIn("imported", lines)
        self.assertTrue((self.mdir / "warm.json").exists())
        with mock.patch.object(installer.procs, "popen") as popen:        # nothing changed: no second run
            self.assertTrue(installer.warm_up(self.recipe, said.append))
        popen.assert_not_called()

    def test_changing_a_package_runs_it_again(self):
        installer.warm_up(self.recipe)
        before = (self.mdir / "warm.json").read_text()
        with mock.patch.object(installer, "environment_signature", return_value="another state"):
            installer.warm_up(self.recipe)
        self.assertNotEqual((self.mdir / "warm.json").read_text(), before)

    def test_a_rebuilt_environment_is_not_trusted(self):
        installer.warm_up(self.recipe)
        (self.mdir / "venv" / "pyvenv.cfg").write_text("home = x")          # a new venv has a new pyvenv.cfg
        with mock.patch.object(installer.procs, "popen", wraps=installer.procs.popen) as popen:
            installer.warm_up(self.recipe)
        popen.assert_called_once()

    def test_a_failure_is_reported_and_not_remembered(self):
        with mock.patch.object(installer, "_WARM", "raise SystemExit('No module named torch')\n"):
            with self.assertRaises(installer.InstallError) as ctx:
                installer.warm_up(self.recipe)
        self.assertIn("No module named torch", str(ctx.exception))
        self.assertFalse((self.mdir / "warm.json").exists())

    def test_cancelling_stops_it_at_once(self):
        import threading
        import time
        cancel = threading.Event()
        threading.Timer(0.5, cancel.set).start()
        t0 = time.time()
        with mock.patch.object(installer, "_WARM", "import time\ntime.sleep(60)\n"):
            self.assertFalse(installer.warm_up(self.recipe, cancel=cancel))
        self.assertLess(time.time() - t0, 10)
        self.assertFalse((self.mdir / "warm.json").exists())

    def test_no_environment_means_nothing_to_do(self):
        (self.mdir / "venv" / "bin" / "python").unlink()
        self.assertTrue(installer.warm_up(self.recipe))


class PrepareEnvironment(unittest.TestCase):
    """What happens before a model process starts (reel/workers.py)."""

    def worker(self):
        from reel import workers
        w = types.SimpleNamespace(recipe={"id": "m"}, message="", log=types.SimpleNamespace(note=mock.Mock(), write=mock.Mock()))
        return workers, w

    def test_a_cancel_during_the_preparation_stops_the_load(self):
        workers, w = self.worker()
        with mock.patch.object(installer, "top_up_packages", return_value=[]), mock.patch.object(installer, "warm_up", return_value=False):
            self.assertFalse(workers.WorkerManager._prepare_environment(w))

    def test_trouble_is_logged_and_the_load_carries_on(self):
        workers, w = self.worker()
        events = []
        with mock.patch.object(installer, "top_up_packages", side_effect=installer.InstallError("offline")):
            self.assertTrue(workers.WorkerManager._prepare_environment(w, events.append))
        self.assertIn("offline", w.log.note.call_args.args[0])
        self.assertEqual(events[0]["event"], "log")

    def test_progress_reaches_the_status_chip(self):
        workers, w = self.worker()
        events = []

        def warm(recipe, say, line, cancel):
            say(installer.WARM_MESSAGE)
            return True

        with mock.patch.object(installer, "top_up_packages", return_value=["scipy"]), mock.patch.object(installer, "warm_up", side_effect=warm):
            self.assertTrue(workers.WorkerManager._prepare_environment(w, events.append))
        self.assertEqual(w.message, installer.WARM_MESSAGE)
        self.assertEqual(events[0], {"event": "phase", "phase": "loading", "message": installer.WARM_MESSAGE})
        self.assertIn("Installed scipy", [c.args[0] for c in w.log.note.call_args_list][0])


class RepairOnlyAddsPackages(unittest.TestCase):
    def job(self, env):
        j = verify.Job("m", False)
        j.environment = env
        return j

    def test_a_missing_package_does_not_rebuild_the_environment(self):
        j = self.job({"ok": False, "torch": "2.7.0+cu128", "cuda_available": True, "kernels_ok": True, "missing_modules": ["scipy"],
                      "error": "scipy import failed: No module named 'scipy'"})
        self.assertTrue(verify.only_packages_missing(j))
        self.assertFalse(verify.needs_new_environment(j))

    def test_a_cpu_only_torch_still_gets_a_new_environment(self):
        j = self.job({"ok": False, "torch": "2.7.0+cpu", "cuda_available": False, "missing_modules": ["scipy"], "error": "scipy import failed"})
        self.assertFalse(verify.only_packages_missing(j))
        self.assertTrue(verify.needs_new_environment(j))

    def test_a_broken_torch_gets_a_new_environment(self):
        j = self.job({"ok": False, "error": "torch import failed: DLL load failed"})
        self.assertTrue(verify.needs_new_environment(j))

    def test_a_healthy_environment_needs_nothing(self):
        j = self.job({"ok": True, "torch": "2.7.0+cu128", "cuda_available": True, "kernels_ok": True})
        self.assertFalse(verify.only_packages_missing(j))
        self.assertFalse(verify.needs_new_environment(j))


if __name__ == "__main__":
    unittest.main()
