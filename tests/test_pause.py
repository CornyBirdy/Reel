"""Pausing a download / install and a generation (no GPU).

    python -m unittest discover tests
"""
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reel import procs  # noqa: E402
from reel.installer import InstallError, InstallManager, Job  # noqa: E402

COUNTER = "import time\nfor i in range(10000):\n    print(i, flush=True)\n    time.sleep(0.02)\n"


def state_of(pid: int) -> str:
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]


@unittest.skipIf(sys.platform == "win32", "uses /proc and SIGSTOP; the Windows path is NtSuspendProcess")
class Freezing(unittest.TestCase):
    def start(self):
        proc = procs.popen([sys.executable, "-c", COUNTER])
        self.addCleanup(procs.kill_tree, proc)
        return proc

    def test_a_frozen_process_stops_printing_and_carries_on_when_resumed(self):
        proc = self.start()
        import threading
        seen = []
        threading.Thread(target=lambda: [seen.append(l) for l in proc.stdout], daemon=True).start()
        time.sleep(0.4)
        self.assertGreater(len(seen), 3)
        self.assertTrue(procs.suspend_tree(proc))
        time.sleep(0.2)
        n = len(seen)
        time.sleep(0.6)
        self.assertEqual(len(seen), n)                       # nothing while frozen
        self.assertEqual(state_of(proc.pid), "T")
        self.assertTrue(procs.resume_tree(proc))
        time.sleep(0.5)
        self.assertGreater(len(seen), n + 3)                 # and it carries on, nothing lost

    def test_suspending_something_that_already_ended_says_so(self):
        proc = procs.popen([sys.executable, "-c", "pass"])
        proc.wait()
        self.assertFalse(procs.suspend_tree(proc))

    def test_a_frozen_process_can_still_be_killed_quickly(self):
        proc = self.start()
        procs.suspend_tree(proc)
        t = time.time()
        procs.kill_tree(proc)
        self.assertLess(time.time() - t, 3)
        self.assertIsNotNone(proc.poll())


@unittest.skipIf(sys.platform == "win32", "uses /proc and SIGSTOP")
class InstallPause(unittest.TestCase):
    def setUp(self):
        self.mgr = InstallManager()
        self.job = Job("some-model")
        self.mgr._jobs["some-model"] = self.job
        self.job.proc = procs.popen([sys.executable, "-c", COUNTER])
        self.addCleanup(procs.kill_tree, self.job.proc)

    def test_pause_freezes_the_current_step_and_resume_thaws_it(self):
        self.mgr.pause("some-model")
        self.assertTrue(self.job.paused)
        self.assertTrue(self.job.to_dict()["paused"])
        self.assertEqual(self.job.to_dict()["speed"], 0.0)
        self.assertEqual(state_of(self.job.proc.pid), "T")
        self.assertTrue(self.mgr.is_installing("some-model"))   # still the active install: nothing else can start meanwhile
        self.mgr.resume("some-model")
        self.assertFalse(self.job.paused)
        time.sleep(0.2)
        self.assertNotEqual(state_of(self.job.proc.pid), "T")

    def test_cancel_works_on_a_paused_install(self):
        self.mgr.pause("some-model")
        t = time.time()
        self.mgr.cancel("some-model")
        self.assertLess(time.time() - t, 3)
        self.assertTrue(self.job.cancel.is_set())
        self.assertFalse(self.job.paused)

    def test_the_next_step_of_a_paused_install_starts_frozen(self):
        self.job.proc = None                                    # between two steps
        self.mgr.pause("some-model")
        self.assertTrue(self.job.paused)
        proc = procs.popen([sys.executable, "-c", COUNTER])
        self.addCleanup(procs.kill_tree, proc)
        if self.job.paused:                                     # what _run_cmd does after starting a step
            procs.suspend_tree(proc)
        time.sleep(0.3)
        self.assertEqual(state_of(proc.pid), "T")

    def test_pausing_something_that_is_not_installing_is_refused(self):
        with self.assertRaises(InstallError):
            self.mgr.pause("nothing-here")
        with self.assertRaises(InstallError):
            self.mgr.resume("some-model")                       # not paused


if __name__ == "__main__":
    unittest.main()
