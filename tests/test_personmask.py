"""Following one person through a video (adapters/personmask.py). Needs numpy; skipped where it is missing."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "adapters"))
try:
    import numpy as np
    import personmask
except ImportError:      # numpy lives in the model environments, not necessarily where the tests run
    np = None


def box(x0, y0, x1, y1, size=(40, 60)):
    m = np.zeros(size, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


@unittest.skipIf(np is None, "numpy is not installed")
class Track(unittest.TestCase):
    def test_biggest_person_without_a_click(self):
        small, big = box(2, 2, 8, 12), box(20, 5, 50, 40)
        first = personmask.pick_first([small, big], None)
        self.assertTrue((first == big).all())

    def test_click_chooses_the_person_under_it(self):
        small, big = box(2, 2, 8, 12), box(20, 5, 50, 40)
        first = personmask.pick_first([small, big], (5 / 60, 5 / 40))
        self.assertTrue((first == small).all())

    def test_click_that_misses_takes_the_nearest(self):
        a, b = box(2, 2, 8, 12), box(45, 25, 55, 38)
        first = personmask.pick_first([a, b], (0.9, 0.8))
        self.assertTrue((first == b).all())

    def test_follows_the_same_person_across_frames(self):
        # two people; the one clicked moves right a little each frame, the other stays put
        frames = [[box(5 + i, 5, 15 + i, 30), box(40, 5, 55, 35)] for i in range(5)]
        masks = personmask.track(frames, (8 / 60, 10 / 40), grow=0)
        for i, m in enumerate(masks):
            self.assertTrue((m == box(5 + i, 5, 15 + i, 30)).all(), i)

    def test_gap_keeps_the_last_mask_and_late_arrival_starts_empty(self):
        person = box(5, 5, 15, 30)
        masks = personmask.track([[], [], [person], [], [person]], None, grow=0)
        self.assertEqual([int(m.sum()) for m in masks[:2]], [0, 0])
        self.assertTrue((masks[3] == person).all())

    def test_nobody_found_says_so(self):
        with self.assertRaises(LookupError):
            personmask.track([[], []], None)

    def test_dilate_grows_without_wrapping(self):
        m = box(0, 0, 3, 3)
        d = personmask.dilate(m, 2)
        self.assertTrue(d[4, 4])
        self.assertFalse(d[-1, -1])           # did not wrap around to the far corner
        self.assertEqual(int(personmask.dilate(m, 0).sum()), int(m.sum()))


if __name__ == "__main__":
    unittest.main()
