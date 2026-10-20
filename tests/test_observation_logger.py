"""Exercise ATDog2 observation capture without launching Isaac Sim."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "reinforcement_learning" / "rsl_rl"))

from observation_logger import OBSERVATIONS, ObservationLogger


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def __getitem__(self, key):
        return FakeTensor(self.value[key])

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value.tolist()

    def item(self):
        return self.value.item()


class ObservationLoggerTests(unittest.TestCase):
    def test_capture_and_plot_all_terms(self):
        slices = {}
        offset = 0
        for name, _, labels in OBSERVATIONS:
            slices[name] = slice(offset, offset + len(labels))
            offset += len(labels)
        self.assertEqual(offset, 47)
        env = SimpleNamespace(
            num_envs=2,
            num_one_step_obs=47,
            policy_one_step_term_slices=slices,
            unwrapped=SimpleNamespace(step_dt=0.02, episode_length_buf=FakeTensor([3, 7])),
        )
        frame = np.arange(47, dtype=float)
        history = FakeTensor(np.stack((np.tile(frame, 6), np.zeros(282))))
        with tempfile.TemporaryDirectory() as directory:
            logger = ObservationLogger(env, directory, sample_every=2, plot_every=2)
            for _ in range(3):
                logger.record(history)
            logger.close()
            output_dir = Path(directory) / "observations"
            with (output_dir / "policy_observations.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(len(rows), 3)
            self.assertEqual([int(row[0]) for row in rows[1:]], [0, 2])
            self.assertEqual([float(value) for value in rows[1][3:]], frame.tolist())
            for name, _, _ in OBSERVATIONS:
                self.assertGreater((output_dir / f"{name}.png").stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
