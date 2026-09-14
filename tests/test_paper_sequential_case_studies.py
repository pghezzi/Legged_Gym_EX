"""Small post-hoc accounting checks without simulator imports or inference."""
import ast
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "legged_gym/scripts/depth_data_pipeline/paper_sequential_case_studies.py"
spec = importlib.util.spec_from_file_location("case_studies", PATH)
case = importlib.util.module_from_spec(spec)
spec.loader.exec_module(case)


def existing_function(path, name):
    # Read the exact existing pure function without initializing a simulator.
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = {"torch": torch, "Sequence": list}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


class CaseTests(unittest.TestCase):
    def test_actual_window_definition_and_sequence_boundaries(self):
        truth = [0, 0, 1, 1, 2, 2, 3, 3, 0]
        ids = ["a"]*6+["b"]*3
        segs, mask = case.segments_and_mask(truth, ids, radius=1)
        reference = existing_function("legged_gym/scripts/depth_data_pipeline/sequential_terrain_filter_extensions.py", "_transition_window_mask")
        np.testing.assert_array_equal(mask, reference(truth, ids, 1).numpy())
        self.assertEqual([(s["start"], s["end"]) for s in segs], [(0,2),(2,4),(4,6),(6,8),(8,9)])
        self.assertFalse(mask[6])  # A change across a sequence reset is not a transition.

    def test_segment_match_not_crop_or_later_segment(self):
        seg = dict(start=2, end=4, label=1)
        self.assertEqual(case.first_match(np.array([0,0,1,0,0]), seg), 0)
        self.assertEqual(case.first_match(np.array([0,0,0,1,0]), seg), 1)
        self.assertIsNone(case.first_match(np.array([0,0,0,0,1]), seg))

    def test_full_length_not_cropped_length(self):
        truth = np.array([0]*3+[1]*12+[2]*3)
        segments, _ = case.segments_and_mask(truth, ["a"]*18)
        middle = segments[1]
        self.assertEqual(middle["end"]-middle["start"], 12)
        self.assertFalse(middle["terminal"])
        self.assertEqual(case.runs_of(np.array([False, True, True, False])), [(1,3)])

    def test_false_transition_definition_and_unavailable(self):
        truth = np.array([0,0,1,1])
        pred = np.array([0,1,1,0])
        metrics = case.metrics(pred, truth, np.ones(4, bool), 0, 4)
        self.assertAlmostEqual(metrics["erroneous_switch_rate"], 2/3)
        self.assertIsNone(metrics["steady_state_accuracy"])


if __name__ == "__main__":
    unittest.main()
