from __future__ import annotations

import unittest

import torch

from src.train import _roi_bounds_from_collated_batch


class TrainROICollateTests(unittest.TestCase):
    def test_normalizes_default_collate_triplets(self) -> None:
        payload = {
            "start_zyx": [torch.tensor([1]), torch.tensor([2]), torch.tensor([3])],
            "end_zyx": [torch.tensor([11]), torch.tensor([22]), torch.tensor([33])],
            "full_shape_zyx": [torch.tensor([64]), torch.tensor([128]), torch.tensor([192])],
            "used_fallback": torch.tensor([False]),
        }

        out = _roi_bounds_from_collated_batch(payload)
        self.assertEqual(out["start_zyx"], (1, 2, 3))
        self.assertEqual(out["end_zyx"], (11, 22, 33))
        self.assertEqual(out["full_shape_zyx"], (64, 128, 192))
        self.assertFalse(out["used_fallback"])

    def test_normalizes_tensor_triplets(self) -> None:
        payload = {
            "start_zyx": torch.tensor([[4, 5, 6]]),
            "end_zyx": torch.tensor([[14, 15, 16]]),
            "full_shape_zyx": torch.tensor([[24, 25, 26]]),
            "used_fallback": torch.tensor([True]),
        }

        out = _roi_bounds_from_collated_batch(payload)
        self.assertEqual(out["start_zyx"], (4, 5, 6))
        self.assertEqual(out["end_zyx"], (14, 15, 16))
        self.assertEqual(out["full_shape_zyx"], (24, 25, 26))
        self.assertTrue(out["used_fallback"])

    def test_rejects_invalid_triplet_layout(self) -> None:
        payload = {
            "start_zyx": torch.tensor([[1, 2]]),
            "end_zyx": [torch.tensor([1]), torch.tensor([2]), torch.tensor([3])],
            "full_shape_zyx": [torch.tensor([10]), torch.tensor([20]), torch.tensor([30])],
            "used_fallback": torch.tensor([False]),
        }

        with self.assertRaisesRegex(ValueError, "roi.start_zyx"):
            _roi_bounds_from_collated_batch(payload)


if __name__ == "__main__":
    unittest.main()
