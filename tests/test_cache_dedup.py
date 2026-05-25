from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.dataset import PiCaiDataset
from src.roi import resolve_roi_settings

from tests.test_roi_stage_cache import _base_case, _fake_payload


class CacheDedupTests(unittest.TestCase):
    def test_roi_storage_cache_does_not_write_redundant_dataset_storage_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = _base_case(root)
            dataset_dir = root / "dataset_cache"
            roi_dir = root / "roi_cache"

            with patch.object(PiCaiDataset, "_load_and_preprocess", return_value=_fake_payload()):
                ds = PiCaiDataset(
                    images_dir=tmp,
                    labels_dir=tmp,
                    cases=[dict(case)],
                    cache_mode="storage",
                    cache_dir=dataset_dir,
                    roi_cache_mode="storage",
                    roi_cache_dir=roi_dir,
                    roi_settings=resolve_roi_settings({"roi": {"mode": "gt_mask"}}),
                    include_full_resampled=True,
                )
                sample = ds[0]

            self.assertIn("roi", sample)
            self.assertEqual(list(dataset_dir.glob("*.pt")), [])
            self.assertEqual(len(list(roi_dir.glob("*.pt"))), 1)


if __name__ == "__main__":
    unittest.main()
