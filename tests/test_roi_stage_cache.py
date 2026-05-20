from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.dataset import PiCaiDataset, _PreprocessedCase
from src.roi import resolve_roi_settings, validate_task_and_roi_config


def _base_case(root: Path) -> dict:
    case_dir = root / "case"
    case_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in ("t2w", "adc", "hbv"):
        path = case_dir / f"sample_{name}.mha"
        path.touch()
        paths[name] = path
    label = case_dir / "sample.nii.gz"
    label.touch()
    prostate = case_dir / "sample_prostate.nii.gz"
    prostate.touch()
    return {
        "case_id": "sample",
        "t2w": paths["t2w"],
        "adc": paths["adc"],
        "hbv": paths["hbv"],
        "label": label,
        "prostate_label": prostate,
    }


def _fake_payload() -> _PreprocessedCase:
    image = torch.arange(1 * 6 * 6 * 6, dtype=torch.float32).reshape(1, 6, 6, 6)
    lesion = torch.zeros((1, 6, 6, 6), dtype=torch.float32)
    lesion[:, 2:4, 2:4, 2:4] = 1.0
    prostate = torch.zeros((1, 6, 6, 6), dtype=torch.float32)
    prostate[:, 2:4, 2:4, 2:4] = 1.0
    return _PreprocessedCase(image=image, lesion_label=lesion, prostate_label=prostate)


class ROIStageConfigTests(unittest.TestCase):
    def test_global_mode_falls_back_for_both_stages(self) -> None:
        cfg = {"roi": {"mode": "gt_mask"}}
        self.assertEqual(resolve_roi_settings(cfg, stage="train").mode, "gt_mask")
        self.assertEqual(resolve_roi_settings(cfg, stage="val").mode, "gt_mask")

    def test_stage_override_wins_over_shared_mode(self) -> None:
        cfg = {
            "roi": {
                "mode": "gt_mask",
                "train": {"mode": "disabled"},
                "val": {"mode": "predicted_mask", "localizer_run": "/tmp/run"},
            }
        }
        self.assertEqual(resolve_roi_settings(cfg, stage="train").mode, "disabled")
        self.assertEqual(resolve_roi_settings(cfg, stage="val").mode, "predicted_mask")

    def test_invalid_stage_override_fails_fast(self) -> None:
        cfg = {"roi": {"train": {"mode": "bad_mode"}}}
        with self.assertRaisesRegex(ValueError, "roi.train.mode"):
            resolve_roi_settings(cfg, stage="train")

    def test_validate_task_and_roi_config_returns_per_stage_settings(self) -> None:
        cfg = {
            "dataset_type": "prostate158",
            "prostate158_prostate_label_col": "t2_prostate_reader1",
            "roi": {
                "mode": "disabled",
                "val": {"mode": "gt_mask"},
            },
        }
        _, roi_by_stage = validate_task_and_roi_config(cfg, "prostate158")
        self.assertEqual(roi_by_stage["train"].mode, "disabled")
        self.assertEqual(roi_by_stage["val"].mode, "gt_mask")


class ROICacheDatasetTests(unittest.TestCase):
    def test_train_full_val_crop_have_different_sample_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = _base_case(Path(tmp))
            with patch.object(PiCaiDataset, "_load_and_preprocess", return_value=_fake_payload()):
                train_ds = PiCaiDataset(
                    images_dir=tmp,
                    labels_dir=tmp,
                    cases=[dict(case)],
                    cache_mode="none",
                    roi_settings=resolve_roi_settings({"roi": {"mode": "disabled"}}),
                )
                val_ds = PiCaiDataset(
                    images_dir=tmp,
                    labels_dir=tmp,
                    cases=[dict(case)],
                    cache_mode="none",
                    roi_settings=resolve_roi_settings({"roi": {"mode": "gt_mask"}}),
                    include_full_resampled=True,
                )

                train_sample = train_ds[0]
                val_sample = val_ds[0]

        self.assertNotIn("roi", train_sample)
        self.assertEqual(tuple(train_sample["image"].shape), (1, 6, 6, 6))
        self.assertIn("roi", val_sample)
        self.assertIn("full_label", val_sample)
        self.assertEqual(tuple(val_sample["full_label"].shape), (1, 6, 6, 6))
        self.assertEqual(tuple(val_sample["image"].shape), (1, 6, 6, 6))

    def test_gt_roi_ram_cache_hits_without_reloading(self) -> None:
        call_count = 0

        def fake_load(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _fake_payload()

        with tempfile.TemporaryDirectory() as tmp:
            case = _base_case(Path(tmp))
            with patch.object(PiCaiDataset, "_load_and_preprocess", side_effect=fake_load):
                ds = PiCaiDataset(
                    images_dir=tmp,
                    labels_dir=tmp,
                    cases=[dict(case)],
                    cache_mode="none",
                    roi_cache_mode="ram",
                    roi_settings=resolve_roi_settings({"roi": {"mode": "gt_mask"}}),
                    include_full_resampled=True,
                )
                sample_a = ds[0]
                sample_b = ds[0]

        self.assertEqual(call_count, 1)
        self.assertEqual(tuple(sample_a["image"].shape), tuple(sample_b["image"].shape))
        self.assertIn("full_label", sample_a)

    def test_roi_storage_cache_key_changes_with_margin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = _base_case(root)
            roi_dir = root / "roi_cache"
            common = {
                "images_dir": tmp,
                "labels_dir": tmp,
                "cases": [dict(case)],
                "cache_mode": "none",
                "roi_cache_mode": "storage",
                "roi_cache_dir": roi_dir,
            }
            ds_a = PiCaiDataset(
                **common,
                roi_settings=resolve_roi_settings(
                    {"roi": {"mode": "gt_mask", "margin_mm": [1.0, 1.0, 1.0]}}
                ),
            )
            ds_b = PiCaiDataset(
                **common,
                roi_settings=resolve_roi_settings(
                    {"roi": {"mode": "gt_mask", "margin_mm": [2.0, 1.0, 1.0]}}
                ),
            )

            self.assertNotEqual(ds_a._roi_storage_cache_path(case), ds_b._roi_storage_cache_path(case))

    def test_roi_storage_cache_reused_across_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = _base_case(root)
            roi_dir = root / "roi_cache"
            cfg = {"roi": {"mode": "gt_mask"}}

            with patch.object(PiCaiDataset, "_load_and_preprocess", return_value=_fake_payload()):
                ds_first = PiCaiDataset(
                    images_dir=tmp,
                    labels_dir=tmp,
                    cases=[dict(case)],
                    cache_mode="none",
                    roi_cache_mode="storage",
                    roi_cache_dir=roi_dir,
                    roi_settings=resolve_roi_settings(cfg),
                    include_full_resampled=True,
                )
                first = ds_first[0]

            with patch.object(PiCaiDataset, "_load_and_preprocess", side_effect=AssertionError("unexpected reload")):
                ds_second = PiCaiDataset(
                    images_dir=tmp,
                    labels_dir=tmp,
                    cases=[dict(case)],
                    cache_mode="none",
                    roi_cache_mode="storage",
                    roi_cache_dir=roi_dir,
                    roi_settings=resolve_roi_settings(cfg),
                    include_full_resampled=True,
                )
                second = ds_second[0]

        self.assertEqual(tuple(first["image"].shape), tuple(second["image"].shape))
        self.assertIn("full_label", second)


if __name__ == "__main__":
    unittest.main()
