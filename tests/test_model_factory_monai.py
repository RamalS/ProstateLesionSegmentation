from __future__ import annotations

import inspect
import unittest

import torch

from src.models import build_model

try:
    from monai.networks.nets import DynUNet, SwinUNETR

    _MONAI_AVAILABLE = True
except ImportError:
    DynUNet = object  # type: ignore[assignment,misc]
    SwinUNETR = object  # type: ignore[assignment,misc]
    _MONAI_AVAILABLE = False


class ModelFactoryRegressionTests(unittest.TestCase):
    def test_existing_model_keys_still_instantiate(self) -> None:
        shared = {
            "modalities": ["t2w", "adc", "hbv"],
            "out_channels": 1,
            "features": [16, 32, 64, 128],
            "deep_supervision": False,
        }

        m_unet = build_model({"model": "unet3d", **shared})
        self.assertEqual(type(m_unet).__name__, "UNet3D")

        m_attn = build_model({"model": "attention_unet3d", **shared})
        self.assertEqual(type(m_attn).__name__, "AttentionUNet3D")

        m_fct = build_model({"model": "fct", **shared})
        self.assertEqual(type(m_fct).__name__, "FCT")

    def test_deconver_key_still_instantiates_when_available(self) -> None:
        cfg = {
            "model": "deconver",
            "modalities": ["adc", "hbv"],
            "out_channels": 1,
            "deep_supervision": False,
            "deconver_encoder_depth": [1, 1, 1],
            "deconver_encoder_width": [16, 32, 64],
            "deconver_strides": [1, 2, 2],
            "deconver_kernel_size": [3, 3, 3],
            "deconver_groups": -1,
            "deconver_ndc_ratio": 2,
        }
        try:
            model = build_model(cfg)
        except ValueError as exc:
            self.skipTest(str(exc))
            return
        self.assertEqual(type(model).__name__.lower(), "deconver")


@unittest.skipUnless(_MONAI_AVAILABLE, "MONAI is not installed")
class MonaiFactoryTests(unittest.TestCase):
    def test_build_dynunet_from_minimal_cfg(self) -> None:
        cfg = {
            "model": "dynunet",
            "modalities": ["t2w", "adc"],
            "out_channels": 1,
            "deep_supervision": True,
            "dynunet_strides": [1, 2, 2],
            "dynunet_kernel_size": [3, 3, 3],
            "dynunet_upsample_kernel_size": [2, 2],
            "dynunet_filters": [16, 32, 64],
        }

        model = build_model(cfg)
        self.assertIsInstance(model, DynUNet)
        self.assertFalse(getattr(model, "deep_supervision", True))

    def test_build_swinunetr_from_minimal_cfg(self) -> None:
        cfg = {
            "model": "swinunetr",
            "modalities": ["t2w"],
            "out_channels": 1,
            "swinunetr_img_size": [32, 32, 32],
            "swinunetr_feature_size": 12,
            "swinunetr_depths": [1, 1, 1, 1],
            "swinunetr_num_heads": [3, 6, 12, 24],
            "swinunetr_use_checkpoint": False,
            "swinunetr_use_v2": False,
        }

        model = build_model(cfg)
        self.assertIsInstance(model, SwinUNETR)

    def test_dynunet_invalid_kernel_strides_shape_raises(self) -> None:
        cfg = {
            "model": "dynunet",
            "modalities": ["t2w"],
            "out_channels": 1,
            "dynunet_strides": [1, 2, 2],
            "dynunet_kernel_size": [3, 3],
            "dynunet_upsample_kernel_size": [2, 2],
        }

        with self.assertRaisesRegex(ValueError, "dynunet_kernel_size"):
            build_model(cfg)

    def test_swinunetr_invalid_depth_num_heads_shape_raises(self) -> None:
        cfg = {
            "model": "swinunetr",
            "modalities": ["t2w"],
            "out_channels": 1,
            "swinunetr_img_size": [32, 32, 32],
            "swinunetr_depths": [1, 1, 1, 1],
            "swinunetr_num_heads": [3, 6, 12],
        }

        with self.assertRaisesRegex(ValueError, "swinunetr_num_heads"):
            build_model(cfg)

    def test_dynunet_forward_shape(self) -> None:
        cfg = {
            "model": "dynunet",
            "modalities": ["t2w", "adc"],
            "out_channels": 2,
            "dynunet_strides": [1, 2, 2],
            "dynunet_kernel_size": [3, 3, 3],
            "dynunet_upsample_kernel_size": [2, 2],
            "dynunet_filters": [16, 32, 64],
        }
        model = build_model(cfg).eval()

        x = torch.randn(1, 2, 16, 32, 32)
        with torch.no_grad():
            y = model(x)

        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(tuple(y.shape), (1, 2, 16, 32, 32))

    def test_swinunetr_forward_shape(self) -> None:
        cfg = {
            "model": "swinunetr",
            "modalities": ["t2w"],
            "out_channels": 2,
            "swinunetr_img_size": [32, 64, 64],
            "swinunetr_feature_size": 12,
            "swinunetr_depths": [1, 1, 1, 1],
            "swinunetr_num_heads": [3, 6, 12, 24],
            "swinunetr_use_checkpoint": False,
            "swinunetr_use_v2": False,
        }
        model = build_model(cfg).eval()

        x = torch.randn(1, 1, 32, 64, 64)
        with torch.no_grad():
            y = model(x)

        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(tuple(y.shape), (1, 2, 32, 64, 64))

    def test_swinunetr_use_v2_guard_respects_signature(self) -> None:
        supports_use_v2 = "use_v2" in inspect.signature(SwinUNETR).parameters
        cfg = {
            "model": "swinunetr",
            "modalities": ["t2w"],
            "out_channels": 1,
            "swinunetr_img_size": [32, 32, 32],
            "swinunetr_feature_size": 12,
            "swinunetr_depths": [1, 1, 1, 1],
            "swinunetr_num_heads": [3, 6, 12, 24],
            "swinunetr_use_v2": True,
        }

        if supports_use_v2:
            model = build_model(cfg)
            self.assertIsInstance(model, SwinUNETR)
        else:
            with self.assertRaisesRegex(ValueError, "use_v2"):
                build_model(cfg)


if __name__ == "__main__":
    unittest.main()
