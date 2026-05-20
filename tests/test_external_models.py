from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from external_models import MonaiBundleProstateMaskAdapter, resolve_external_model_request


class MonaiBundleAdapterTests(unittest.TestCase):
    def _fake_runtime_modules(self, load_fn) -> dict[str, types.ModuleType]:
        monai_mod = types.ModuleType("monai")
        monai_bundle_mod = types.ModuleType("monai.bundle")
        monai_bundle_mod.load = load_fn
        monai_mod.bundle = monai_bundle_mod
        return {
            "monai": monai_mod,
            "monai.bundle": monai_bundle_mod,
            "huggingface_hub": types.ModuleType("huggingface_hub"),
        }

    def test_preflight_failure_message_includes_proxy_and_cache_details(self) -> None:
        spec = resolve_external_model_request("monai:prostate_mri_anatomy@0.3.5")
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_root = Path(tmp_dir) / "cache_root"
            checked_path = cache_root / spec.bundle_name
            fake_modules = self._fake_runtime_modules(
                load_fn=lambda **_: torch.nn.Identity()
            )
            with (
                patch.dict(
                    os.environ,
                    {"HTTP_PROXY": "http://proxy.local:3128"},
                    clear=False,
                ),
                patch.dict(sys.modules, fake_modules, clear=False),
                patch.object(
                    MonaiBundleProstateMaskAdapter,
                    "_resolve_cached_bundle_dir",
                    return_value=(False, checked_path, [checked_path]),
                ),
                patch.object(
                    MonaiBundleProstateMaskAdapter,
                    "_bundle_probe_urls",
                    return_value=("https://example.invalid",),
                ),
                patch.object(
                    MonaiBundleProstateMaskAdapter,
                    "_check_url_reachable",
                    side_effect=RuntimeError("Network is unreachable"),
                ),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    MonaiBundleProstateMaskAdapter(
                        spec=spec,
                        device=torch.device("cpu"),
                        cache_root=cache_root,
                    )

        msg = str(ctx.exception)
        self.assertIn("preflight failed before load_bundle()", msg)
        self.assertIn("Network is unreachable", msg)
        self.assertIn("Proxy env:", msg)
        self.assertIn("HTTP_PROXY=set", msg)
        self.assertIn(str(cache_root), msg)

    def test_cache_miss_runs_preflight_then_loads_bundle(self) -> None:
        spec = resolve_external_model_request("monai:prostate_mri_anatomy@0.3.5")
        load_calls: list[dict[str, object]] = []

        def _fake_load(**kwargs):
            load_calls.append(kwargs)
            return torch.nn.Identity()

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_root = Path(tmp_dir) / "cache_root"
            checked_path = cache_root / spec.bundle_name
            fake_modules = self._fake_runtime_modules(load_fn=_fake_load)
            with (
                patch.dict(sys.modules, fake_modules, clear=False),
                patch.object(
                    MonaiBundleProstateMaskAdapter,
                    "_resolve_cached_bundle_dir",
                    return_value=(False, checked_path, [checked_path]),
                ),
                patch.object(
                    MonaiBundleProstateMaskAdapter,
                    "_check_url_reachable",
                    return_value=None,
                ) as preflight_check,
            ):
                adapter = MonaiBundleProstateMaskAdapter(
                    spec=spec,
                    device=torch.device("cpu"),
                    cache_root=cache_root,
                )

        self.assertIsInstance(adapter.model, torch.nn.Module)
        self.assertGreaterEqual(preflight_check.call_count, 1)
        self.assertEqual(len(load_calls), 1)
        self.assertEqual(load_calls[0]["bundle_dir"], str(cache_root))

    def test_cache_hit_skips_preflight(self) -> None:
        spec = resolve_external_model_request("monai:prostate_mri_anatomy@0.3.5")
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_root = Path(tmp_dir) / "cache_root"
            checked_path = cache_root / spec.bundle_name
            fake_modules = self._fake_runtime_modules(
                load_fn=lambda **_: torch.nn.Identity()
            )
            with (
                patch.dict(sys.modules, fake_modules, clear=False),
                patch.object(
                    MonaiBundleProstateMaskAdapter,
                    "_resolve_cached_bundle_dir",
                    return_value=(True, checked_path, [checked_path]),
                ),
                patch.object(
                    MonaiBundleProstateMaskAdapter,
                    "_preflight_bundle_connectivity",
                ) as preflight,
            ):
                MonaiBundleProstateMaskAdapter(
                    spec=spec,
                    device=torch.device("cpu"),
                    cache_root=cache_root,
                )

        preflight.assert_not_called()


if __name__ == "__main__":
    unittest.main()
