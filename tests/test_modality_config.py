from __future__ import annotations

from pathlib import Path
import sys
import unittest
import warnings


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import resolve_active_modalities, resolve_active_modality_pairs


class ResolveActiveModalitiesTests(unittest.TestCase):
    def test_modalities_only_preserves_order_and_normalizes(self) -> None:
        cfg = {"modalities": [" hbv", "ADC ", "t2w"]}
        self.assertEqual(resolve_active_modalities(cfg), ("hbv", "adc", "t2w"))

    def test_legacy_only_flags_match_expected(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolved = resolve_active_modalities(
                {"use_t2w": True, "use_adc": False, "use_hbv": True}
            )
        self.assertEqual(resolved, ("t2w", "hbv"))
        self.assertTrue(any("deprecated" in str(w.message).lower() for w in caught))

    def test_default_without_modality_keys_keeps_historical_behavior(self) -> None:
        self.assertEqual(resolve_active_modalities({}), ("t2w", "adc", "hbv"))

    def test_modalities_take_precedence_over_legacy_flags(self) -> None:
        cfg = {
            "modalities": ["adc"],
            "use_t2w": True,
            "use_adc": False,
            "use_hbv": True,
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolved = resolve_active_modalities(cfg)
        self.assertEqual(resolved, ("adc",))
        self.assertTrue(any("ignoring legacy" in str(w.message).lower() for w in caught))

    def test_invalid_unknown_modality_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown modality"):
            resolve_active_modalities({"modalities": ["t2w", "flair"]})

    def test_invalid_duplicate_modality_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate modality"):
            resolve_active_modalities({"modalities": ["t2w", "T2W"]})

    def test_invalid_empty_modalities_list_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            resolve_active_modalities({"modalities": []})

    def test_invalid_all_legacy_false_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "No modalities enabled"):
            resolve_active_modalities(
                {"use_t2w": False, "use_adc": False, "use_hbv": False}
            )

    def test_resolve_pairs_uses_active_order(self) -> None:
        cfg = {"modalities": ["adc", "t2w"]}
        suffixes = {"t2w": "_t2w.mha", "adc": "_adc.mha", "hbv": "_hbv.mha"}
        self.assertEqual(
            resolve_active_modality_pairs(cfg, suffixes),
            [("adc", "_adc.mha"), ("t2w", "_t2w.mha")],
        )


if __name__ == "__main__":
    unittest.main()
