from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import torch

from roi import keep_largest_component


logger = logging.getLogger(__name__)

_PROXY_ENV_VARS: tuple[str, ...] = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
)
_DEFAULT_BUNDLE_PREFLIGHT_TIMEOUT_SEC = 3.0
_BUNDLE_PROBE_URLS_BY_SOURCE: dict[str, tuple[str, ...]] = {
    "monaihosting": (
        "https://api.ngc.nvidia.com/v2/models/nvidia/monaihosting",
        "https://huggingface.co",
    ),
    "huggingface_hub": ("https://huggingface.co",),
    "github": ("https://github.com",),
    "ngc": ("https://api.ngc.nvidia.com",),
    "ngc_private": ("https://api.ngc.nvidia.com",),
}


@dataclass(frozen=True)
class ExternalModelSpec:
    model_id: str
    model_source: str
    display_name: str
    bundle_name: str
    bundle_version: str
    dataset_type: str
    task: str
    required_modalities: tuple[str, ...]
    target_spacing: tuple[float, float, float]
    patch_size: tuple[int, int, int]
    sw_overlap: float
    bundle_source: str = "monaihosting"
    repo: str | None = None
    pred_threshold: float = 0.5

    @property
    def versioned_id(self) -> str:
        return f"{self.model_id}@{self.bundle_version}"


_SUPPORTED_EXTERNAL_MODELS: dict[tuple[str, str], ExternalModelSpec] = {
    (
        "monai:prostate_mri_anatomy",
        "0.3.5",
    ): ExternalModelSpec(
        model_id="monai:prostate_mri_anatomy",
        model_source="monai_bundle",
        display_name="MONAI prostate_mri_anatomy",
        bundle_name="prostate_mri_anatomy",
        bundle_version="0.3.5",
        dataset_type="prostate158",
        task="prostate_localization",
        required_modalities=("t2w",),
        target_spacing=(0.5, 0.5, 0.5),
        patch_size=(96, 96, 96),
        sw_overlap=0.5,
    ),
}

_EXTERNAL_LOCALIZER_PREFIX = "external://"


def list_supported_external_models() -> list[ExternalModelSpec]:
    return sorted(
        _SUPPORTED_EXTERNAL_MODELS.values(),
        key=lambda spec: (spec.model_source, spec.model_id, spec.bundle_version),
    )


def resolve_external_model_request(
    model_id: str,
    version: str = "",
) -> ExternalModelSpec:
    token = str(model_id).strip()
    version_text = str(version).strip()
    if not token:
        raise ValueError("External model id must not be empty.")

    if "@" in token:
        token, parsed_version = token.split("@", 1)
        token = token.strip()
        parsed_version = parsed_version.strip()
        if version_text and version_text != parsed_version:
            raise ValueError(
                "Conflicting external model versions were provided via "
                "--external-model and --external-model-version."
            )
        version_text = parsed_version

    if ":" not in token:
        token = f"monai:{token}"

    key = (token, version_text or "0.3.5")
    spec = _SUPPORTED_EXTERNAL_MODELS.get(key)
    if spec is None:
        supported = ", ".join(
            spec.versioned_id for spec in list_supported_external_models()
        )
        raise ValueError(
            f"Unsupported external model '{token}' version "
            f"'{version_text or 'latest'}'. Supported values: {supported}"
        )
    return spec


def default_external_model_cache_root(repo_root: Path) -> Path:
    docker_cache = Path("/cache/monai_bundles")
    if docker_cache.parent.exists():
        return docker_cache
    return (repo_root / "cache" / "monai_bundles").resolve()


def build_external_localizer_ref(model_id: str, version: str) -> str:
    spec = resolve_external_model_request(model_id, version)
    return f"{_EXTERNAL_LOCALIZER_PREFIX}{spec.versioned_id}"


def parse_external_localizer_ref(value: str) -> ExternalModelSpec | None:
    token = str(value).strip()
    if not token.startswith(_EXTERNAL_LOCALIZER_PREFIX):
        return None
    payload = token[len(_EXTERNAL_LOCALIZER_PREFIX) :]
    return resolve_external_model_request(payload)


def build_external_eval_config(
    spec: ExternalModelSpec,
    *,
    prostate158_root: str = "",
    prostate158_label_reader: str = "",
) -> dict[str, Any]:
    modalities = [str(key) for key in spec.required_modalities]
    return {
        "model_source": spec.model_source,
        "external_model_id": spec.model_id,
        "external_model_version": spec.bundle_version,
        "dataset_type": spec.dataset_type,
        "task": spec.task,
        "modalities": modalities,
        # Transition compatibility for unchanged callsites that still inspect
        # legacy modality flags.
        "use_t2w": "t2w" in modalities,
        "use_adc": "adc" in modalities,
        "use_hbv": "hbv" in modalities,
        "target_spacing": list(spec.target_spacing),
        "patch_size": list(spec.patch_size),
        "sw_overlap": spec.sw_overlap,
        "pred_threshold": spec.pred_threshold,
        "postprocess_enabled": False,
        "dwi_hbv_preprocess": {"enabled": False},
        "roi": {"mode": "disabled"},
        "prostate158_test_dir": prostate158_root or "data/prostate158_test",
        "prostate158_test_split": "test",
        "prostate158_label_target": "tumor",
        "prostate158_label_reader": prostate158_label_reader or "1",
    }


class MonaiBundleProstateMaskAdapter:
    def __init__(
        self,
        spec: ExternalModelSpec,
        *,
        device: torch.device,
        cache_root: Path,
    ) -> None:
        self.spec = spec
        self.device = device
        self.cache_root = cache_root
        self.model = self._load_model()
        self.model.eval()

    def _load_model(self) -> torch.nn.Module:
        try:
            from monai.bundle import load as load_bundle
        except ImportError as exc:
            raise RuntimeError(
                "MONAI bundle support requires the 'monai' package in the runtime environment."
            ) from exc
        try:
            import huggingface_hub  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "MONAI bundle download requires 'huggingface_hub'. "
                "Install it locally or rebuild the Docker image after updating requirements.txt."
            ) from exc

        self.cache_root.mkdir(parents=True, exist_ok=True)
        cache_hit, cache_path, checked_paths = self._resolve_cached_bundle_dir()
        checked_paths_desc = ", ".join(str(path) for path in checked_paths)
        if cache_hit:
            logger.info(
                "External MONAI bundle cache hit: %s (checked: %s).",
                cache_path,
                checked_paths_desc,
            )
        else:
            logger.info(
                "External MONAI bundle cache miss for %s@%s (checked: %s).",
                self.spec.bundle_name,
                self.spec.bundle_version,
                checked_paths_desc,
            )
            self._preflight_bundle_connectivity(checked_paths)
        logger.info(
            "Preparing external MONAI bundle %s@%s in %s. "
            "First-time download can take several minutes and may appear quiet.",
            self.spec.bundle_name,
            self.spec.bundle_version,
            self.cache_root,
        )
        started = time.monotonic()
        try:
            model = load_bundle(
                name=self.spec.bundle_name,
                version=self.spec.bundle_version,
                workflow_type="inference",
                bundle_dir=str(self.cache_root),
                source=self.spec.bundle_source,
                repo=self.spec.repo,
                device=self.device,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to load external MONAI bundle "
                f"{self.spec.bundle_name}@{self.spec.bundle_version}. "
                f"Reason: {exc.__class__.__name__}: {exc}. "
                f"Proxy env: {self._proxy_env_status()}. "
                f"Bundle cache root: {self.cache_root}. "
                f"Checked cache path(s): {', '.join(str(path) for path in checked_paths)}."
            ) from exc
        elapsed = time.monotonic() - started
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"Expected MONAI bundle loader to return nn.Module, got {type(model)!r}."
            )
        logger.info(
            "External MONAI bundle %s@%s loaded in %.1fs.",
            self.spec.bundle_name,
            self.spec.bundle_version,
            elapsed,
        )
        return model.to(self.device)

    def _bundle_cache_candidates(self) -> tuple[Path, ...]:
        return (
            self.cache_root / self.spec.bundle_name,
            self.cache_root / f"{self.spec.bundle_name}_{self.spec.bundle_version}",
            self.cache_root / f"{self.spec.bundle_name}_v{self.spec.bundle_version}",
            self.cache_root / self.spec.bundle_name / self.spec.bundle_version,
        )

    @staticmethod
    def _looks_like_bundle_dir(path: Path) -> bool:
        return (
            path.is_dir()
            and (path / "configs" / "metadata.json").is_file()
            and (path / "models").is_dir()
        )

    def _resolve_cached_bundle_dir(self) -> tuple[bool, Path, list[Path]]:
        checked_paths: list[Path] = []
        seen: set[Path] = set()

        def _track(path: Path) -> None:
            if path not in seen:
                seen.add(path)
                checked_paths.append(path)

        for candidate in self._bundle_cache_candidates():
            _track(candidate)
            if self._looks_like_bundle_dir(candidate):
                return True, candidate, checked_paths

        if self.cache_root.exists():
            for metadata_path in self.cache_root.rglob("metadata.json"):
                if metadata_path.parent.name != "configs":
                    continue
                bundle_dir = metadata_path.parent.parent
                if self.spec.bundle_name not in bundle_dir.name:
                    continue
                _track(bundle_dir)
                if self._looks_like_bundle_dir(bundle_dir):
                    return True, bundle_dir, checked_paths

        return False, self.cache_root / self.spec.bundle_name, checked_paths

    @staticmethod
    def _proxy_env_status() -> str:
        return ", ".join(
            f"{key}={'set' if os.getenv(key) else 'unset'}" for key in _PROXY_ENV_VARS
        )

    def _bundle_probe_urls(self) -> tuple[str, ...]:
        return _BUNDLE_PROBE_URLS_BY_SOURCE.get(
            self.spec.bundle_source,
            ("https://huggingface.co",),
        )

    @staticmethod
    def _check_url_reachable(url: str, timeout_sec: float) -> None:
        try:
            request = Request(url=url, method="HEAD")
            with urlopen(request, timeout=timeout_sec):
                return
        except HTTPError as exc:
            if exc.code == 407:
                raise RuntimeError(
                    "Proxy authentication failed (HTTP 407) while checking connectivity."
                ) from exc
            return
        except URLError as exc:
            raise RuntimeError(str(exc.reason or exc)) from exc
        except TimeoutError as exc:
            raise RuntimeError("connection timed out") from exc

    def _preflight_bundle_connectivity(self, checked_paths: list[Path]) -> None:
        timeout_sec = float(
            os.getenv(
                "MONAI_BUNDLE_PREFLIGHT_TIMEOUT_SEC",
                str(_DEFAULT_BUNDLE_PREFLIGHT_TIMEOUT_SEC),
            )
        )
        failures: list[tuple[str, str]] = []
        for url in self._bundle_probe_urls():
            try:
                self._check_url_reachable(url, timeout_sec=timeout_sec)
                logger.info(
                    "External bundle connectivity preflight succeeded via %s (timeout %.1fs).",
                    url,
                    timeout_sec,
                )
                return
            except Exception as exc:  # noqa: BLE001
                failures.append((url, f"{exc.__class__.__name__}: {exc}"))

        failure_details = "; ".join(f"{url} -> {msg}" for url, msg in failures)
        raise RuntimeError(
            "External MONAI bundle download preflight failed before load_bundle(). "
            f"Unable to reach hosting endpoints for source={self.spec.bundle_source!r}. "
            f"Failure(s): {failure_details}. "
            f"Proxy env: {self._proxy_env_status()}. "
            f"Bundle cache root: {self.cache_root}. "
            f"Checked cache path(s): {', '.join(str(path) for path in checked_paths)}."
        )

    @staticmethod
    def _normalize_t2w(images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5:
            raise ValueError(
                f"Expected external-model input shape (B, C, D, H, W), got {tuple(images.shape)}."
            )
        if images.shape[1] != 1:
            raise ValueError(
                f"MONAI prostate bundle expects T2w-only input (1 channel), got {images.shape[1]} channels."
            )
        x = images.float()
        flat = x.flatten(start_dim=2)
        mins = flat.min(dim=2).values.view(x.shape[0], 1, 1, 1, 1)
        maxs = flat.max(dim=2).values.view(x.shape[0], 1, 1, 1, 1)
        x = (x - mins) / torch.clamp(maxs - mins, min=1e-6)
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        std = x.std(dim=(2, 3, 4), keepdim=True, unbiased=False)
        x = (x - mean) / torch.clamp(std, min=1e-6)
        return x

    @staticmethod
    def _foreground_mask_from_multiclass(logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 5:
            raise ValueError(
                f"Expected MONAI bundle logits shape (B, C, D, H, W), got {tuple(logits.shape)}."
            )
        if logits.shape[1] <= 1:
            return (logits > 0).float()

        labels = torch.argmax(logits, dim=1)
        masks: list[torch.Tensor] = []
        for idx in range(labels.shape[0]):
            fg_mask = (labels[idx] > 0).detach().cpu().numpy().astype(np.uint8)
            fg_mask = keep_largest_component(fg_mask).astype(np.float32)
            masks.append(torch.from_numpy(fg_mask))
        return torch.stack(masks, dim=0).unsqueeze(1).to(logits.device)

    @classmethod
    def _binary_logits_from_output(cls, logits: torch.Tensor) -> torch.Tensor:
        fg_mask = cls._foreground_mask_from_multiclass(logits)
        return (fg_mask * 200.0) - 100.0

    def predict_logits(self, images: torch.Tensor, *, sw_batch_size: int) -> torch.Tensor:
        try:
            from monai.inferers import sliding_window_inference
        except ImportError as exc:
            raise RuntimeError(
                "MONAI sliding-window inference is unavailable; install the 'monai' package."
            ) from exc
        normed = self._normalize_t2w(images.to(self.device))
        raw_logits = sliding_window_inference(
            inputs=normed,
            roi_size=self.spec.patch_size,
            sw_batch_size=sw_batch_size,
            predictor=self.model,
            overlap=self.spec.sw_overlap,
        )
        return self._binary_logits_from_output(raw_logits.float())
