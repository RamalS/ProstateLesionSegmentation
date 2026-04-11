"""
notify.py — ntfy push notification helper.

Sends training event notifications to an ntfy server.
All functions are **silent no-ops** when ``ntfy_url`` or ``ntfy_topic`` are
absent or empty in the config dict, so no config changes are needed to
disable notifications.  Network errors are caught and logged as warnings so
a dead ntfy server never crashes a training run.

Usage
-----
    from src.notify import send_ntfy

    send_ntfy(
        cfg,
        title="Training complete: baseline_run",
        message="Best dice: 0.7834\\nRun dir: outputs/runs/baseline_run_20260411_120000",
        tags=["white_check_mark"],
        priority="default",
    )

Config keys (all optional)
--------------------------
    ntfy_url              : str  — ntfy server base URL, e.g. "https://ntfy.sh"
    ntfy_topic            : str  — topic/channel name, e.g. "my-training-alerts"
    ntfy_notify_best_model: bool — send a notification on new best model (default: True)
"""

from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def send_ntfy(
    cfg: dict[str, Any],
    title: str,
    message: str,
    tags: list[str] | None = None,
    priority: str = "default",
) -> None:
    """
    Send a push notification via ntfy.

    Does nothing when ``ntfy_url`` or ``ntfy_topic`` are absent or empty.
    Network failures are caught and emitted as a ``WARNING`` log entry so
    they never propagate to the caller.

    Parameters
    ----------
    cfg      : training config dict (must contain ``ntfy_url`` and ``ntfy_topic``
               for any notification to be sent).
    title    : notification title shown in the ntfy client.
    message  : notification body text.
    tags     : list of ntfy emoji tag names (e.g. ``["trophy", "rocket"]``).
               See https://docs.ntfy.sh/emojis/ for the full list.
    priority : ntfy priority string — one of ``"min"``, ``"low"``,
               ``"default"``, ``"high"``, ``"urgent"``.
    """
    url: str = cfg.get("ntfy_url", "") or ""
    topic: str = cfg.get("ntfy_topic", "") or ""

    if not url or not topic:
        return

    endpoint = f"{url.rstrip('/')}/{topic}"
    headers: dict[str, str] = {
        "Title": title,
        "Priority": priority,
        "Content-Type": "text/plain",
    }
    if tags:
        headers["Tags"] = ",".join(tags)

    try:
        response = requests.post(
            endpoint,
            data=message.encode("utf-8"),
            headers=headers,
            timeout=10,
        )
        response.raise_for_status()
        logger.debug("ntfy notification sent: %s → %s", title, endpoint)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ntfy notification failed (training continues): %s", exc)
