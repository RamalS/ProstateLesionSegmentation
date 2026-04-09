#!/usr/bin/env python3
"""
select_checkpoint.py — Interactive checkpoint selector.

Scans outputs/runs/ for checkpoints, displays them grouped by run
(newest run first), and launches:

    docker compose run --rm trainer train --resume <selected_checkpoint>

Use arrow keys (or j/k) to navigate, Enter to confirm, q/Esc to quit.
"""

from __future__ import annotations

import curses
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RUNS_DIR = Path(__file__).parent.parent / "outputs" / "runs"
DOCKER_CMD = ["docker", "compose", "run", "--rm", "trainer", "train"]

_RUN_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_(.+)$")

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def parse_run_folder(folder_name: str) -> str:
    """
    Convert a raw folder name to a human-readable run label.

    Parameters
    ----------
    folder_name : raw directory name, e.g. ``20260406_165336_baseline_run``

    Returns
    -------
    str
        Human-readable label, e.g. ``Apr 06 2026  16:53  ·  baseline run``
    """
    m = _RUN_RE.match(folder_name)
    if not m:
        return folder_name
    year, month, day, hour, minute, second, experiment = m.groups()
    dt = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
    exp_display = experiment.replace("_", " ")
    return f"{dt.strftime('%b %d %Y  %H:%M')}  ·  {exp_display}"


def _checkpoint_sort_key(name: str) -> tuple[int, int]:
    """Sort: best.pt first, then epoch files in descending order."""
    if name == "best.pt":
        return (0, 0)
    m = re.match(r"epoch_(\d+)\.pt$", name)
    if m:
        return (1, -int(m.group(1)))
    return (2, 0)


def _checkpoint_label(name: str) -> str:
    """Convert a filename like ``epoch_0005.pt`` to ``epoch 5``."""
    if name == "best.pt":
        return "best"
    m = re.match(r"epoch_0*(\d+)\.pt$", name)
    if m:
        return f"epoch {m.group(1)}"
    return name


def collect_items() -> list[dict]:
    """
    Build the flat item list consumed by the TUI.

    Returns a list of dicts with either::

        {"type": "header",     "label": <str>}
        {"type": "checkpoint", "label": <str>, "host_path": <str>, "container_path": <str>}
    """
    if not RUNS_DIR.exists():
        return []

    run_dirs = sorted(
        [d for d in RUNS_DIR.iterdir() if d.is_dir() and _RUN_RE.match(d.name)],
        key=lambda d: d.name,
        reverse=True,  # newest first
    )

    items: list[dict] = []
    for run_dir in run_dirs:
        ckpt_dir = run_dir / "checkpoints"
        if not ckpt_dir.exists():
            continue
        pts = sorted(
            [f for f in ckpt_dir.iterdir() if f.suffix == ".pt"],
            key=lambda f: _checkpoint_sort_key(f.name),
        )
        if not pts:
            continue

        items.append({"type": "header", "label": parse_run_folder(run_dir.name)})
        for pt in pts:
            # Map host path → container path (/outputs/runs/...)
            # compose.yml mounts ./outputs → /outputs inside the container
            rel = pt.relative_to(RUNS_DIR.parent.parent)  # outputs/runs/.../....pt
            container_path = "/" + str(rel)
            items.append({
                "type": "checkpoint",
                "label": f"  {_checkpoint_label(pt.name)}",
                "host_path": str(pt),
                "container_path": container_path,
            })

    return items


# ---------------------------------------------------------------------------
# TUI
# ---------------------------------------------------------------------------

_COLOR_SELECTED = 1
_COLOR_HEADER   = 2
_COLOR_DIM      = 3


def _draw(
    stdscr: curses.window,
    items: list[dict],
    selectable: list[int],
    sel_idx: int,
    scroll: int,
) -> None:
    """Redraw the full screen."""
    height, width = stdscr.getmaxyx()
    visible_rows = height - 4  # title bar (2) + footer (1) + 1 padding

    stdscr.erase()

    # ---- Title bar ----
    title = " Checkpoint Selector   ↑↓ / j k  navigate     Enter  select     q  quit "
    stdscr.addstr(0, 0, title[:width - 1], curses.A_BOLD | curses.A_REVERSE)
    stdscr.addstr(1, 0, ("─" * (width - 1))[:width - 1])

    # ---- Item list ----
    for row in range(visible_rows):
        item_idx = scroll + row
        if item_idx >= len(items):
            break
        y = row + 2
        item = items[item_idx]

        if item["type"] == "header":
            label = f" {item['label']}"
            stdscr.addstr(y, 0, label[:width - 1], curses.color_pair(_COLOR_HEADER) | curses.A_BOLD)

        else:
            is_sel = selectable[sel_idx] == item_idx
            label = item["label"]
            if is_sel:
                padded = label.ljust(width - 1)
                stdscr.addstr(y, 0, padded[:width - 1], curses.color_pair(_COLOR_SELECTED))
            else:
                stdscr.addstr(y, 0, label[:width - 1])

    # ---- Footer: path preview ----
    footer_y = height - 1
    preview = items[selectable[sel_idx]]["container_path"]
    stdscr.addstr(footer_y, 0, f" {preview}"[:width - 1], curses.color_pair(_COLOR_DIM))

    stdscr.refresh()


def tui(stdscr: curses.window, items: list[dict]) -> str | None:
    """
    Run the interactive TUI and return the selected container path, or
    ``None`` if the user cancelled.
    """
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(_COLOR_SELECTED, curses.COLOR_BLACK, curses.COLOR_CYAN)
    curses.init_pair(_COLOR_HEADER,   curses.COLOR_YELLOW, -1)
    curses.init_pair(_COLOR_DIM,      curses.COLOR_WHITE,  -1)

    selectable = [i for i, it in enumerate(items) if it["type"] == "checkpoint"]
    if not selectable:
        return None

    sel_idx = 0   # index into selectable[]
    scroll  = 0   # first item_idx rendered

    while True:
        height, _ = stdscr.getmaxyx()
        visible_rows = height - 4
        sel_item_idx = selectable[sel_idx]

        # Keep selected item within the visible window
        if sel_item_idx - scroll >= visible_rows:
            scroll = sel_item_idx - visible_rows + 1
        if sel_item_idx - scroll < 0:
            scroll = sel_item_idx

        _draw(stdscr, items, selectable, sel_idx, scroll)

        key = stdscr.getch()

        if key in (curses.KEY_UP, ord("k")):
            if sel_idx > 0:
                sel_idx -= 1

        elif key in (curses.KEY_DOWN, ord("j")):
            if sel_idx < len(selectable) - 1:
                sel_idx += 1

        elif key in (curses.KEY_ENTER, 10, 13):
            return items[selectable[sel_idx]]["container_path"]

        elif key in (ord("q"), 27):   # q or Escape
            return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    items = collect_items()

    if not items:
        print(f"No checkpoints found under {RUNS_DIR}")
        sys.exit(1)

    selected_container_path = curses.wrapper(tui, items)

    if selected_container_path is None:
        print("Aborted.")
        sys.exit(0)

    cmd = DOCKER_CMD + ["--resume", selected_container_path]
    print(f"\nRunning: {' '.join(cmd)}\n")
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
