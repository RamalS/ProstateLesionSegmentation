#!/usr/bin/env python3
"""
report_pipeline.py - End-to-end run reporting automation.

Pipeline steps per processed run:
1) Evaluate checkpoint (best.pt auto-selected in non-interactive mode)
2) Export 3D orbit GIF for a fixed comparison case
3) Generate explainability artifacts via explain_case.py
4) Save visual artifacts under visualizations/
5) Regenerate report.md (or custom --report-output)

Default mode processes only runs missing required artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


RUN_NAME_RE = re.compile(r"^\d{8}_\d{6}_.+$")


@dataclass
class RunArtifacts:
    run_dir: Path
    eval_png: Path
    orbit_gif: Path
    xai_summary: Path
    needs_eval: bool
    needs_orbit: bool
    needs_xai: bool
    missing_reasons: list[str]


def _abs(path_like: str | Path) -> Path:
    p = Path(path_like).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return p.resolve()


def _find_runs(base_runs_dir: Path, run_args: list[str]) -> list[Path]:
    if run_args:
        runs: list[Path] = []
        for raw in run_args:
            run_dir = _abs(raw)
            if not run_dir.exists() or not run_dir.is_dir():
                raise FileNotFoundError(f"Run path not found or not a directory: {run_dir}")
            runs.append(run_dir)
        return sorted(runs, key=lambda p: p.name, reverse=True)

    if not base_runs_dir.exists():
        raise FileNotFoundError(f"Base runs directory does not exist: {base_runs_dir}")

    runs = [
        d for d in base_runs_dir.iterdir()
        if d.is_dir()
        and d.name != "latest"
        and RUN_NAME_RE.match(d.name)
    ]
    return sorted(runs, key=lambda p: p.name, reverse=True)


def _host_path_from_artifact(path_value: str, run_dir: Path, repo_root: Path) -> Path:
    p = Path(path_value).expanduser()
    if not p.is_absolute():
        return (run_dir / p).resolve()
    if p.exists():
        return p.resolve()
    for prefix, host_root in (
        ("/outputs", repo_root / "outputs"),
        ("/workspace", repo_root),
        ("/data", repo_root / "data"),
    ):
        try:
            rel = p.relative_to(prefix)
        except ValueError:
            continue
        return (host_root / rel).resolve()
    return p.resolve()


def _summary_points_to_eval_png(
    summary_json: Path,
    run_dir: Path,
    expected_eval_png: Path,
    repo_root: Path,
) -> tuple[bool, str | None]:
    if not summary_json.exists():
        return False, f"missing evaluation summary: {summary_json.name}"

    try:
        with open(summary_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        return False, f"invalid evaluation summary JSON ({type(exc).__name__}: {exc})"

    if not isinstance(payload, dict):
        return False, "evaluation summary is not a JSON object"

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        return False, "evaluation summary missing artifacts block"

    raw_png_path = artifacts.get("eval_visualization_png")
    if not isinstance(raw_png_path, str) or not raw_png_path:
        return False, "evaluation summary missing artifacts.eval_visualization_png"

    resolved = _host_path_from_artifact(raw_png_path, run_dir=run_dir, repo_root=repo_root)
    expected = expected_eval_png.resolve()
    if resolved != expected:
        return False, (
            "evaluation summary points to non-canonical eval PNG "
            f"({resolved} != {expected})"
        )
    if not resolved.exists():
        return False, f"evaluation summary eval PNG missing on disk: {resolved}"
    return True, None


def _safe_filename_component(raw: str) -> str:
    token = "".join(ch if (ch.isalnum() or ch in ("_", "-")) else "_" for ch in raw)
    return token.strip("_") or "item"


def _summary_points_to_xai_artifacts(
    summary_json: Path,
    run_dir: Path,
    expected_case_id: str,
    required_methods: set[str],
    repo_root: Path,
) -> tuple[bool, str | None]:
    if not summary_json.exists():
        return False, f"missing XAI summary: {summary_json.name}"

    try:
        with open(summary_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        return False, f"invalid XAI summary JSON ({type(exc).__name__}: {exc})"

    if not isinstance(payload, dict):
        return False, "XAI summary is not a JSON object"

    case_obj = payload.get("case")
    if not isinstance(case_obj, dict):
        return False, "XAI summary missing case block"
    got_case_id = case_obj.get("case_id")
    if got_case_id != expected_case_id:
        return False, f"XAI summary case mismatch ({got_case_id!r} != {expected_case_id!r})"

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        return False, "XAI summary missing artifacts block"
    method_errors_raw = payload.get("method_errors")
    method_errors = method_errors_raw if isinstance(method_errors_raw, dict) else {}

    required_artifacts: dict[str, tuple[str, ...]] = {
        "gradcam": ("gradcam_png",),
        "saliency": ("saliency_png",),
        "modality-ablation": ("modality_ablation_json", "modality_ablation_png"),
    }

    for method_name in sorted(required_methods):
        failed_msg = method_errors.get(method_name)
        if isinstance(failed_msg, str) and failed_msg.strip():
            continue

        for key in required_artifacts.get(method_name, ()):
            raw_path = artifacts.get(key)
            if not isinstance(raw_path, str) or not raw_path:
                return False, f"XAI summary missing artifacts.{key}"
            resolved = _host_path_from_artifact(raw_path, run_dir=run_dir, repo_root=repo_root)
            if not resolved.exists():
                return False, f"XAI artifact missing on disk ({key}): {resolved}"

    return True, None


def _compose_run_prefix(uid_gid: str | None, compose_files: list[str]) -> list[str]:
    cmd = ["docker", "compose"]
    for compose_file in compose_files:
        cmd.extend(["-f", compose_file])
    cmd.extend(["run", "--rm", "-T"])
    if uid_gid is not None:
        cmd.extend(["--user", uid_gid])
    cmd.append("trainer")
    return cmd


def _to_container_path(host_path: Path, repo_root: Path) -> str:
    p = host_path.resolve()
    mappings: tuple[tuple[Path, str], ...] = (
        ((repo_root / "outputs").resolve(), "/outputs"),
        ((repo_root / "data").resolve(), "/data"),
        (repo_root.resolve(), "/workspace"),
    )
    for host_root, container_root in mappings:
        try:
            rel = p.relative_to(host_root)
        except ValueError:
            continue
        rel_posix = rel.as_posix()
        if rel_posix in ("", "."):
            return container_root
        return f"{container_root}/{rel_posix}"
    raise ValueError(
        f"Cannot map host path into container mounts: {p} "
        f"(expected under {repo_root}, {repo_root / 'outputs'} or {repo_root / 'data'})"
    )


def _run_cmd(cmd: list[str], *, cwd: Path, env: dict[str, str] | None, dry_run: bool) -> None:
    print(f"$ {shlex.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run evaluation + visual export pipeline and regenerate report.md. "
            "Default mode processes runs missing required artifacts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--missing-only",
        action="store_true",
        help="Process only runs with missing or non-canonical artifacts (default behavior).",
    )
    mode_group.add_argument(
        "--all",
        action="store_true",
        help="Process all selected runs.",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Specific run directory to process. Can be provided multiple times.",
    )
    parser.add_argument(
        "--base-runs-dir",
        type=str,
        default="outputs/runs",
        help="Base runs directory used when --run is not provided.",
    )
    parser.add_argument(
        "--visualizations-dir",
        type=str,
        default="visualizations",
        help="Directory for centralized visual artifacts (PNG/GIF/HTML).",
    )
    parser.add_argument(
        "--xai-output-dir",
        type=str,
        default="visualizations/xai",
        help="Directory for explainability artifacts generated by explain_case.py.",
    )
    parser.add_argument(
        "--xai-method",
        choices=("gradcam", "saliency", "modality-ablation", "all"),
        default="all",
        help="Explanation method(s) passed to explain_case.py.",
    )
    parser.add_argument(
        "--xai-view-mode",
        choices=("panels", "overlay"),
        default="panels",
        help="Visualization layout passed to explain_case.py.",
    )
    parser.add_argument(
        "--xai-n-cols",
        type=int,
        default=12,
        help="Number of slices per XAI figure row passed to explain_case.py.",
    )
    parser.add_argument(
        "--xai-sw-batch-size",
        type=int,
        default=None,
        help="Optional sliding-window batch size override passed to explain_case.py.",
    )
    parser.add_argument(
        "--skip-xai",
        action="store_true",
        help="Skip XAI generation and XAI completeness checks.",
    )
    parser.add_argument(
        "--case-id",
        type=str,
        default="10726_1000742",
        help="Case ID used for per-run orbit GIF generation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Inference device passed to evaluate/visualize commands.",
    )
    parser.add_argument(
        "--report-output",
        type=str,
        default="report.md",
        help="Output markdown report path.",
    )
    parser.add_argument(
        "--execution",
        choices=("docker", "local"),
        default="docker",
        help=(
            "Execution backend. docker (default) uses docker compose trainer commands; "
            "local runs scripts with the current Python environment."
        ),
    )
    parser.add_argument(
        "--compose-file",
        action="append",
        default=[],
        help=(
            "Additional docker compose file(s), e.g. --compose-file compose.volta.yml. "
            "Applied in the provided order."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected runs and commands without executing them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    base_runs_dir = _abs(args.base_runs_dir)
    visualizations_dir = _abs(args.visualizations_dir)
    xai_output_dir = _abs(args.xai_output_dir)
    report_output = _abs(args.report_output)
    case_t2w_name = f"{args.case_id}_t2w.mha"
    case_t2w_host = (repo_root / "data" / "test_images" / case_t2w_name).resolve()
    xai_required_methods: set[str]
    if args.xai_method == "all":
        xai_required_methods = {"gradcam", "saliency", "modality-ablation"}
    else:
        xai_required_methods = {args.xai_method}

    all_run_dirs = _find_runs(base_runs_dir=base_runs_dir, run_args=args.run)
    if not all_run_dirs:
        raise RuntimeError("No runs found for processing.")

    plans: list[RunArtifacts] = []
    for run_dir in all_run_dirs:
        eval_png = (visualizations_dir / f"{run_dir.name}_eval_visualization.png").resolve()
        orbit_gif = (visualizations_dir / f"{run_dir.name}_{args.case_id}_t2w_orbit.gif").resolve()
        xai_summary = (
            xai_output_dir
            / f"{_safe_filename_component(run_dir.name)}_{_safe_filename_component(args.case_id)}_summary.json"
        ).resolve()
        summary_json = (run_dir / "evaluation_summary.json").resolve()

        missing_reasons: list[str] = []
        summary_ok, summary_reason = _summary_points_to_eval_png(
            summary_json=summary_json,
            run_dir=run_dir,
            expected_eval_png=eval_png,
            repo_root=repo_root,
        )
        if not summary_ok and summary_reason is not None:
            missing_reasons.append(summary_reason)
        eval_png_ok = eval_png.exists()
        if not eval_png_ok:
            missing_reasons.append(f"missing eval PNG: {eval_png.name}")
        orbit_ok = orbit_gif.exists()
        if not orbit_ok:
            missing_reasons.append(f"missing orbit GIF: {orbit_gif.name}")

        needs_xai = False
        if not args.skip_xai:
            xai_ok, xai_reason = _summary_points_to_xai_artifacts(
                summary_json=xai_summary,
                run_dir=run_dir,
                expected_case_id=args.case_id,
                required_methods=xai_required_methods,
                repo_root=repo_root,
            )
            needs_xai = not xai_ok
            if xai_reason is not None and not xai_ok:
                missing_reasons.append(xai_reason)

        plans.append(
            RunArtifacts(
                run_dir=run_dir,
                eval_png=eval_png,
                orbit_gif=orbit_gif,
                xai_summary=xai_summary,
                needs_eval=(not summary_ok) or (not eval_png_ok),
                needs_orbit=(not orbit_ok),
                needs_xai=needs_xai,
                missing_reasons=missing_reasons,
            )
        )

    missing_only = not args.all
    if missing_only:
        target_plans = [p for p in plans if p.missing_reasons]
    else:
        target_plans = plans

    print(f"Repository root      : {repo_root}")
    print(f"Execution backend    : {args.execution}")
    print(f"Total runs scanned   : {len(plans)}")
    print(f"Runs selected        : {len(target_plans)}")
    print(f"Visualizations dir   : {visualizations_dir}")
    print(f"XAI output dir       : {xai_output_dir}")
    print(f"XAI enabled          : {not args.skip_xai}")
    if not args.skip_xai:
        print(f"XAI method           : {args.xai_method}")
        print(f"XAI view mode        : {args.xai_view_mode}")
        print(f"XAI columns          : {args.xai_n_cols}")
        if args.xai_sw_batch_size is not None:
            print(f"XAI sw_batch_size    : {args.xai_sw_batch_size}")
    print(f"Report output        : {report_output}")
    print(f"Fixed GIF case       : {args.case_id}")
    print(f"Selection mode       : {'missing-only' if missing_only else 'all'}")

    skipped = [p for p in plans if p not in target_plans]
    if skipped and missing_only:
        print(f"Skipped complete runs: {len(skipped)}")

    if target_plans:
        print("\nSelected runs:")
        for plan in target_plans:
            if plan.missing_reasons:
                print(f"  - {plan.run_dir.name}")
                for reason in plan.missing_reasons:
                    print(f"      * {reason}")
            else:
                print(f"  - {plan.run_dir.name}")
    else:
        print("\nNo runs require processing under current selection.")

    if args.execution == "docker":
        uid_gid: str | None = None
        if hasattr(os, "getuid") and hasattr(os, "getgid"):
            uid_gid = f"{os.getuid()}:{os.getgid()}"
        compose_files = [str(_abs(cf)) for cf in args.compose_file]
        if compose_files:
            compose_files = [str((repo_root / "compose.yml").resolve()), *compose_files]
        docker_prefix = _compose_run_prefix(uid_gid=uid_gid, compose_files=compose_files)
        local_env: dict[str, str] | None = None
    else:
        docker_prefix = []
        local_env = os.environ.copy()
        local_env["PYTHONPATH"] = str(repo_root)

    if target_plans and not case_t2w_host.exists():
        raise FileNotFoundError(
            f"Required visualization case not found: {case_t2w_host}. "
            "Provide a different --case-id or ensure data/test_images is populated."
        )

    visualizations_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_xai:
        xai_output_dir.mkdir(parents=True, exist_ok=True)
    for idx, plan in enumerate(target_plans, start=1):
        print(f"\n[{idx}/{len(target_plans)}] Processing run: {plan.run_dir.name}")
        plan.eval_png.parent.mkdir(parents=True, exist_ok=True)

        if args.execution == "docker":
            run_c = _to_container_path(plan.run_dir, repo_root=repo_root)
            eval_png_c = _to_container_path(plan.eval_png, repo_root=repo_root)
            orbit_gif_c = _to_container_path(plan.orbit_gif, repo_root=repo_root)
            t2w_c = _to_container_path(case_t2w_host, repo_root=repo_root)
            xai_output_dir_c = _to_container_path(xai_output_dir, repo_root=repo_root)

            eval_cmd = docker_prefix + [
                "evaluate",
                "--run", run_c,
                "--device", args.device,
                "--vis-output", eval_png_c,
            ]
            vis_cmd = docker_prefix + [
                "visualize-3d",
                "--t2w", t2w_c,
                "--run", run_c,
                "--checkpoint", "best.pt",
                "--device", args.device,
                "--gif", orbit_gif_c,
            ]
            xai_cmd = docker_prefix + [
                "explain-case",
                "--run", run_c,
                "--t2w", t2w_c,
                "--checkpoint", "best.pt",
                "--method", args.xai_method,
                "--output-dir", xai_output_dir_c,
                "--device", args.device,
                "--n-cols", str(args.xai_n_cols),
                "--view-mode", args.xai_view_mode,
            ]
            if args.xai_sw_batch_size is not None:
                xai_cmd.extend(["--sw-batch-size", str(args.xai_sw_batch_size)])
        else:
            eval_cmd = [
                sys.executable,
                str(repo_root / "scripts" / "evaluate_checkpoint.py"),
                "--run", str(plan.run_dir),
                "--device", args.device,
                "--vis-output", str(plan.eval_png),
            ]
            vis_cmd = [
                sys.executable,
                str(repo_root / "scripts" / "visualize_3d.py"),
                "--t2w", str(case_t2w_host),
                "--run", str(plan.run_dir),
                "--checkpoint", "best.pt",
                "--device", args.device,
                "--gif", str(plan.orbit_gif),
            ]
            xai_cmd = [
                sys.executable,
                str(repo_root / "scripts" / "explain_case.py"),
                "--run", str(plan.run_dir),
                "--t2w", str(case_t2w_host),
                "--checkpoint", "best.pt",
                "--method", args.xai_method,
                "--output-dir", str(xai_output_dir),
                "--device", args.device,
                "--n-cols", str(args.xai_n_cols),
                "--view-mode", args.xai_view_mode,
            ]
            if args.xai_sw_batch_size is not None:
                xai_cmd.extend(["--sw-batch-size", str(args.xai_sw_batch_size)])

        run_eval = (not missing_only) or plan.needs_eval
        run_orbit = (not missing_only) or plan.needs_orbit
        run_xai = (not args.skip_xai) and ((not missing_only) or plan.needs_xai)

        if run_eval:
            _run_cmd(eval_cmd, cwd=repo_root, env=local_env, dry_run=args.dry_run)
        else:
            print("  - Skipping evaluate (artifact already complete)")

        if run_orbit:
            _run_cmd(vis_cmd, cwd=repo_root, env=local_env, dry_run=args.dry_run)
        else:
            print("  - Skipping visualize-3d GIF (artifact already complete)")

        if run_xai:
            _run_cmd(xai_cmd, cwd=repo_root, env=local_env, dry_run=args.dry_run)
        elif not args.skip_xai:
            print("  - Skipping explain-case (XAI artifacts already complete)")

    if args.execution == "docker":
        report_cmd = docker_prefix + [
            "report-runs",
            "--base-dir", _to_container_path(base_runs_dir, repo_root=repo_root),
            "--visualizations-dir", _to_container_path(visualizations_dir, repo_root=repo_root),
            "--xai-dir", _to_container_path(xai_output_dir, repo_root=repo_root),
            "--output", _to_container_path(report_output, repo_root=repo_root),
        ]
    else:
        report_cmd = [
            sys.executable,
            str(repo_root / "scripts" / "report_runs.py"),
            "--base-dir", str(base_runs_dir),
            "--visualizations-dir", str(visualizations_dir),
            "--xai-dir", str(xai_output_dir),
            "--output", str(report_output),
        ]

    print("\nRegenerating run report...")
    _run_cmd(report_cmd, cwd=repo_root, env=local_env, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"Report pipeline complete: {report_output}")


if __name__ == "__main__":
    main()
