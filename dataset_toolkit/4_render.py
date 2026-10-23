import argparse
import shutil
import tempfile
from pathlib import Path

from tools.common import (
    add_common_arguments, append_journal, dataset_relative, ensure_layout, existing_render,
    parallel_jobs, patch_rows, print_summary, reconcile_metadata, resolve_dataset_path, run_command, stable_token, stage_log_dir, sync_due,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--terrain-scale", type=float, default=25.0,
                        help="Patch half-width used to place cameras (metres).")
    parser.add_argument("--num-views", type=int, default=150,
                        help="Total rendered camera views per patch.")
    parser.add_argument(
        "--local-view-num", type=int, default=50,
        help="Local segment-view count; defaults to half of --num-views.",
    )
    parser.add_argument("--resolution", type=int, default=512, help="Square image resolution in pixels.")
    parser.add_argument("--yaw-min", type=float, default=0.0, help="Minimum camera yaw in degrees.")
    parser.add_argument("--yaw-max", type=float, default=360.0, help="Maximum camera yaw in degrees.")
    parser.add_argument("--pitch-min", type=float, default=-30.0, help="Minimum camera pitch in degrees.")
    parser.add_argument("--pitch-max", type=float, default=30.0, help="Maximum camera pitch in degrees.")
    parser.add_argument("--radius-min", type=float, default=None, help="Optional minimum camera radius in metres.")
    parser.add_argument("--radius-max", type=float, default=None, help="Optional maximum camera radius in metres.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_views < 1 or not 0 <= args.local_view_num < args.num_views:
        raise SystemExit("--local-view-num must satisfy 0 <= local-view-num < --num-views")
    root = args.dataset_root
    ensure_layout(root)
    tool = Path(__file__).with_name("tools") / "render_model.py"
    if not tool.is_file():
        raise SystemExit(f"Render tool missing: {tool}")
    rows = reconcile_metadata(root)
    patches = patch_rows(rows)
    pending = [(patch_id, row, root / "renders" / f"render_{patch_id}")
               for patch_id, row in patches.items()
               if args.overwrite or not existing_render(root, row)]
    logs = stage_log_dir(root, "step4", args.log_dir)
    completed, failed = 0, 0
    skipped = 0 if args.overwrite else len(patches) - len(pending)
    if args.dry_run:
        print_summary(root, "step4", processed=len(pending), completed=len(pending), skipped=skipped, failed=0)
        return

    def render(task: tuple[str, dict[str, str], Path]) -> tuple[str, Path | None]:
        patch_id, row, target = task
        try:
            patch = resolve_dataset_path(root, row["patch_file"])
            if not patch.is_file():
                return patch_id, None
            with tempfile.TemporaryDirectory(prefix="step4_", dir=root / "temp") as raw_work:
                work = Path(raw_work)
                produced = work / "render"
                render_seed = int(stable_token(args.seed, patch_id, length=8), 16)
                command = [args.python, str(tool.resolve()), "--blender", str(Path(args.blender).resolve()),
                           "--toolkit-root", str(args.toolkit_root.resolve()), "--object-path", str(patch.resolve()),
                           "--output-folder", str(produced.resolve()), "--terrain-scale", str(args.terrain_scale),
                           "--num-views", str(args.num_views), "--resolution", str(args.resolution),
                           "--yaw-min", str(args.yaw_min), "--yaw-max", str(args.yaw_max),
                           "--pitch-min", str(args.pitch_min), "--pitch-max", str(args.pitch_max),
                           "--seed", str(render_seed)]
                if args.local_view_num is not None:
                    command.extend(["--local-view-num", str(args.local_view_num)])
                if args.radius_min is not None:
                    command.extend(["--radius-min", str(args.radius_min)])
                if args.radius_max is not None:
                    command.extend(["--radius-max", str(args.radius_max)])
                rc = run_command(command, logs / f"render_{patch_id}.log")
                if rc != 0 or not (produced / "transforms.json").is_file():
                    return patch_id, None
                if target.exists():
                    shutil.rmtree(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(produced), target)
            return patch_id, target
        except Exception:
            return patch_id, None

    failed_ids = []
    for result, _ in parallel_jobs(pending, render, workers=args.workers, desc="Step4 rendering", unit="patch"):
        patch_id, target = result or ("unknown", None)
        if target is None:
            failed += 1
            failed_ids.append(patch_id)
            continue
        append_journal(root, "step4", {"kind": "patch", "key": patch_id, "updates": {
            "render_subdir_id": target.name, "render_dir": dataset_relative(root, target),
        }})
        completed += 1
        if sync_due(completed, args.sync_every):
            reconcile_metadata(root)
    reconcile_metadata(root)
    print_summary(root, "step4", processed=len(pending), completed=completed, skipped=skipped,
                  failed=failed, failed_ids=failed_ids)


if __name__ == "__main__":
    main()
