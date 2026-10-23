import argparse
import os
import shutil
import tempfile
from pathlib import Path

from tools.common import (
    add_common_arguments, append_journal, dataset_relative, ensure_layout, existing_file,
    parallel_jobs, print_summary, reconcile_metadata, resolve_dataset_path, run_command, stage_log_dir, sync_due,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--max-bound", type=float, default=25.0,
                        help="Half-width of the normalized voxel volume (metres).")
    parser.add_argument("--voxel-size", type=int, default=128,
                        help="Voxel-grid resolution per axis.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.dataset_root
    ensure_layout(root)
    voxelizer = args.toolkit_root / "voxel.py"
    if not voxelizer.is_file():
        raise SystemExit(f"Voxelizer not found: {voxelizer}")
    rows = reconcile_metadata(root)
    eligible = [row for row in rows if row.get("scan_id") and existing_file(root, row, "scan_path")]
    pending = [row for row in eligible if args.overwrite or not existing_file(root, row, "voxel_path")]
    logs = stage_log_dir(root, "step3", args.log_dir)
    completed, failed = 0, 0
    skipped = 0 if args.overwrite else len(eligible) - len(pending)
    if args.dry_run:
        print_summary(root, "voxel-deprecated", processed=len(pending), completed=len(pending), skipped=skipped, failed=0)
        return

    def voxelize(row: dict[str, str]) -> tuple[str, Path] | None:
        scan_id = row["scan_id"]
        suffix = scan_id.rsplit("_", 1)[-1]
        target = root / "voxels" / f"voxel_{suffix}.ply"
        if args.overwrite:
            target.unlink(missing_ok=True)
        with tempfile.TemporaryDirectory(prefix="step3_", dir=root / "temp") as raw_work:
            work = Path(raw_work)
            scans = work / "scans"
            scans.mkdir()
            source = resolve_dataset_path(root, row["scan_path"])
            os.symlink(source.resolve(), scans / "scan.npy")
            command = [args.python, str(voxelizer), "--input_dir", str(work), "--output_dir", str(work),
                       "--max_bound", str(args.max_bound), "--voxel_size", str(args.voxel_size)]
            rc = run_command(command, logs / f"{scan_id}.log")
            produced = work / "voxels" / "voxel.ply"
            if rc != 0 or not produced.is_file():
                return None
            shutil.move(str(produced), target)
        return scan_id, target

    for result, _ in parallel_jobs(pending, voxelize, workers=args.workers, desc="Step3 voxelizing", unit="scan"):
        if result is None:
            failed += 1
            continue
        scan_id, target = result
        append_journal(root, "step3", {"kind": "scan", "key": scan_id, "updates": {
            "voxel_id": target.stem, "voxel_path": dataset_relative(root, target),
        }})
        completed += 1
        if sync_due(completed, args.sync_every):
            reconcile_metadata(root)
    reconcile_metadata(root)
    print_summary(root, "voxel-deprecated", processed=len(pending), completed=completed,
                  skipped=skipped, failed=failed)


if __name__ == "__main__":
    main()
