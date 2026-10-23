import argparse
import os
import shutil
import tempfile
import time
from pathlib import Path

from tools.common import (
    add_common_arguments, append_journal, dataset_relative, ensure_layout, existing_file,
    parallel_jobs, print_summary, reconcile_metadata, run_command, stable_token, stage_log_dir, sync_due,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--target-samples", type=int, required=True,
                        help="Target number of successful scan rows in metadata.")
    parser.add_argument("--scale-factor", nargs=6, type=float, default=[0.5, 10.0, 0.5, 10.0, 0.5, 10.0],
                        help="Min/max random scale ranges for auxiliary x, y, and z.")
    parser.add_argument("--random-pose-rate", type=float, default=0.3,
                        help="Probability of using a random scan path.")
    parser.add_argument("--lidar-height", type=float, default=0.83,
                        help="LiDAR mounting height above the patch (metres).")
    return parser.parse_args()


def discover_patches(root: Path) -> list[tuple[str, int, Path]]:
    items: list[tuple[str, int, Path]] = []
    base = root / "1_terrain_splitting"
    if not base.is_dir():
        return items
    for terrain in sorted(path for path in base.iterdir() if path.is_dir()):
        for patch in sorted((terrain / "patches").glob("patch_*.glb")):
            try:
                index = int(patch.stem.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            items.append((terrain.name, index, patch))
    return items


def main() -> None:
    args = parse_args()
    if args.target_samples < 1:
        raise SystemExit("--target-samples must be positive")
    root = args.dataset_root
    ensure_layout(root)
    source = args.toolkit_root / "tools" / "aux_scanning.py"
    aux_dir = root / "0_aux_models_preproc"
    if not source.is_file():
        raise SystemExit("Step2 scanner is missing.")
    if not any(aux_dir.glob("*.glb")):
        raise SystemExit(f"No preprocessed auxiliary GLB files in {aux_dir}; run Step0 first.")
    patches = discover_patches(root)
    if not patches:
        raise SystemExit("No valid terrain patches; run Step1 first.")
    rows = reconcile_metadata(root)
    valid = [row for row in rows if row.get("scan_id") and existing_file(root, row, "scan_path")]
    successful_patch_ids = {row["patch_id"] for row in valid if row.get("patch_id")}
    uncovered_patches = [item for item in patches if f"{item[0]}_p{item[1]}" not in successful_patch_ids]
    current = len(valid)
    if current >= args.target_samples and not uncovered_patches:
        print(f"[step2] target already met: {current}/{args.target_samples}")
        print_summary(root, "step2", processed=0, completed=0, skipped=current, failed=0)
        return
    logs = stage_log_dir(root, "step2", args.log_dir)
    # ``target-samples`` is a minimum.  A resumed run must also repair
    # patches whose first scan failed, otherwise later patch-level stages
    # silently omit them forever.
    needed = max(args.target_samples - current, len(uncovered_patches))
    journal = root / "records" / "step2.jsonl"
    attempt_base = sum(1 for line in journal.open(encoding="utf-8") if line.strip()) if journal.is_file() else 0
    run_nonce = time.time_ns()
    tasks = []
    for offset in range(needed):
        # Metadata contains only unique scan IDs; the journal count also
        # includes prior successful attempts that later collided.  Starting
        # after it guarantees a resumed invocation never reuses a scan ID.
        attempt = attempt_base + offset + 1
        if offset < len(uncovered_patches):
            terrain_tag, patch_index, patch = uncovered_patches[offset]
        else:
            terrain_tag, patch_index, patch = patches[(attempt - 1) % len(patches)]
        patch_id = f"{terrain_tag}_p{patch_index}"
        suffix = stable_token(args.seed, run_nonce, terrain_tag, patch_index, attempt, length=10)
        scan_id = f"scan_{patch_id}_{suffix}"
        tasks.append((terrain_tag, patch_index, patch, patch_id, scan_id,
                      root / "scans" / f"{scan_id}.npy", int(stable_token(args.seed, run_nonce, patch_id, attempt, length=8), 16)))
    if args.dry_run:
        print_summary(root, "step2", processed=len(tasks), completed=len(tasks), skipped=0, failed=0)
        return

    def scan(task):
        terrain_tag, patch_index, patch, patch_id, scan_id, scan_target, derived_seed = task
        try:
            with tempfile.TemporaryDirectory(prefix="step2_", dir=root / "temp") as raw_work:
                work = Path(raw_work)
                terrain_dir = work / "terrain"
                terrain_dir.mkdir()
                local_patch = terrain_dir / patch.name
                try:
                    os.symlink(patch.resolve(), local_patch)
                except OSError:
                    shutil.copy2(patch, local_patch)
                command = [args.blender, "-b", "-P", str(source), "--",
                           "--aux-model-dir", str(aux_dir), "--terrain-model-dir", str(terrain_dir),
                           "--output-folder", str(work), "--scale-factor", *map(str, args.scale_factor),
                           "--random-pose-rate", str(args.random_pose_rate), "--lidar-height", str(args.lidar_height),
                           "--seed", str(derived_seed)]
                rc = run_command(command, logs / f"{scan_id}.log")
                produced, meta_path = work / "scans" / "scan.npy", work / "scans" / "scan_meta.json"
                if rc != 0 or not produced.is_file():
                    return scan_id, None
                shutil.move(str(produced), scan_target)
                if meta_path.is_file():
                    shutil.move(str(meta_path), logs / f"{scan_id}.json")
            return scan_id, (terrain_tag, patch, patch_id, scan_id, scan_target)
        except Exception:
            return scan_id, None

    completed = failed = 0
    failed_ids = []
    for result, _ in parallel_jobs(tasks, scan, workers=args.workers, desc="Step2 scans", unit="scan",
                                   initial=current, total=max(args.target_samples, current + needed)):
        scan_id, payload = result or ("unknown", None)
        if payload is None:
            failed += 1
            failed_ids.append(scan_id)
            continue
        terrain_tag, patch, patch_id, scan_id, scan_target = payload
        row = {
            "terrain_source": terrain_tag.split("_", 1)[0], "terrain_tag": terrain_tag,
            "patch_file": dataset_relative(root, patch),
            "patch_id": patch_id, "scan_id": scan_id, "scan_path": dataset_relative(root, scan_target),
        }
        append_journal(root, "step2", {"kind": "new_scan", "row": row})
        completed += 1
        if sync_due(completed, args.sync_every):
            reconcile_metadata(root)
    reconcile_metadata(root)
    print_summary(root, "step2", processed=len(tasks), completed=completed, skipped=0,
                  failed=failed, failed_ids=failed_ids)


if __name__ == "__main__":
    main()
