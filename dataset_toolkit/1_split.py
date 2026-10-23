import argparse
import csv
import json
import random
import shutil
import tempfile
import time
from pathlib import Path

try:
    from tools.common import add_common_arguments, ensure_layout, print_summary, run_command, stage_log_dir, tqdm
except ModuleNotFoundError:
    from dataset_toolkit.tools.common import add_common_arguments, ensure_layout, print_summary, run_command, stage_log_dir, tqdm


def timestamp_seed() -> int:
    """Return the last five digits of the current Unix timestamp."""
    return int(str(int(time.time()))[-5:])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, seed=False)
    parser.add_argument("--terrain-input", type=Path, required=True,
                        help="Directory laid out as <terrain_tag>/model.glb.")
    parser.add_argument("--lidar-height", type=float, default=0.83,
                        help="LiDAR height used to align each patch (metres).")
    parser.add_argument("--patch-aspect-max", type=float, default=1.25,
                        help="Largest allowed horizontal patch aspect ratio.")
    parser.add_argument("--patch-min-edge", type=float, default=24.5,
                        help="Minimum horizontal patch edge after splitting (metres).")
    parser.add_argument("--scale_augment", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply independent x/y/z scale factors after 100 m normalization (default: enabled).")
    parser.add_argument("--patch_size_meters", type=float, default=25.0,
                        help="Fixed grid-cell size in metres (default: 25).")
    parser.add_argument("--xy_scale_range", nargs=2, type=float, default=(0.5, 1.5), metavar=("MIN", "MAX"),
                        help="Independent uniform x/y augmentation range (default: 0.5 1.5).")
    parser.add_argument("--z_scale_range", nargs=2, type=float, default=(0.8, 1.2), metavar=("MIN", "MAX"),
                        help="Independent uniform z augmentation range (default: 0.8 1.2).")
    parser.add_argument("--patch_num", type=int, default=1,
                        help="Generate valid patches until this total is reached (default: 1).")
    parser.add_argument("--offroad_prob", type=float, default=0.7,
                        help="Probability of choosing offroad when both terrain types are available (default: 0.7).")
    parser.add_argument("--max-failed-attempts", type=int, default=20,
                        help="Stop after this many consecutive failed/empty split attempts (default: 20).")
    return parser.parse_args()


def terrain_models(root: Path) -> dict[str, list[tuple[str, Path]]]:
    groups = {"offroad": [], "city": []}
    for model in sorted(root.glob("*/model.glb")):
        terrain_type = model.parent.name.split("_", 1)[0].lower()
        if terrain_type in groups:
            groups[terrain_type].append((model.parent.name, model))
    return groups


def split_settings(args: argparse.Namespace) -> dict[str, object]:
    return {
        "lidar_height": args.lidar_height,
        "patch_aspect_max": args.patch_aspect_max,
        "patch_min_edge": args.patch_min_edge,
        "scale_augment": args.scale_augment,
        "patch_size_meters": args.patch_size_meters,
        "xy_scale_range": list(args.xy_scale_range),
        "z_scale_range": list(args.z_scale_range),
    }


def split_matches_settings(output: Path, args: argparse.Namespace) -> bool:
    marker = output / "split_complete.json"
    try:
        return marker.is_file() and json.loads(marker.read_text(encoding="utf-8")) == split_settings(args)
    except json.JSONDecodeError:
        return False


def valid_patch_count(output: Path, args: argparse.Namespace) -> int:
    if not split_matches_settings(output, args):
        return 0
    return sum(path.is_file() and path.stat().st_size > 0 for path in (output / "patches").glob("patch_*.glb"))


def choose_terrain(groups: dict[str, list[tuple[str, Path]]], rng: random.Random,
                   offroad_prob: float) -> tuple[str, Path] | None:
    available = [kind for kind, models in groups.items() if models]
    if not available:
        return None
    kind = available[0] if len(available) == 1 else ("offroad" if rng.random() < offroad_prob else "city")
    return rng.choice(groups[kind])


def next_patch_index(patches_dir: Path) -> int:
    indices = []
    for patch in patches_dir.glob("patch_*.glb"):
        try:
            indices.append(int(patch.stem.removeprefix("patch_")))
        except ValueError:
            continue
    return max(indices, default=-1) + 1


def filter_patches(patches: list[Path], dimensions: dict[str, list[float] | None],
                   args: argparse.Namespace) -> list[tuple[Path, dict[str, object]]]:
    """Delete rejected temporary patches and return the valid ones with their report rows."""
    kept = []
    for patch in patches:
        try:
            dx, dy, dz = map(float, dimensions[patch.name] or [])
            min_edge = min(dx, dy)
            aspect = max(dx, dy) / max(min_edge, 1e-9)
            valid = aspect <= args.patch_aspect_max and min_edge >= args.patch_min_edge
        except (KeyError, TypeError, ValueError):
            dx = dy = dz = aspect = min_edge = float("nan")
            valid = False
        row = {"patch_file": patch.name, "dx": dx, "dy": dy, "dz": dz,
               "aspect": aspect, "min_edge": min_edge, "kept": valid}
        if valid:
            kept.append((patch, row))
        else:
            patch.unlink(missing_ok=True)
    return kept


def append_filter_report(report: Path, rows: list[dict[str, object]]) -> None:
    fields = ["patch_file", "dx", "dy", "dz", "aspect", "min_edge", "kept"]
    write_header = not report.exists() or report.stat().st_size == 0
    with report.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def validate(args: argparse.Namespace, bounds_adapter: Path) -> None:
    if not args.terrain_input.is_dir():
        raise SystemExit(f"--terrain-input is not a directory: {args.terrain_input}")
    if args.patch_num < 1 or args.patch_size_meters <= 0 or args.max_failed_attempts < 1:
        raise SystemExit("--patch_num, --patch_size_meters, and --max-failed-attempts must be positive")
    if not 0.0 <= args.offroad_prob <= 1.0:
        raise SystemExit("--offroad_prob must be between 0 and 1")
    for option, values in (("--xy_scale_range", args.xy_scale_range), ("--z_scale_range", args.z_scale_range)):
        if values[0] <= 0 or values[0] > values[1]:
            raise SystemExit(f"{option} must be positive and ordered MIN MAX")
    if not bounds_adapter.is_file():
        raise SystemExit(f"Patch bounds adapter not found: {bounds_adapter}")


def main() -> None:
    args = parse_args()
    root = args.dataset_root
    toolkit = Path(__file__).resolve().parent
    splitter = toolkit / "tools" / "terrain_split.py"
    bounds_adapter = toolkit / "tools" / "patch_bounds.py"
    validate(args, bounds_adapter)
    ensure_layout(root)

    groups = terrain_models(args.terrain_input)
    if not any(groups.values()):
        raise SystemExit("No <offroad|city>_<tag>/model.glb terrain models were found.")
    outputs = [root / "1_terrain_splitting" / tag for models in groups.values() for tag, _ in models]
    if args.overwrite:
        if not args.dry_run:
            for output in outputs:
                shutil.rmtree(output, ignore_errors=True)
        current = 0
    else:
        current = sum(valid_patch_count(output, args) for output in outputs)
    if current >= args.patch_num:
        print(f"[step1] patch target already met: {current}/{args.patch_num}")
        print_summary(root, "step1", processed=0, completed=0, skipped=current, failed=0)
        return

    logs = stage_log_dir(root, "step1", args.log_dir)
    completed = skipped = failed = 0
    failed_ids = []
    failed_attempts = 0
    rng = random.Random(timestamp_seed())
    progress = tqdm(total=args.patch_num, initial=min(current, args.patch_num), desc="Step1 valid patches", unit="patch")
    try:
        while current < args.patch_num:
            selected = choose_terrain(groups, rng, args.offroad_prob)
            if selected is None:
                print("[step1] no unprocessed offroad/city terrain models remain")
                break
            terrain_tag, model = selected
            output = root / "1_terrain_splitting" / terrain_tag
            patches_dir = output / "patches"
            marker = output / "split_complete.json"
            if marker.exists() and not args.overwrite and not split_matches_settings(output, args):
                message = f"{terrain_tag} was split with different settings; use --overwrite to rebuild it."
                if args.dry_run:
                    print(f"Would rebuild {model} with changed split settings (requires --overwrite).")
                    completed += 1
                    continue
                raise SystemExit(message)
            if args.dry_run:
                print(f"Would split {model} -> {patches_dir}")
                completed += 1
                progress.update(args.patch_num - current)
                current = args.patch_num
                continue

            if output.exists() and not marker.exists():
                shutil.rmtree(output)
            output.mkdir(parents=True, exist_ok=True)
            patches_dir.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="step1_", dir=root / "temp") as raw_work:
                work = Path(raw_work)
                generated = work / "generated"
                command = [args.blender, "-b", "-P", str(splitter), "--", "--object", str(model),
                           "--output_folder", str(generated), "--lidar_height", str(args.lidar_height),
                           "--patch_size_meters", str(args.patch_size_meters), "--xy_scale_range", *map(str, args.xy_scale_range),
                           "--z_scale_range", *map(str, args.z_scale_range),
                           "--scale_augment" if args.scale_augment else "--no-scale_augment"]
                rc = run_command(command, logs / f"{terrain_tag}.log")
                patches = sorted((generated / "patches").glob("patch_*.glb")) if rc == 0 else []
                bounds_file = work / "patch_bounds.json"
                bounds_command = [args.blender, "-b", "-P", str(bounds_adapter), "--", "--patch-dir",
                                  str(generated / "patches"), "--output-json", str(bounds_file)]
                if not patches or run_command(bounds_command, logs / f"{terrain_tag}_bounds.log") != 0 or not bounds_file.is_file():
                    failed += 1
                    failed_ids.append(terrain_tag)
                    failed_attempts += 1
                    print(f"[step1] unable to generate or measure patches for {terrain_tag}")
                    if failed_attempts >= args.max_failed_attempts:
                        raise RuntimeError("Too many consecutive failed/empty split attempts.")
                    continue
                dimensions = json.loads(bounds_file.read_text(encoding="utf-8"))
                kept = filter_patches(patches, dimensions, args)

                next_index = next_patch_index(patches_dir)
                report_rows = []
                merged_bounds = {}
                for patch, row in kept:
                    target = patches_dir / f"patch_{next_index}.glb"
                    next_index += 1
                    shutil.move(str(patch), target)
                    row["patch_file"] = target.name
                    report_rows.append(row)
                    merged_bounds[target.name] = dimensions.get(patch.name)
            if report_rows:
                append_filter_report(output / "patch_filter.csv", report_rows)
            bounds_path = output / "patch_bounds.json"
            if bounds_path.is_file():
                try:
                    merged_bounds = {**json.loads(bounds_path.read_text(encoding="utf-8")), **merged_bounds}
                except json.JSONDecodeError:
                    pass
            bounds_path.write_text(json.dumps(merged_bounds, indent=2), encoding="utf-8")
            marker.write_text(json.dumps(split_settings(args), indent=2), encoding="utf-8")
            completed += 1
            added = len(kept)
            current += added
            progress.update(added)
            failed_attempts = 0 if added else failed_attempts + 1
            if not added:
                print(f"[step1] all patches rejected for {terrain_tag}")
                failed += 1
                failed_ids.append(terrain_tag)
                if failed_attempts >= args.max_failed_attempts:
                    raise RuntimeError("Too many consecutive failed/empty split attempts.")
    finally:
        progress.close()
    print_summary(root, "step1", processed=completed + failed, completed=completed, skipped=skipped,
                  failed=failed, failed_ids=failed_ids)


if __name__ == "__main__":
    main()
