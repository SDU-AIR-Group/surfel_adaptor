import argparse
import os
import shutil
import tempfile
from pathlib import Path

from tools.common import add_common_arguments, ensure_layout, parallel_jobs, print_summary, run_command, stable_token, stage_log_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--aux-input", type=Path, required=True, help="Root containing raw auxiliary .glb files.")
    return parser.parse_args()


def preprocess(task: tuple[Path, Path, argparse.Namespace, Path, Path]) -> tuple[str, bool]:
    """Preprocess one auxiliary asset in an isolated Blender workspace."""
    source, target, args, tool, logs = task
    try:
        with tempfile.TemporaryDirectory(prefix="step0_", dir=args.dataset_root / "temp") as raw_work:
            work = Path(raw_work)
            input_dir, output_dir = work / "input", work / "output"
            input_dir.mkdir()
            try:
                os.symlink(source.resolve(), input_dir / source.name)
            except OSError:
                shutil.copy2(source, input_dir / source.name)
            rc = run_command([args.blender, "-b", "-P", str(tool), "--", "--input_folder", str(input_dir),
                              "--output_folder", str(output_dir), "--save_mesh"], logs / f"{stable_token(source.resolve())}.log")
            produced = output_dir / source.name
            if rc != 0 or not produced.is_file():
                return source.stem, False
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                target.unlink()
            shutil.move(str(produced), target)
        return source.stem, True
    except Exception:
        return source.stem, False


def main() -> None:
    args = parse_args()
    if args.workers < 1 or not args.aux_input.is_dir():
        raise SystemExit("--workers must be positive and --aux-input must be a directory")
    root = args.dataset_root
    ensure_layout(root)
    tool = args.toolkit_root / "tools" / "0aux_preproc.py"
    if not tool.is_file():
        raise SystemExit(f"Auxiliary preprocessor not found: {tool}")
    logs = stage_log_dir(root, "step0", args.log_dir)
    output_root = root / "0_aux_models_preproc"
    sources = sorted(path for path in args.aux_input.rglob("*.glb") if path.is_file())
    tasks, skipped = [], 0
    for source in sources:
        target = output_root / f"aux_{stable_token(source.resolve())}_{source.name}"
        if target.is_file() and target.stat().st_size > 0 and not args.overwrite:
            skipped += 1
        else:
            tasks.append((source, target, args, tool, logs))
    if args.dry_run:
        print_summary(root, "step0", processed=len(tasks), completed=len(tasks), skipped=skipped, failed=0)
        return
    completed = failed = 0
    failed_ids = []
    for result, _ in parallel_jobs(tasks, preprocess, workers=args.workers, desc="Step0 auxiliary preprocessing", unit="model"):
        asset_id, success = result or ("unknown", False)
        if success:
            completed += 1
        else:
            failed += 1
            failed_ids.append(asset_id)
    print_summary(root, "step0", processed=len(tasks), completed=completed, skipped=skipped,
                  failed=failed, failed_ids=failed_ids)


if __name__ == "__main__":
    main()
