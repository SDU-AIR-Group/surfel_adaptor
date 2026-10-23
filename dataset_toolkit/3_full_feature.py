import argparse
import traceback
from pathlib import Path

import numpy as np

from tools.common import (
    add_common_arguments, append_journal, dataset_relative, ensure_layout,
    patch_rows, print_summary, reconcile_metadata, resolve_dataset_path, stage_log_dir, sync_due, tqdm,
)
from tools.extract_full_feature import (
    feature_archive, forward_batch, is_oom_error, load_sonata, prepare_patch, write_archive,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--sonata-checkpoint", type=Path, required=True,
                        help="Sonata model checkpoint file.")
    parser.add_argument("--sonata-root", type=Path, default=None,
                        help="Sonata package directory or its parent containing extensions/sonata.")
    parser.add_argument("--model", default="sonata", help="Sonata model name.")
    parser.add_argument("--terrain-scale", type=float, default=25.0,
                        help="Patch half-width used to normalize geometry (metres).")
    parser.add_argument("--voxel-size", type=int, default=128,
                        help="Output feature voxel-grid resolution.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Patches per Sonata forward pass; default 1 for 900k-point inputs.")
    return parser.parse_args()


def batches(items: list[tuple[str, dict[str, str], Path]], size: int):
    for index in range(0, len(items), size):
        yield items[index:index + size]


def valid_feature(path: Path) -> bool:
    """A completed feature is a readable archive with the two downstream arrays."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with np.load(path) as archive:
            return {"indices", "patchtokens"}.issubset(archive.files)
    except (OSError, ValueError):
        return False


def main() -> None:
    args = parse_args()
    if args.workers != 1:
        raise SystemExit("Step3 owns one CUDA model; use --batch-size instead of --workers.")
    if args.batch_size < 1 or args.terrain_scale <= 0 or not 0 < args.voxel_size <= 256:
        raise SystemExit("--batch-size and --terrain-scale must be positive; --voxel-size must be in 1..256")
    root = args.dataset_root
    ensure_layout(root)
    rows = reconcile_metadata(root)
    patches = patch_rows(rows)
    pending = [
        (patch_id, row, root / "full_features" / f"full_feat_{patch_id}.npz")
        for patch_id, row in patches.items()
        if args.overwrite or not valid_feature(root / "full_features" / f"full_feat_{patch_id}.npz")
    ]
    skipped = 0 if args.overwrite else len(patches) - len(pending)
    if args.dry_run:
        print_summary(root, "step3", processed=len(pending), completed=len(pending), skipped=skipped, failed=0)
        return
    if not pending:
        print_summary(root, "step3", processed=0, completed=0, skipped=skipped, failed=0)
        return

    if not args.sonata_checkpoint.is_file():
        print(f"[step3] checkpoint does not exist: {args.sonata_checkpoint}")
        print_summary(root, "step3", processed=len(pending), completed=0, skipped=skipped,
                      failed=len(pending), failed_ids=[patch_id for patch_id, _row, _target in pending])
        return

    logs = stage_log_dir(root, "step3", args.log_dir)
    try:
        runtime = load_sonata(
            toolkit_root=args.toolkit_root.resolve(), checkpoint=args.sonata_checkpoint.resolve(),
            sonata_root=args.sonata_root.resolve() if args.sonata_root else None, model_name=args.model,
        )
    except Exception as exc:
        print(f"[step3] failed to load Sonata: {exc}")
        print_summary(root, "step3", processed=len(pending), completed=0, skipped=skipped,
                      failed=len(pending), failed_ids=[patch_id for patch_id, _row, _target in pending])
        return
    completed = failed = 0
    failed_ids = []

    def fail(patch_id: str, exc: BaseException) -> None:
        nonlocal failed
        failed += 1
        failed_ids.append(patch_id)
        (logs / f"full_feat_{patch_id}.log").write_text(traceback.format_exc() if exc.__traceback__ else str(exc),
                                                          encoding="utf-8")

    def save(patch_id: str, target: Path, raw_points, features) -> None:
        nonlocal completed
        archive = feature_archive(raw_points, features, terrain_scale=args.terrain_scale,
                                  voxel_size=args.voxel_size, torch=runtime.torch)
        write_archive(target, archive)
        append_journal(root, "step3", {"kind": "patch", "key": patch_id, "updates": {
            "full_feat_id": target.stem, "full_feat_path": dataset_relative(root, target),
        }})
        completed += 1
        if sync_due(completed, args.sync_every):
            reconcile_metadata(root)

    with tqdm(total=len(pending), desc="Step3 full features", unit="patch") as progress:
        for group in batches(pending, args.batch_size):
            prepared = []
            for task in group:
                patch_id, row, target = task
                patch = resolve_dataset_path(root, row["patch_file"])
                try:
                    prepared.append((task, prepare_patch(patch, runtime)))
                except Exception as exc:
                    fail(patch_id, exc)
                    progress.update(1)
            if not prepared:
                continue

            def run(items) -> None:
                outputs = forward_batch(runtime, [prepared_item for _, prepared_item in items])
                for ((patch_id, _row, target), _prepared), (raw_points, features) in zip(items, outputs):
                    try:
                        save(patch_id, target, raw_points, features)
                    except Exception as exc:
                        fail(patch_id, exc)
                    progress.update(1)

            try:
                run(prepared)
            except Exception as exc:
                if len(prepared) == 1 or not is_oom_error(exc):
                    for (patch_id, _row, _target), _prepared in prepared:
                        fail(patch_id, exc)
                        progress.update(1)
                else:
                    runtime.torch.cuda.empty_cache()
                    for item in prepared:
                        try:
                            run([item])
                        except Exception as retry_exc:
                            fail(item[0][0], retry_exc)
                            progress.update(1)
    reconcile_metadata(root)
    print_summary(root, "step3", processed=len(pending), completed=completed, skipped=skipped,
                  failed=failed, failed_ids=failed_ids)


if __name__ == "__main__":
    main()
