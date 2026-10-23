import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from tools.common import (
    add_common_arguments, append_journal, dataset_relative, ensure_layout, existing_file,
    existing_render, patch_rows, print_summary, reconcile_metadata, resolve_dataset_path, sync_due, tqdm,
)

DEFAULT_SURFEL_ADAPTOR = Path(__file__).resolve().parent.parent / "surfel_adaptor"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--encoder-config", type=Path, required=True,
                        help="JSON config containing models.encoder.name and models.encoder.args.")
    parser.add_argument("--encoder-checkpoint", type=Path, required=True,
                        help="Checkpoint for the sparse latent encoder.")
    parser.add_argument("--surfel-adaptor-root", type=Path, default=Path(DEFAULT_SURFEL_ADAPTOR),
                        help="Project root used to import encoder modules.")
    parser.add_argument("--device", default="cuda", help="Torch device for encoder inference.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Patches per encoder forward pass; default 1 is the safest GPU setting.")
    return parser.parse_args()


def load_state_dict(path: Path, torch: Any) -> dict[str, Any]:
    # Keep this compatible with the PyTorch versions used by the original toolkit.
    loaded = torch.load(path, map_location="cpu")
    if isinstance(loaded, dict) and isinstance(loaded.get("state_dict"), dict):
        loaded = loaded["state_dict"]
    if not isinstance(loaded, dict):
        raise TypeError("Encoder checkpoint is not a state dict")
    if any(str(key).startswith("module.") for key in loaded):
        return {str(key).removeprefix("module."): value for key, value in loaded.items()}
    return loaded


def read_encoder_spec(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))["models"]["encoder"]
        if not isinstance(spec["name"], str) or not isinstance(spec["args"], dict):
            raise TypeError("encoder name or args has an invalid type")
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid encoder config: {path}") from exc
    return spec


def load_encoder(args: argparse.Namespace, encoder_spec: dict[str, Any]):
    if not args.encoder_config.is_file():
        raise FileNotFoundError(args.encoder_config)
    if not args.encoder_checkpoint.is_file():
        raise FileNotFoundError(args.encoder_checkpoint)
    package_root = args.surfel_adaptor_root.resolve()
    if not (package_root / "surfel_adaptor").is_dir():
        raise FileNotFoundError(f"surfel_adaptor package not found below {package_root}")
    sys.path.insert(0, str(package_root))
    import torch
    import surfel_adaptor.models as models
    import surfel_adaptor.modules.sparse as sparse

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    try:
        encoder = getattr(models, encoder_spec["name"])(**encoder_spec["args"])
    except (AttributeError, TypeError) as exc:
        raise ValueError(f"Invalid encoder config: {args.encoder_config}") from exc
    state_dict = load_state_dict(args.encoder_checkpoint, torch)
    missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[step5] checkpoint compatibility: missing={len(missing)} unexpected={len(unexpected)}")
    encoder = encoder.to(args.device).eval()
    return torch, sparse, encoder


def batches(items: list[Any], size: int):
    for index in range(0, len(items), size):
        yield items[index:index + size]


def read_full_feature(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as archive:
        if "patchtokens" not in archive or "indices" not in archive:
            raise KeyError("full feature must contain patchtokens and indices")
        tokens, indices = archive["patchtokens"], archive["indices"]
    if tokens.ndim != 2 or indices.ndim != 2 or indices.shape[1] != 3 or len(tokens) != len(indices):
        raise ValueError("invalid full-feature dimensions")
    return tokens.astype(np.float32, copy=False), indices.astype(np.int32, copy=False)


def encode_batch(torch: Any, sparse: Any, encoder: Any, device: str,
                 features: list[tuple[np.ndarray, np.ndarray]]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Encode several patches together and split their sparse outputs by batch ID."""
    if not features:
        return []
    tokens = np.concatenate([token for token, _ in features], axis=0)
    coords = np.concatenate([
        np.column_stack((np.full(len(indices), batch_id, dtype=np.int32), indices))
        for batch_id, (_tokens, indices) in enumerate(features)
    ], axis=0)
    input_sparse = sparse.SparseTensor(
        feats=torch.from_numpy(tokens).float(), coords=torch.from_numpy(coords).int(),
    ).to(device)
    with torch.inference_mode():
        latent = encoder(input_sparse, sample_posterior=False)
    latent_feats = latent.feats.detach().cpu().float().numpy().astype(np.float32, copy=False)
    latent_coords = latent.coords.detach().cpu().numpy()
    if latent_coords.ndim != 2 or latent_coords.shape[1] != 4 or len(latent_coords) != len(latent_feats):
        raise ValueError("encoder emitted invalid sparse latent coordinates")
    if not np.isfinite(latent_feats).all():
        raise ValueError("encoder emitted non-finite values")
    return [
        (latent_feats[latent_coords[:, 0] == batch_id],
         latent_coords[latent_coords[:, 0] == batch_id, 1:].astype(np.uint8, copy=False))
        for batch_id in range(len(features))
    ]


def write_latent(path: Path, feats: np.ndarray, coords: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.stem}.", suffix=".npz", dir=path.parent,
                                     delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, feats=feats, coords=coords)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_latent_features(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        if "feats" not in archive:
            raise KeyError("latent archive must contain feats")
        feats = archive["feats"]
    if feats.ndim != 2 or not len(feats) or not np.isfinite(feats).all():
        raise ValueError("invalid latent features")
    return feats.astype(np.float32, copy=False)


def latent_statistics(root: Path, rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return per-channel population statistics for all metadata-tracked latents."""
    feature_sum: np.ndarray | None = None
    feature_sq_sum: np.ndarray | None = None
    feature_count = patch_count = 0
    for patch_id, row in patch_rows(rows).items():
        if not existing_file(root, row, "latent_path"):
            continue
        feats = read_latent_features(resolve_dataset_path(root, row["latent_path"]))
        current_sum = feats.sum(axis=0, dtype=np.float64)
        current_sq_sum = np.square(feats, dtype=np.float64).sum(axis=0, dtype=np.float64)
        if feature_sum is None:
            feature_sum, feature_sq_sum = current_sum, current_sq_sum
        elif feature_sum.shape != current_sum.shape:
            raise ValueError(f"inconsistent latent channel count for {patch_id}")
        else:
            feature_sum += current_sum
            feature_sq_sum += current_sq_sum
        feature_count += len(feats)
        patch_count += 1
    if feature_sum is None or feature_sq_sum is None:
        raise ValueError("no valid latent archives")
    mean = feature_sum / feature_count
    std = np.sqrt(np.maximum(feature_sq_sum / feature_count - np.square(mean), 0.0))
    return mean, std, patch_count, feature_count


def write_abstract(path: Path, *, encoder_spec: dict[str, Any], checkpoint: Path,
                   mean: np.ndarray, std: np.ndarray, patch_count: int, feature_count: int) -> None:
    """Atomically persist the encoder provenance and aggregate latent statistics."""
    abstract = {
        "model_type": encoder_spec["name"],
        "weights_filename": checkpoint.name,
        "model_parameters": encoder_spec["args"],
        "latent_mean": mean.tolist(),
        "latent_std": std.tolist(),
        "latent_patch_count": patch_count,
        "latent_feature_count": feature_count,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.stem}.", suffix=".json", dir=path.parent,
                                     mode="w", encoding="utf-8", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(abstract, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def is_oom_error(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def main() -> None:
    args = parse_args()
    if args.workers != 1:
        raise SystemExit("Step5 owns one encoder instance; use --batch-size instead of --workers.")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    root = args.dataset_root
    ensure_layout(root)
    rows = reconcile_metadata(root)
    patches = patch_rows(rows)
    pending: list[tuple[str, dict[str, str], Path]] = []
    for patch_id, row in patches.items():
        target = root / "latents" / f"latent_{patch_id}.npz"
        ready = existing_file(root, row, "full_feat_path") and existing_render(root, row)
        if ready and (args.overwrite or not existing_file(root, row, "latent_path") or not target.is_file()):
            pending.append((patch_id, row, target))
    if args.dry_run:
        for patch_id, row, target in pending:
            print(f"Would encode {row['full_feat_path']} -> {target}")
        print_summary(root, "step5", processed=len(pending), completed=len(pending),
                      skipped=0 if args.overwrite else len(patches) - len(pending), failed=0)
        return
    skipped = 0 if args.overwrite else len(patches) - len(pending)
    try:
        encoder_spec = read_encoder_spec(args.encoder_config)
        if not args.encoder_checkpoint.is_file():
            raise FileNotFoundError(args.encoder_checkpoint)
    except Exception as exc:
        print(f"[step5] failed to read encoder files: {exc}")
        print_summary(root, "step5", processed=len(pending), completed=0, skipped=skipped,
                      failed=len(pending), failed_ids=[patch_id for patch_id, _row, _target in pending])
        return
    torch = sparse = encoder = None
    if pending:
        try:
            torch, sparse, encoder = load_encoder(args, encoder_spec)
        except Exception as exc:
            print(f"[step5] failed to load encoder: {exc}")
            print_summary(root, "step5", processed=len(pending), completed=0, skipped=skipped,
                          failed=len(pending), failed_ids=[patch_id for patch_id, _row, _target in pending])
            return
    completed, failed = 0, 0
    failed_ids = []
    with tqdm(total=len(pending), desc="Step5 latents", unit="patch") as progress:
        for group in batches(pending, args.batch_size):
            prepared = []
            for task in group:
                patch_id, row, _target = task
                try:
                    prepared.append((task, read_full_feature(resolve_dataset_path(root, row["full_feat_path"]))))
                except Exception as exc:
                    failed += 1
                    failed_ids.append(patch_id)
                    print(f"[step5] failed {patch_id}: {exc}")
                    progress.update(1)
            if not prepared:
                continue

            def run(items: list[tuple[tuple[str, dict[str, str], Path], tuple[np.ndarray, np.ndarray]]]) -> None:
                nonlocal completed, failed
                outputs = encode_batch(torch, sparse, encoder, args.device, [feature for _, feature in items])
                for ((patch_id, _row, target), _feature), (latent_feats, latent_coords) in zip(items, outputs):
                    try:
                        write_latent(target, latent_feats, latent_coords)
                        append_journal(root, "step5", {"kind": "patch", "key": patch_id, "updates": {
                            "latent_id": target.stem, "latent_path": dataset_relative(root, target),
                        }})
                        completed += 1
                        if sync_due(completed, args.sync_every):
                            reconcile_metadata(root)
                    except Exception as exc:
                        failed += 1
                        failed_ids.append(patch_id)
                        print(f"[step5] failed {patch_id}: {exc}")
                    progress.update(1)

            try:
                run(prepared)
            except Exception as exc:
                if len(prepared) == 1 or not is_oom_error(exc):
                    for (patch_id, _row, _target), _feature in prepared:
                        failed += 1
                        failed_ids.append(patch_id)
                        print(f"[step5] failed {patch_id}: {exc}")
                        progress.update(1)
                else:
                    torch.cuda.empty_cache()
                    for item in prepared:
                        try:
                            run([item])
                        except Exception as retry_exc:
                            failed += 1
                            failed_ids.append(item[0][0])
                            print(f"[step5] failed {item[0][0]}: {retry_exc}")
                            progress.update(1)
    final_rows = reconcile_metadata(root)
    eligible_count = sum(
        existing_file(root, row, "full_feat_path") and existing_render(root, row)
        for row in patch_rows(final_rows).values()
    )
    try:
        mean, std, latent_patch_count, latent_feature_count = latent_statistics(root, final_rows)
        if failed or latent_patch_count != eligible_count:
            print("[step5] abstract.json was not updated: latent generation is incomplete.")
        else:
            abstract_path = root / "latents" / "abstract.json"
            write_abstract(
                abstract_path, encoder_spec=encoder_spec, checkpoint=args.encoder_checkpoint,
                mean=mean, std=std, patch_count=latent_patch_count,
                feature_count=latent_feature_count,
            )
            print(f"[step5] latent mean={mean.tolist()}\n[step5] latent std={std.tolist()}")
            print(f"[step5] wrote {abstract_path}")
    except (OSError, KeyError, ValueError) as exc:
        print(f"[step5] abstract.json was not updated: {exc}")
    print_summary(root, "step5", processed=len(pending), completed=completed, skipped=skipped,
                  failed=failed, failed_ids=failed_ids)


if __name__ == "__main__":
    main()
