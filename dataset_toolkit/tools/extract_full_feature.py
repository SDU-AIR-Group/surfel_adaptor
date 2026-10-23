from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass
class SonataRuntime:
    torch: Any
    sonata: Any
    model: Any


def find_sonata_root(toolkit_root: Path, checkpoint: Path, sonata_root: Path | None) -> Path:
    candidates = [sonata_root, toolkit_root, toolkit_root.parent]
    if len(checkpoint.parents) > 3:
        candidates.append(checkpoint.parents[3])
    for root in candidates:
        if root is None:
            continue
        if (root / "extensions" / "sonata").is_dir():
            return root
        if root.name == "sonata" and root.parent.name == "extensions" and root.is_dir():
            return root.parent.parent
    raise FileNotFoundError("Cannot find extensions/sonata; supply --sonata-root.")


def load_sonata(*, toolkit_root: Path, checkpoint: Path, sonata_root: Path | None, model_name: str) -> SonataRuntime:
    if model_name != "sonata":
        raise ValueError(f"Unsupported full-feature model: {model_name}")
    root = find_sonata_root(toolkit_root, checkpoint, sonata_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import torch
    import extensions.sonata as sonata

    if not torch.cuda.is_available():
        raise RuntimeError("Sonata full-feature extraction requires CUDA")
    try:
        import flash_attn  # noqa: F401
        config = None
    except ImportError:
        config = {"enc_patch_size": [1024] * 5, "enable_flash": False}
    model = sonata.load(str(checkpoint), repo_id="facebook/sonata", custom_config=config).cuda().eval()
    return SonataRuntime(torch=torch, sonata=sonata, model=model)


def sample_surface(object_path: Path) -> np.ndarray:
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(object_path))
    if mesh.is_empty():
        raise ValueError(f"No mesh geometry in {object_path}")
    mesh.rotate(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float), center=(0, 0, 0))
    return np.asarray(mesh.sample_points_uniformly(number_of_points=900_000).points)


def prepare_patch(object_path: Path, runtime: SonataRuntime) -> tuple[np.ndarray, dict[str, Any]]:
    import open3d as o3d

    raw_points = sample_surface(object_path)
    if len(raw_points) == 0:
        raise ValueError(f"Patch contains no sampled terrain points: {object_path}")
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(raw_points)
    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=10))
    points = runtime.sonata.transform.default()({
        "coord": raw_points.copy(),
        "color": np.zeros((len(raw_points), 3)),
        "normal": np.asarray(cloud.normals),
    })
    return raw_points, points


def _unpool_features(pack: Any, torch: Any) -> Any:
    for _ in range(2):
        if "pooling_parent" not in pack:
            break
        parent = pack.pop("pooling_parent")
        inverse = pack.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, pack.feat[inverse]], dim=-1)
        pack = parent
    while "pooling_parent" in pack:
        parent = pack.pop("pooling_parent")
        inverse = pack.pop("pooling_inverse")
        parent.feat = pack.feat[inverse]
        pack = parent
    return pack.feat[pack.inverse]


def forward_batch(runtime: SonataRuntime, prepared: Iterable[tuple[np.ndarray, dict[str, Any]]]) -> list[tuple[np.ndarray, Any]]:
    prepared = list(prepared)
    point = runtime.sonata.data.collate_fn([data for _, data in prepared])
    with runtime.torch.inference_mode():
        for key, value in point.items():
            if isinstance(value, runtime.torch.Tensor):
                point[key] = value.cuda(non_blocking=True)
        features = _unpool_features(runtime.model(point), runtime.torch)
    counts = [len(raw_points) for raw_points, _ in prepared]
    if len(features) != sum(counts):
        raise RuntimeError("Sonata output does not align with collated patch point counts")
    return list(zip((raw for raw, _ in prepared), runtime.torch.split(features, counts)))


def feature_archive(raw_points: np.ndarray, features: Any, *, terrain_scale: float, voxel_size: int, torch: Any) -> dict[str, np.ndarray]:
    rank = min(128, int(features.shape[0]), int(features.shape[1]))
    if rank < 1:
        raise ValueError("Sonata returned an empty feature tensor")
    _, _, basis = torch.pca_lowrank(features, center=True, q=rank, niter=5)
    projected = (features @ basis).detach().cpu().numpy().astype(np.float32)
    half = terrain_scale / 2.0
    width = terrain_scale / voxel_size
    clipped = np.clip(raw_points, -half + 1e-6, half - 1e-6)
    point_indices = np.floor((clipped + half) / width).astype(np.int64)
    indices, inverse = np.unique(point_indices, axis=0, return_inverse=True)
    tokens = np.zeros((len(indices), projected.shape[1]), dtype=np.float32)
    for index in range(len(indices)):
        tokens[index] = projected[inverse == index].mean(axis=0)
    return {"indices": indices.astype(np.uint8), "patchtokens": tokens}


def write_archive(path: Path, archive: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.npz")
    try:
        np.savez_compressed(temporary, **archive)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def is_oom_error(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()
