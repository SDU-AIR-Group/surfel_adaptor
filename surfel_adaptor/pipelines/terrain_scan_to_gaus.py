from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .. import samplers
from .base import Pipeline
from .. import models
from ..gaussian import Gaussian
from ..octree import DfsOctree as Octree
from ..modules import sparse as sp

try:
    import open3d as o3d
except ImportError:
    o3d = None


ScanInput = Union[str, os.PathLike[str], np.ndarray, torch.Tensor]


class TerrainScanToGausPipeline(Pipeline):
    """
    Inference pipeline for:
    scan points -> denser -> dense voxels -> denoiser -> latents -> decoder -> gaussians
    """

    def __init__(
        self,
        models: Optional[Dict[str, nn.Module]] = None,
        latent_sampler: Optional[samplers.Sampler] = None,
        latent_sampler_params: Optional[Dict[str, Any]] = None,
        slat_normalization: Optional[Dict[str, Sequence[float]]] = None,
        dense_threshold: float = 0.0,
        scan_preprocess: Optional[Dict[str, Any]] = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.latent_sampler = latent_sampler
        self.latent_sampler_params = latent_sampler_params or {}
        self.slat_normalization = slat_normalization
        self.dense_threshold = dense_threshold
        self.scan_preprocess = self._merge_scan_preprocess(scan_preprocess)

    @staticmethod
    def from_pretrained(path: str) -> "TerrainScanToGausPipeline":
        config_path = TerrainScanToGausPipeline._resolve_pipeline_config(path)
        root = config_path.parent
        with open(config_path, "r", encoding="utf-8") as f:
            args = json.load(f)["args"]

        loaded_models = {}
        for model_key, model_ref in args["models"].items():
            loaded_models[model_key] = TerrainScanToGausPipeline._load_model(root, model_ref)

        latent_sampler_cfg = args["latent_sampler"]
        latent_sampler = getattr(samplers, latent_sampler_cfg["name"])(**latent_sampler_cfg.get("args", {}))

        pipeline = TerrainScanToGausPipeline(
            models=loaded_models,
            latent_sampler=latent_sampler,
            latent_sampler_params=latent_sampler_cfg.get("params", {}),
            slat_normalization=args.get("slat_normalization"),
            dense_threshold=args.get("dense_threshold", 0.0),
            scan_preprocess=args.get("scan_preprocess"),
        )
        pipeline._pretrained_args = args
        return pipeline

    @staticmethod
    def _default_scan_preprocess() -> Dict[str, Any]:
        max_bound = 25.0
        half = max_bound / 2.0
        return {
            "max_bound": max_bound,
            "axis_ranges": {
                "x": [-half, half],
                "y": [-half, half],
                "z": [-half, half],
            },
        }

    @classmethod
    def _merge_scan_preprocess(cls, overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        config = cls._default_scan_preprocess()
        if overrides is None:
            return config

        if "max_bound" in overrides and "axis_ranges" not in overrides:
            half = float(overrides["max_bound"]) / 2.0
            config["axis_ranges"] = {
                "x": [-half, half],
                "y": [-half, half],
                "z": [-half, half],
            }

        merged = {
            **config,
            **{k: v for k, v in overrides.items() if k != "axis_ranges"},
        }
        axis_ranges = {
            axis: list(bounds)
            for axis, bounds in config["axis_ranges"].items()
        }
        for axis, bounds in overrides.get("axis_ranges", {}).items():
            axis_ranges[axis] = list(bounds)
        merged["axis_ranges"] = axis_ranges
        return merged

    @staticmethod
    def _resolve_pipeline_config(path: str) -> Path:
        local_config = Path(path) / "pipeline.json"
        if local_config.exists():
            return local_config

        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(path, "pipeline.json"))

    @staticmethod
    def _resolve_model_files(root: Path, ref: str) -> tuple[Path, Path]:
        stem = root / ref
        config_path = Path(f"{stem}.json")
        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found: {config_path}")

        for suffix in (".pt", ".pth", ".safetensors"):
            weight_path = Path(f"{stem}{suffix}")
            if weight_path.exists():
                return config_path, weight_path

        raise FileNotFoundError(f"No checkpoint found for {stem}")

    @staticmethod
    def _read_model_config(config_path: Path) -> Dict[str, Any]:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        if "name" in config and "args" in config:
            return config

        if "models" in config and len(config["models"]) == 1:
            return next(iter(config["models"].values()))

        raise ValueError(f"Unsupported model config format: {config_path}")

    @staticmethod
    def _extract_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
        if isinstance(payload, dict):
            if payload and all(torch.is_tensor(v) for v in payload.values()):
                return payload
            for key in (
                "state_dict",
                "model_state_dict",
                "model",
                "weights",
                "params",
                "module",
                "ema_state_dict",
            ):
                nested = payload.get(key)
                if isinstance(nested, dict) and nested and all(torch.is_tensor(v) for v in nested.values()):
                    return nested
        raise ValueError("Unsupported checkpoint payload format")

    @staticmethod
    def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        variants = [state_dict]
        prefixes = ("module.", "model.", "_orig_mod.", "denser.", "denoiser.", "decoder.")
        for prefix in prefixes:
            if state_dict and all(key.startswith(prefix) for key in state_dict):
                variants.append({key[len(prefix):]: value for key, value in state_dict.items()})
        return variants

    @staticmethod
    def _load_model(root: Path, ref: str) -> nn.Module:
        config_path, weight_path = TerrainScanToGausPipeline._resolve_model_files(root, ref)
        model_cfg = TerrainScanToGausPipeline._read_model_config(config_path)
        model = getattr(models, model_cfg["name"])(**model_cfg["args"])

        if weight_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state_dict = load_file(str(weight_path))
        else:
            try:
                payload = torch.load(weight_path, map_location="cpu", weights_only=True)
            except TypeError:
                payload = torch.load(weight_path, map_location="cpu")
            state_dict = TerrainScanToGausPipeline._extract_state_dict(payload)

        last_error: Optional[RuntimeError] = None
        for candidate in TerrainScanToGausPipeline._normalize_state_dict_keys(state_dict):
            try:
                model.load_state_dict(candidate)
                return model
            except RuntimeError as exc:
                last_error = exc

        raise RuntimeError(f"Failed to load checkpoint {weight_path}: {last_error}")

    @property
    def resolution(self) -> int:
        return int(self.models["latent_flow_model"].resolution)

    def _prepare_dense_volume(self, volume: torch.Tensor) -> torch.Tensor:
        if volume.ndim == 3:
            volume = volume.unsqueeze(0).unsqueeze(0)
        elif volume.ndim == 4:
            volume = volume.unsqueeze(1)
        elif volume.ndim != 5:
            raise ValueError(f"Unsupported dense scan tensor shape: {tuple(volume.shape)}")

        if volume.shape[1] != 1:
            raise ValueError("Dense scan tensor must have a single channel")

        if volume.shape[-1] != self.resolution or volume.shape[-2] != self.resolution or volume.shape[-3] != self.resolution:
            raise ValueError(
                f"Dense scan tensor resolution mismatch: expected {self.resolution}, got {tuple(volume.shape[-3:])}"
            )

        return (volume > 0).float()

    def _auto_scale_coords(
        self,
        coords: torch.Tensor,
        coord_mode: Literal["auto", "index", "zero_one", "centered"] = "auto",
    ) -> torch.Tensor:
        coords = coords.float()
        if coord_mode == "index":
            scaled = coords
        elif coord_mode == "zero_one":
            scaled = coords * self.resolution
        elif coord_mode == "centered":
            scaled = (coords + 0.5) * self.resolution
        else:
            cmin = float(coords.min().item())
            cmax = float(coords.max().item())
            if cmin >= -1e-6 and cmax <= 1.0 + 1e-6:
                scaled = coords * self.resolution
            elif cmin >= -0.5 - 1e-6 and cmax <= 0.5 + 1e-6:
                scaled = (coords + 0.5) * self.resolution
            else:
                scaled = coords
        return torch.floor(scaled).long()

    def _coords_to_dense(
        self,
        coords: torch.Tensor,
        coord_mode: Literal["auto", "index", "zero_one", "centered"] = "auto",
    ) -> torch.Tensor:
        if coords.ndim != 2 or coords.shape[1] not in (3, 4):
            raise ValueError(f"Point coordinates must have shape [N, 3] or [N, 4], got {tuple(coords.shape)}")

        if coords.shape[1] == 4:
            coords = coords[:, 1:]

        coords = self._auto_scale_coords(coords, coord_mode=coord_mode)
        valid_mask = ((coords >= 0) & (coords < self.resolution)).all(dim=1)
        coords = coords[valid_mask]
        if coords.numel() == 0:
            raise ValueError("No valid scan points remain after voxelization")

        coords = torch.unique(coords, dim=0)
        dense = torch.zeros(1, 1, self.resolution, self.resolution, self.resolution, dtype=torch.float32)
        dense[0, 0, coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0
        return dense

    def _infer_scan_mode(
        self,
        scan: torch.Tensor,
        scan_mode: Literal["auto", "pointcloud", "coords", "dense"],
    ) -> Literal["pointcloud", "coords", "dense"]:
        if scan_mode != "auto":
            return scan_mode

        if scan.ndim in (3, 4, 5):
            return "dense"

        if scan.ndim == 2 and scan.shape[1] in (3, 4):
            xyz = scan[:, :3].float()
            cmin = float(xyz.min().item())
            cmax = float(xyz.max().item())
            if cmin >= -0.5 - 1e-6 and cmax <= 1.0 + 1e-6:
                return "coords"
            if cmin >= 0.0 and cmax < float(self.resolution) + 1e-6:
                return "coords"
            return "pointcloud"

        raise ValueError(f"Unable to infer scan mode for tensor shape {tuple(scan.shape)}")

    def _preprocess_point_cloud(
        self,
        scan: torch.Tensor,
        preprocess_overrides: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if o3d is None:
            raise ImportError("open3d is required for point cloud preprocessing")

        preprocess_cfg = self._merge_scan_preprocess(preprocess_overrides)
        xyz = scan[:, :3].float().cpu().numpy()
        if xyz.size == 0:
            raise ValueError("Empty point cloud")

        center = xyz.mean(axis=0, keepdims=True)
        centered_xyz = xyz - center

        axis_ranges = preprocess_cfg["axis_ranges"]
        mask = np.ones(centered_xyz.shape[0], dtype=bool)
        for dim, axis in enumerate(("x", "y", "z")):
            lower, upper = axis_ranges[axis]
            mask &= centered_xyz[:, dim] >= lower
            mask &= centered_xyz[:, dim] <= upper
        filtered_xyz = centered_xyz[mask]
        if filtered_xyz.size == 0:
            raise ValueError("No points remain after XYZ pass-through filtering")

        max_bound = float(preprocess_cfg["max_bound"])
        voxel_res = max_bound / float(self.resolution)
        min_bound = np.array([axis_ranges["x"][0], axis_ranges["y"][0], axis_ranges["z"][0]], dtype=np.float64)
        max_bound_xyz = np.array([axis_ranges["x"][1], axis_ranges["y"][1], axis_ranges["z"][1]], dtype=np.float64)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filtered_xyz.astype(np.float64, copy=False))
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
            pcd,
            voxel_size=voxel_res,
            min_bound=min_bound,
            max_bound=max_bound_xyz,
        )
        voxel_indices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()], dtype=np.int32)
        if voxel_indices.size == 0:
            raise ValueError("No occupied voxels after voxelization")

        normalized_coords = (voxel_indices.astype(np.float32) + 0.5) / float(self.resolution) - 0.5
        metadata = {
            "scan_center": torch.from_numpy(center.squeeze(0)).float(),
            "scan_points_centered": torch.from_numpy(filtered_xyz.astype(np.float32)),
            "scan_voxel_coords_normalized": torch.from_numpy(normalized_coords),
            "scan_points_num_before_filter": int(xyz.shape[0]),
            "scan_points_num_after_filter": int(filtered_xyz.shape[0]),
            "scan_voxels_num": int(voxel_indices.shape[0]),
        }
        dense = self._coords_to_dense(torch.from_numpy(normalized_coords), coord_mode="centered")
        return dense, metadata

    def _load_scan_file(self, path: Union[str, os.PathLike[str]]) -> torch.Tensor:
        scan_path = Path(path)
        suffix = scan_path.suffix.lower()

        if suffix == ".npz":
            with np.load(scan_path) as data:
                for key in ("indices", "coords", "points", "xyz", "position"):
                    if key in data:
                        return torch.from_numpy(data[key])
            raise KeyError(f"Unsupported npz keys in {scan_path}")

        if suffix == ".npy":
            return torch.from_numpy(np.load(scan_path))

        if suffix in (".txt", ".xyz"):
            return torch.from_numpy(np.loadtxt(scan_path))

        if suffix in (".ply", ".pcd"):
            if o3d is None:
                raise ImportError("open3d is required to load .ply/.pcd scan files")
            point_cloud = o3d.io.read_point_cloud(str(scan_path))
            return torch.from_numpy(np.asarray(point_cloud.points))

        raise ValueError(f"Unsupported scan file format: {scan_path}")

    def preprocess_scan(
        self,
        scan: ScanInput,
        scan_mode: Literal["auto", "pointcloud", "coords", "dense"] = "auto",
        coord_mode: Literal["auto", "index", "zero_one", "centered"] = "auto",
        preprocess_overrides: Optional[Dict[str, Any]] = None,
        return_metadata: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        if isinstance(scan, (str, os.PathLike)):
            scan = self._load_scan_file(scan)
        elif isinstance(scan, np.ndarray):
            scan = torch.from_numpy(scan)
        elif not isinstance(scan, torch.Tensor):
            raise TypeError(f"Unsupported scan input type: {type(scan)}")

        metadata: Dict[str, Any] = {}
        inferred_mode = self._infer_scan_mode(scan, scan_mode)
        if inferred_mode == "dense":
            dense = self._prepare_dense_volume(scan.float())
        elif inferred_mode == "coords":
            dense = self._coords_to_dense(scan, coord_mode=coord_mode)
        else:
            dense, metadata = self._preprocess_point_cloud(scan, preprocess_overrides=preprocess_overrides)

        dense = dense.to(self.device)
        if return_metadata:
            return dense, metadata
        return dense

    @torch.no_grad()
    def run_denser(
        self,
        scan_voxels: torch.Tensor,
        num_samples: int = 1,
        sample_posterior: bool = False,
    ) -> torch.Tensor:
        if scan_voxels.shape[0] == 1 and num_samples > 1:
            scan_voxels = scan_voxels.repeat(num_samples, 1, 1, 1, 1)
        dense_logits = self.models["denser"](scan_voxels.float(), sample_posterior=sample_posterior, return_raw=False)
        return dense_logits

    @torch.no_grad()
    def dense_to_coords(self, dense_logits: torch.Tensor) -> torch.Tensor:
        dense_mask = dense_logits > self.dense_threshold
        coords = torch.argwhere(dense_mask)[:, [0, 2, 3, 4]].int()
        if coords.numel() == 0:
            raise ValueError("Denser produced no occupied voxels")
        return coords
    
    @torch.no_grad()
    def voxel_to_octree(self, voxels: torch.Tensor) -> Octree:
        representation = Octree(
            depth=20,
            aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
            device='cuda',
            primitive='voxel',
            sh_degree=0,
            primitive_config={'solid': True},
        )
        coords = torch.nonzero(voxels[0, 0], as_tuple=False)
        representation.position = coords.float() / self.resolution
        representation.depth = torch.full((representation.position.shape[0], 1), int(np.log2(self.resolution)), dtype=torch.uint8, device='cuda')
        return representation

    @torch.no_grad()
    def sample_latents(
        self,
        coords: torch.Tensor,
        sampler_params: Optional[Dict[str, Any]] = None,
    ) -> sp.SparseTensor:
        flow_model = self.models["latent_flow_model"]
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels, device=self.device),
            coords=coords.to(self.device),
        )
        params = {**self.latent_sampler_params, **(sampler_params or {})}
        latents = self.latent_sampler.sample(flow_model, noise, verbose=True, **params).samples

        if self.slat_normalization is not None:
            std = torch.tensor(self.slat_normalization["std"], device=latents.device, dtype=latents.feats.dtype).unsqueeze(0)
            mean = torch.tensor(self.slat_normalization["mean"], device=latents.device, dtype=latents.feats.dtype).unsqueeze(0)
            latents = latents * std + mean

        return latents

    @torch.no_grad()
    def decode_gaussians(self, latents: sp.SparseTensor) -> List[Gaussian]:
        if self.device.type != "cuda":
            raise RuntimeError("TerrainScanToGausPipeline currently requires CUDA for gaussian decoding")
        return self.models["gaussian_decoder"](latents)

    @torch.no_grad()
    def run(
        self,
        scan: ScanInput,
        num_samples: int = 1,
        seed: int = 42,
        scan_mode: Literal["auto", "pointcloud", "coords", "dense"] = "auto",
        coord_mode: Literal["auto", "index", "zero_one", "centered"] = "auto",
        scan_preprocess_overrides: Optional[Dict[str, Any]] = None,
        sample_denser_posterior: bool = False,
        latent_sampler_params: Optional[Dict[str, Any]] = None,
        return_intermediates: bool = True,
    ) -> Dict[str, Any]:
        scan_voxels, scan_metadata = self.preprocess_scan(
            scan,
            scan_mode=scan_mode,
            coord_mode=coord_mode,
            preprocess_overrides=scan_preprocess_overrides,
            return_metadata=True,
        )
        torch.manual_seed(seed)

        dense_logits = self.run_denser(
            scan_voxels,
            num_samples=num_samples,
            sample_posterior=sample_denser_posterior,
        )
        dense_coords = self.dense_to_coords(dense_logits)
        latents = self.sample_latents(dense_coords, sampler_params=latent_sampler_params)
        gaussians = self.decode_gaussians(latents)

        outputs: Dict[str, Any] = {"gaussian": gaussians}
        if return_intermediates:
            outputs.update(
                {
                    "scan_voxels": scan_voxels,
                    "dense_logits": dense_logits,
                    "dense_voxels": (dense_logits > self.dense_threshold).to(dense_logits.dtype),
                    "dense_coords": dense_coords,
                    "latents": latents,
                }
            )
            outputs.update(scan_metadata)
        return outputs
