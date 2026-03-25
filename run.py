import argparse
import os
import sys
from pathlib import Path
from typing import *
import imageio.v2 as imageio
import torch

os.environ.setdefault("SPCONV_ALGO", "native")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from surfel_adaptor.pipelines import TerrainScanToGausPipeline
from surfel_adaptor.utils import render_utils


DEFAULT_PIPELINE_ROOT = Path(__file__).resolve().parent / "surfel_adaptor" / "configs" / "eval" / "terrain_scan_to_gaus"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run terrain scan to gaussian inference.")
    parser.add_argument("--input", type=str, required=True, help="Path to scan file (.npz/.npy/.txt/.ply/.pcd).")
    parser.add_argument("--output", type=str, default="terrain_gaussian.ply", help="Output gaussian ply path.")
    parser.add_argument("--pretrained", type=str, default=str(DEFAULT_PIPELINE_ROOT), help="Pipeline config directory.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Inference device.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of gaussian samples to generate.")
    parser.add_argument("--sample_index", type=int, default=0, help="Which sample to export when num_samples > 1.")
    parser.add_argument("--scan_mode", type=str, default="auto", choices=["auto", "pointcloud", "coords", "dense"], help="Interpret input scan as real-scale point cloud, voxel coords, or dense occupancy.")
    parser.add_argument("--coord_mode", type=str, default="auto", choices=["auto", "index", "zero_one", "centered"], help="How to interpret input point coordinates.")
    parser.add_argument("--max_bound", type=float, default=None, help="Bounding box size used for point cloud pass-through and voxelization.")
    parser.add_argument("--x_min", type=float, default=None, help="Optional X lower bound after centering.")
    parser.add_argument("--x_max", type=float, default=None, help="Optional X upper bound after centering.")
    parser.add_argument("--y_min", type=float, default=None, help="Optional Y lower bound after centering.")
    parser.add_argument("--y_max", type=float, default=None, help="Optional Y upper bound after centering.")
    parser.add_argument("--z_min", type=float, default=None, help="Optional Z lower bound after centering.")
    parser.add_argument("--z_max", type=float, default=None, help="Optional Z upper bound after centering.")
    parser.add_argument("--steps", type=int, default=None, help="Override latent sampler steps.")
    parser.add_argument("--rescale_t", type=float, default=None, help="Override latent sampler rescale_t.")
    parser.add_argument("--sample_denser_posterior", action="store_true", help="Sample the denser posterior instead of using the mean.")
    parser.add_argument("--video", type=str, default=None, help="Output gaussian video path. Defaults to <output stem>.mp4.")
    return parser


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        raise RuntimeError("CPU inference is not supported by the current gaussian decoder implementation")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for terrain scan to gaussian inference")
    return torch.device("cuda")


def build_scan_preprocess_overrides(args: argparse.Namespace) -> Optional[Dict[str, object]]:
    overrides: Dict[str, object] = {}
    if args.max_bound is not None:
        overrides["max_bound"] = args.max_bound

    axis_ranges = {}
    for axis in ("x", "y", "z"):
        lower = getattr(args, f"{axis}_min")
        upper = getattr(args, f"{axis}_max")
        if lower is not None or upper is not None:
            if lower is None or upper is None:
                raise ValueError(f"Both --{axis}_min and --{axis}_max must be provided together")
            axis_ranges[axis] = [lower, upper]
    if axis_ranges:
        overrides["axis_ranges"] = axis_ranges

    return overrides or None


def main() -> None:
    args = build_parser().parse_args()

    pipeline = TerrainScanToGausPipeline.from_pretrained(args.pretrained)
    device = resolve_device(args.device)
    pipeline.to(device)

    sampler_overrides = {}
    if args.steps is not None:
        sampler_overrides["steps"] = args.steps
    if args.rescale_t is not None:
        sampler_overrides["rescale_t"] = args.rescale_t
    scan_preprocess_overrides = build_scan_preprocess_overrides(args)

    outputs = pipeline.run(
        args.input,
        num_samples=args.num_samples,
        seed=args.seed,
        scan_mode=args.scan_mode,
        coord_mode=args.coord_mode,
        scan_preprocess_overrides=scan_preprocess_overrides,
        sample_denser_posterior=args.sample_denser_posterior,
        latent_sampler_params=sampler_overrides or None,
    )

    gaussians = outputs["gaussian"]
    ss_voxel = outputs["scan_voxels"]
    ds_voxel = outputs["dense_voxels"]
    ss_octree = pipeline.voxel_to_octree(ss_voxel)
    ds_octree = pipeline.voxel_to_octree(ds_voxel)

    if args.sample_index < 0 or args.sample_index >= len(gaussians):
        raise IndexError(f"sample_index {args.sample_index} out of range for {len(gaussians)} outputs")

    selected_gaussian = gaussians[args.sample_index]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # gaussain_video = render_utils.render_video(selected_gaussian, r=60)["color"]
    # imageio.mimsave("sample_gs.mp4", gaussain_video, fps=30)        
    # ss_voxel_video = render_utils.render_video(ss_octree, r=2, colors_overwrite=ss_octree.position)["color"]
    # imageio.mimsave("sample_ss.mp4", ss_voxel_video, fps=30)        
    # ds_voxel_video = render_utils.render_video(ds_octree, r=2, colors_overwrite=ds_octree.position)["color"]
    # imageio.mimsave("sample_ds.mp4", ds_voxel_video, fps=30)        
    multisample_video = render_utils.render_multisample_video([ss_octree, ds_octree, selected_gaussian],
                                                              r = [2, 2, 50])["color"]
    imageio.mimsave("sample.mp4", multisample_video, fps=30)        
    print(f"Saved video to sample.mp4")

    selected_gaussian.save_ply(str(output_path))
    print(f"Saved gaussian ply to {output_path}")


if __name__ == "__main__":
    main()
