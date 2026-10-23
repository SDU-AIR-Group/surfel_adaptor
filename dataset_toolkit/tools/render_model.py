import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    from common import sphere_hammersley_sequence, sphere_segment_hammersley_sequence
except ImportError:  # Supports importing this module from the toolkit root.
    from tools.common import sphere_hammersley_sequence, sphere_segment_hammersley_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", required=True, help="Blender executable for headless rendering.")
    parser.add_argument("--toolkit-root", type=Path, required=True, help="Directory containing blender_scripts/render.py.")
    parser.add_argument("--object-path", type=Path, required=True, help="Terrain patch GLB file to render.")
    parser.add_argument("--output-folder", type=Path, required=True, help="Directory for EXR frames and transforms.json.")
    parser.add_argument("--terrain-scale", type=float, default=25.0,
                        help="Patch half-width used to place cameras (metres).")
    parser.add_argument("--num-views", type=int, default=100, help="Total camera views to render.")
    parser.add_argument(
        "--local-view-num", type=int, default=None,
        help="Number of local segment views; defaults to half of the complete view sequence.",
    )
    parser.add_argument("--view-offset", type=int, default=0,
                        help="First index within the deterministic view sequence.")
    parser.add_argument("--view-total", type=int, default=None,
                        help="Total sequence length; defaults to --num-views.")
    parser.add_argument("--resolution", type=int, default=512, help="Square image resolution in pixels.")
    parser.add_argument("--yaw-min", type=float, default=0.0, help="Minimum camera yaw in degrees.")
    parser.add_argument("--yaw-max", type=float, default=360.0, help="Maximum camera yaw in degrees.")
    parser.add_argument("--pitch-min", type=float, default=-30.0, help="Minimum camera pitch in degrees.")
    parser.add_argument("--pitch-max", type=float, default=30.0, help="Maximum camera pitch in degrees.")
    parser.add_argument("--radius-min", type=float, default=None, help="Optional minimum camera radius in metres.")
    parser.add_argument("--radius-max", type=float, default=None, help="Optional maximum camera radius in metres.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for deterministic view sampling.")
    return parser.parse_args()


def build_views(args: argparse.Namespace, total_views: int) -> list[dict[str, float]]:
    """Build the complete deterministic local-plus-global camera sequence."""
    fov = math.radians(40.0)
    base_radius = (args.terrain_scale * math.sqrt(2) * 0.5) * 1.1 / math.tan(fov * 0.5)
    minimum = args.radius_min if args.radius_min is not None else base_radius * 0.2
    maximum = args.radius_max if args.radius_max is not None else base_radius * 0.9
    minimum, maximum = min(minimum, maximum), max(minimum, maximum)
    local_count = total_views // 2 if args.local_view_num is None else args.local_view_num
    if not 0 <= local_count < total_views:
        raise SystemExit("--local-view-num must satisfy 0 <= local-view-num < total views")

    rng = np.random.default_rng(args.seed)
    local_offset = rng.random(3)
    global_offset = rng.random(2)
    yaw_range = (math.radians(args.yaw_min), math.radians(args.yaw_max))
    pitch_range = (math.radians(args.pitch_min), math.radians(args.pitch_max))
    views: list[dict[str, float]] = []
    for index in range(local_count):
        yaw, pitch, radius = sphere_segment_hammersley_sequence(
            index, local_count, yaw_range=yaw_range, pitch_range=pitch_range,
            radius_range=(minimum, maximum), offset=local_offset,
        )
        views.append({"yaw": yaw, "pitch": pitch, "radius": radius, "fov": fov})

    global_count = total_views - local_count
    for index in range(global_count):
        yaw, pitch = sphere_hammersley_sequence(index, global_count, offset=global_offset)
        views.append({"yaw": yaw, "pitch": pitch, "radius": base_radius, "fov": fov})
    return views


def main() -> None:
    args = parse_args()
    if args.num_views < 1 or args.view_offset < 0:
        raise SystemExit("--num-views must be positive")
    total_views = args.view_total if args.view_total is not None else args.num_views
    if total_views < args.num_views + args.view_offset:
        raise SystemExit("--view-total must cover --view-offset + --num-views")
    renderer = args.toolkit_root / "blender_scripts" / "render.py"
    if not renderer.is_file():
        raise SystemExit(f"Renderer not found: {renderer}")
    views = build_views(args, total_views)[args.view_offset:args.view_offset + args.num_views]
    args.output_folder.mkdir(parents=True, exist_ok=True)
    command = [args.blender, "-b", "-P", str(renderer), "--", "--views", json.dumps(views),
               "--object", str(args.object_path), "--resolution", str(args.resolution),
               "--output_folder", str(args.output_folder), "--engine", "CYCLES", "--save_mesh",
               "--frame-offset", str(args.view_offset)]
    print("$", " ".join(command))
    result = subprocess.run(command, check=False)
    if result.returncode:
        raise SystemExit(result.returncode)
    if not (args.output_folder / "transforms.json").is_file():
        raise RuntimeError("Renderer returned success without transforms.json")


if __name__ == "__main__":
    main()
