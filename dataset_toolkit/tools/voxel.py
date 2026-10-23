import argparse
from pathlib import Path

import numpy as np


def write_ply(path: Path, vertices: np.ndarray) -> None:
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        np.savetxt(handle, vertices, fmt="%.8f %.8f %.8f")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory containing scans/scan.npy.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for voxels/voxel.ply.")
    parser.add_argument("--max_bound", type=float, default=25.0,
                        help="Half-width of the normalized voxel volume (metres).")
    parser.add_argument("--voxel_size", type=int, default=128, help="Voxel-grid resolution per axis.")
    args = parser.parse_args()
    if args.max_bound <= 0 or args.voxel_size < 1:
        raise SystemExit("--max_bound and --voxel_size must be positive")

    scan = np.load(args.input_dir / "scans" / "scan.npy")
    if scan.ndim != 2 or scan.shape[1] < 4:
        raise ValueError("scan.npy must have shape (N, >=4) with class ID in column 3")
    points = scan[scan[:, 3] == 0, :3]
    half = args.max_bound / 2.0
    if len(points) == 0:
        indices = np.empty((0, 3), dtype=np.int64)
    else:
        if np.any(points < -half) or np.any(points >= half):
            raise ValueError("terrain scan points are outside --max_bound")
        indices = np.floor((points + half) / (args.max_bound / args.voxel_size)).astype(np.int64)
        indices = np.unique(indices, axis=0)
    normalized = (indices.astype(np.float32) + 0.5) / args.voxel_size - 0.5

    output = args.output_dir / "voxels" / "voxel.ply"
    output.parent.mkdir(parents=True, exist_ok=True)
    write_ply(output, normalized)
    print(f"Saved {output}: {len(normalized)} voxels")


if __name__ == "__main__":
    main()
