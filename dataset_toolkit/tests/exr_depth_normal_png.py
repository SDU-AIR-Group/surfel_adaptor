import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory containing multilayer EXR frames.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _channel_names(header: dict) -> set[str]:
    return {str(name) for name in header["channels"].keys()}


def _read_layer(exr, header: dict, layer: str, components: tuple[str, ...], size: tuple[int, int], pixel_type):
    names = _channel_names(header)
    channels = []
    aliases = {"R": ("R", "X", "V", "Z"), "G": ("G", "Y"), "B": ("B", "Z")}
    for component in components:
        candidates = tuple(f"{layer}.{name}" for name in aliases.get(component, (component,))) + (component,)
        name = next((candidate for candidate in candidates if candidate in names), None)
        if name is None:
            raise ValueError(f"EXR is missing {layer!r} channel {component!r}; found {sorted(names)}")
        channels.append(np.frombuffer(exr.channel(name, pixel_type), dtype=np.float32).reshape(size))
    return np.stack(channels, axis=-1)


def read_depth_and_normal(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import Imath
        import OpenEXR
    except ImportError as exc:
        raise RuntimeError("OpenEXR and Imath are required; run this tool with the GDP environment.") from exc

    exr = OpenEXR.InputFile(str(path))
    header = exr.header()
    window = header["dataWindow"]
    size = (window.max.y - window.min.y + 1, window.max.x - window.min.x + 1)
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    try:
        depth = _read_layer(exr, header, "depth", ("R",), size, pixel_type)[..., 0]
        normal = _read_layer(exr, header, "normal", ("R", "G", "B"), size, pixel_type)
    finally:
        exr.close()
    return depth, normal


def make_preview(depth: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Return an RGB preview: grayscale depth on the left, world normal on the right."""
    if depth.ndim != 2 or normal.shape != (*depth.shape, 3):
        raise ValueError("depth must be HxW and normal must be HxWx3")
    valid = np.isfinite(depth) & (depth > 0.0) & np.isfinite(normal).all(axis=-1)
    depth_u8 = np.zeros(depth.shape, dtype=np.uint8)
    depth_u8[valid] = np.rint(np.clip(depth[valid], 0.0, 1.0) * 255.0).astype(np.uint8)
    depth_rgb = np.repeat(depth_u8[..., None], 3, axis=-1)

    normal_rgb = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if np.any(valid):
        vectors = normal[valid]
        lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
        normal_rgb[valid] = np.rint(
            np.clip((vectors / np.maximum(lengths, 1e-8) + 1.0) * 0.5, 0.0, 1.0) * 255.0
        ).astype(np.uint8)
    return np.concatenate((depth_rgb, normal_rgb), axis=1)


def main() -> None:
    args = parse_args()
    frames = sorted(args.input_dir.glob("*.exr"))
    if not frames:
        raise SystemExit(f"No EXR files found in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    for frame in frames:
        depth, normal = read_depth_and_normal(frame)
        Image.fromarray(make_preview(depth, normal), mode="RGB").save(args.output_dir / f"{frame.stem}.png")
    print(f"Converted {len(frames)} frames to {args.output_dir}")


if __name__ == "__main__":
    main()
