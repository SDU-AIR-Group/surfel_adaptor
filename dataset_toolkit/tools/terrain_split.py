import argparse
import random
import sys
import time
from pathlib import Path


TOOLKIT_ROOT = Path(__file__).resolve().parent.parent
if str(TOOLKIT_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLKIT_ROOT))


def timestamp_seed() -> int:
    """Return the last five digits of the current Unix timestamp."""
    return int(str(int(time.time()))[-5:])


def scale_factors(enabled: bool, xy_range: tuple[float, float],
                  z_range: tuple[float, float]) -> tuple[float, float, float]:
    """Draw independent x/y/z scale factors after terrain normalization."""
    if not enabled:
        return 1.0, 1.0, 1.0
    rng = random.Random(timestamp_seed())
    return rng.uniform(*xy_range), rng.uniform(*xy_range), rng.uniform(*z_range)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", required=True, help="Source terrain GLB file.")
    parser.add_argument("--output_folder", required=True, help="Directory for generated patch GLB files.")
    parser.add_argument("--lidar_height", type=float, default=0.83,
                        help="LiDAR height used to align patch origins (metres).")
    parser.add_argument("--scale_augment", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply independent x/y/z scale augmentation.")
    parser.add_argument("--patch_size_meters", type=float, default=25.0,
                        help="Grid-cell width used to split the terrain (metres).")
    parser.add_argument("--xy_scale_range", nargs=2, type=float, default=(0.5, 1.5),
                        help="Uniform scale range for x and y.")
    parser.add_argument("--z_scale_range", nargs=2, type=float, default=(0.8, 1.2),
                        help="Uniform scale range for z.")
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else None
    return parser.parse_args(arguments)


def main(args: argparse.Namespace) -> None:
    import bpy
    import numpy as np
    from mathutils import Vector
    from blender_scripts.basic import (
        get_normal_at_point, init_render, init_scene, load_object, object_bbox, orig_init,
        save_object, scale_objects, set_transform_object, split_mesh_to_grid,
    )

    output = Path(args.output_folder)
    (output / "patches").mkdir(parents=True, exist_ok=True)
    init_render()
    init_scene()
    load_object(args.object)
    terrain = bpy.context.scene.objects.values()[0]
    objects = bpy.context.scene.objects.values()
    orig_init(objects)
    bounds = object_bbox(terrain)
    horizontal_size = max(bounds[1][0] - bounds[0][0], bounds[1][1] - bounds[0][1])
    if horizontal_size <= 0:
        raise RuntimeError("Terrain has no horizontal extent")
    scale_objects(objects, (100.0 / horizontal_size,) * 3)
    factors = scale_factors(args.scale_augment, tuple(args.xy_scale_range), tuple(args.z_scale_range))
    scale_objects([terrain], factors)
    orig_init([terrain])
    set_transform_object(terrain, np.identity(4).tolist())
    print(f"Scale augmentation: enabled={args.scale_augment}, factors={factors}")

    for index, patch in enumerate(split_mesh_to_grid(terrain, 4, max_size=args.patch_size_meters)):
        patch.select_set(True)
        patch.name = "terrain"
        orig_init([patch])
        set_transform_object(patch, np.identity(4).tolist())
        _, intersection = get_normal_at_point(patch, Vector((0.0, 0.0, 100.0)))
        if intersection is not None:
            transform = np.identity(4)
            transform[:3, 3] = [0.0, 0.0, -intersection[2] - args.lidar_height]
            set_transform_object(patch, transform.tolist())
            save_object(str(output / "patches" / f"patch_{index}.glb"), patch)
        bpy.ops.object.delete()
        bpy.ops.object.select_all(action="DESELECT")




if __name__ == "__main__":
    main(parse_args())
