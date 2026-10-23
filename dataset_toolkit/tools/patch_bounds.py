import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def dimensions(path: Path):
    clear_scene()
    bpy.ops.import_scene.gltf(filepath=str(path))
    points = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        points.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    if not points:
        return None
    mins = [min(point[i] for point in points) for i in range(3)]
    maxs = [max(point[i] for point in points) for i in range(3)]
    return [maxs[i] - mins[i] for i in range(3)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-dir", required=True, help="Directory containing patch_*.glb files.")
    parser.add_argument("--output-json", required=True, help="Path for the extracted bounds JSON file.")
    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)
    result = {path.name: dimensions(path) for path in sorted(Path(args.patch_dir).glob("patch_*.glb"))}
    Path(args.output_json).write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
