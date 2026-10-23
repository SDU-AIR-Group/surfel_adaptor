import argparse, sys, os, math, re, glob
from typing import *
import bpy
from mathutils import Vector, Matrix
import numpy as np
import json
import glob
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blender_scripts.basic import *


def main(arg):
    os.makedirs(arg.output_folder, exist_ok=True)
    
    # Initialize context
    init_render()

    if arg.object.endswith(".blend"):
        delete_invisible_objects()
    else:
        init_scene()
        # Load origin model
        load_object(arg.object)
        if arg.split_normal:
            split_mesh_normal()
        # delete_custom_normals()
    print('[INFO] Scene initialized.')

    # Scaling terrain object to (100, 100)
    print(f"Imported objects: {bpy.context.scene.objects.values()}")
    terrain_obj = bpy.context.scene.objects.values()[0]
    orig_init(bpy.context.scene.objects.values())
    print('[INFO] origin is initialized.')
    terrain_box = object_bbox(terrain_obj)
    print(f'terrain_box: {terrain_box}')
    normalize_scale = 100.0 / max(terrain_box[1][0] - terrain_box[0][0], terrain_box[1][1] - terrain_box[0][1])
    print(f"normalize_scale: {(normalize_scale,)* 3}")
    scale_objects(bpy.context.scene.objects.values(), (normalize_scale,)* 3)
    orig_init(bpy.context.scene.objects.values())
    set_transform_object(terrain_obj, np.identity(4).tolist())
    
    # Override material
    if arg.geo_mode:
        override_material()
        
    if arg.save_mesh:
        save_object(os.path.join(arg.output_folder, 'model.glb'))

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
    parser.add_argument('--object', type=str, help='Path to the 3D model file to be rendered.')
    parser.add_argument('--output_folder', type=str, default='/tmp', help='The path the output will be dumped to.')
    parser.add_argument('--geo_mode', action='store_true', help='Geometry mode for rendering.')
    parser.add_argument('--split_normal', action='store_true', help='Split the normals of the mesh.')
    parser.add_argument('--save_mesh', action='store_true', help='Save the mesh as a .ply file.')
    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)
    main(args)
    