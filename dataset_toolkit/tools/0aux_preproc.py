import argparse, sys, os, math, re, glob
from typing import *
import bpy
from mathutils import Vector, Matrix
import numpy as np
import glob
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blender_scripts.basic import *


def main(arg):
    os.makedirs(arg.output_folder, exist_ok=True)
    init_scene()

    # Process every GLB in the input tree.
    for root, dirs, files in os.walk(arg.input_folder):
        for file in files:
            if file.lower().endswith('.glb'):
                init_scene()
                
                # Load origin model
                input_file_path = os.path.join(root, file)
                load_object(input_file_path)
                
                # Keep a small, merged mesh for scanning.
                objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
                if objects == [] or len(objects) > 10:
                    continue
                for obj in objects:
                    obj.parent = None
                
                bpy.ops.object.select_all(action='DESELECT')
                for obj in objects:
                    obj.select_set(True)

                bpy.context.view_layer.objects.active = objects[0]
                bpy.ops.object.join()
                target_obj = bpy.context.selected_objects[0]
                orig_init([target_obj])
                b_box = object_bbox(target_obj)
                # print(f'terrain_box: {terrain_box[1] - terrain_box[0]}')
                normalize_scale = 1.0 / max(b_box[1][0] - b_box[0][0], b_box[1][1] - b_box[0][1])
                print(f"normalize_scale: {(normalize_scale,)* 3}")
                scale_objects([target_obj], (normalize_scale,)* 3)
                orig_init([target_obj])
                set_transform_object(target_obj, np.identity(4))

                # Reset material for scan
                # if hasattr(target_obj.data, "materials"): continue
                if len(target_obj.material_slots)==0:
                    m=bpy.data.materials.new(target_obj.name+"_Mat"); target_obj.data.materials.append(m)
                for i,slot in enumerate(target_obj.material_slots):
                    if slot.material is None:
                        slot.material=bpy.data.materials.new(f"{target_obj.name}_Mat_{i:02d}")
                    mat=slot.material
                    if mat is None: continue
                    mat.use_nodes=True
                    nt=mat.node_tree; nodes=nt.nodes; links=nt.links
                    out=None
                    for n in nodes:
                        if n.type=='OUTPUT_MATERIAL' and getattr(n,'is_active_output',True): out=n; break
                    if out is None:
                        out=nodes.new('ShaderNodeOutputMaterial'); out.is_active_output=True; out.location=(300,0)
                    surf=out.inputs.get('Surface')
                    if surf is None: continue
                    principled=None
                    if surf.is_linked:
                        fn=surf.links[0].from_node
                        if fn.type=='BSDF_PRINCIPLED': principled=fn
                        else:
                            for l in list(surf.links): nt.links.remove(l)
                    if principled is None:
                        for n in nodes:
                            if n.type=='BSDF_PRINCIPLED': principled=n; break
                    if principled is None:
                        principled=nodes.new('ShaderNodeBsdfPrincipled'); principled.location=(0,0)
                    principled.inputs['Base Color'].default_value=(1,1,1,1)
                    principled.inputs['Roughness'].default_value=0.5
                    principled.inputs['Metallic'].default_value=0.0
                    if not surf.is_linked or surf.links[0].from_node!=principled:
                        links.new(principled.outputs['BSDF'], surf)

                if arg.save_mesh:
                    output_file_path = os.path.join(arg.output_folder, file)
                    save_object(output_file_path, target_obj)
                bpy.ops.object.select_all(action='SELECT')
                bpy.ops.object.delete()
                print(f"Processed {input_file_path} and saved to {output_file_path}")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
    parser.add_argument('--input_folder', type=str, help='Path to the 3D model file to be rendered.')
    parser.add_argument('--output_folder', type=str, default='/tmp', help='The path the output will be dumped to.')
    parser.add_argument('--save_mesh', action='store_true', help='Save the mesh as a .ply file.')
    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)
    main(args)
    
