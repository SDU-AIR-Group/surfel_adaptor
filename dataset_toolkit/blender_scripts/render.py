import argparse, sys, os, math, re, glob
from typing import *
import bpy
from mathutils import Vector, Matrix
import numpy as np
import json
import glob
sys.path.append(os.path.dirname(__file__))
from basic import *

GEOMETRY_NORMAL_AOV = 'geometry_normal'


def config_geometry_normal():
    """Removing material normal maps."""
    view_layer = bpy.context.view_layer
    aov = next((item for item in view_layer.aovs if item.name == GEOMETRY_NORMAL_AOV), None)
    if aov is None:
        aov = view_layer.aovs.add()
        aov.name = GEOMETRY_NORMAL_AOV
    aov.type = 'COLOR'

    materials = set()
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue
        if not obj.material_slots:
            material = bpy.data.materials.new(f'{obj.name}_surface_normal')
            obj.data.materials.append(material)
        materials.update(slot.material for slot in obj.material_slots if slot.material is not None)

    for material in materials:
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        geometry = nodes.get('GEOMETRY_NORMAL_SOURCE')
        if geometry is None:
            geometry = nodes.new('ShaderNodeNewGeometry')
            geometry.name = 'GEOMETRY_NORMAL_SOURCE'
            geometry.label = 'Geometric surface normal'
        output = next((node for node in nodes
                       if node.bl_idname == 'ShaderNodeOutputAOV'
                       and node.aov_name == GEOMETRY_NORMAL_AOV), None)
        if output is None:
            output = nodes.new('ShaderNodeOutputAOV')
            output.name = 'GEOMETRY_NORMAL_AOV_OUTPUT'
        output.aov_name = GEOMETRY_NORMAL_AOV
        for link in tuple(output.inputs['Color'].links):
            links.remove(link)
        links.new(geometry.outputs['True Normal'], output.inputs['Color'])

def init_depth_node():
    """Create one multilayer EXR output containing depth and geometric normals."""
    outputs = {}
    spec_nodes = {}
    
    bpy.context.scene.use_nodes = True
    print(bpy.context.scene.view_layers.values())
    bpy.context.scene.view_layers['ViewLayer'].use_pass_z = True
    bpy.context.scene.view_layers['ViewLayer'].use_pass_normal = False
    
    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links
    for n in nodes:
        nodes.remove(n)
    
    render_layers = nodes.new('CompositorNodeRLayers')

    multilayer_output = nodes.new('CompositorNodeOutputFile')
    multilayer_output.base_path = ''
    multilayer_output.file_slots[0].use_node_format = True
    # Must be fixed before the first compositor evaluation; Blender caches it.
    multilayer_output.file_slots[0].path = 'frame_'
    multilayer_output.format.file_format = 'OPEN_EXR_MULTILAYER'
    multilayer_output.format.color_depth = '16'
    multilayer_output.format.color_mode = 'RGB'
    multilayer_output.layer_slots[0].name = 'depth'
    multilayer_output.file_slots.new('normal')
    # Remap to 0-1
    map = nodes.new(type="CompositorNodeMapRange")
    map.inputs[1].default_value = 0  # (min value you will be getting)
    map.inputs[2].default_value = 10 # (max value you will be getting)
    map.inputs[3].default_value = 0  # (min value you will map to)
    map.inputs[4].default_value = 1  # (max value you will map to)
    
    links.new(render_layers.outputs['Depth'], map.inputs[0])
    links.new(map.outputs[0], multilayer_output.inputs[0])
    normal = render_layers.outputs.get(GEOMETRY_NORMAL_AOV)
    if normal is None:
        raise RuntimeError('Geometric-normal AOV was not configured.')
    links.new(normal, multilayer_output.inputs['normal'])
    
    # Both names intentionally reference one File Output node: one frame is a
    # single multilayer EXR, whose layers are named ``depth`` and ``normal``.
    outputs['depth'] = multilayer_output
    outputs['normal'] = multilayer_output
    spec_nodes['depth_map'] = map
        
    return outputs, spec_nodes

def main(arg):
    os.makedirs(arg.output_folder, exist_ok=True)
    
    # Initialize context
    init_render(engine=arg.engine, resolution=arg.resolution, geo_mode=arg.geo_mode)
    if arg.object.endswith(".blend"):
        delete_invisible_objects()
    else:
        init_scene()
        load_object(arg.object)
        if arg.split_normal:
            split_mesh_normal()
        # delete_custom_normals()
    config_geometry_normal()
    outputs, spec_nodes = init_depth_node()
    print('[INFO] Scene initialized.')
    
    # normalize scene
    bbox_min, bbox_max = object_bbox(bpy.context.scene.objects.values()[0])
    scale = max(np.array(bbox_max) - np.array(bbox_min))
    # scale, offset = normalize_scene()
    
    # Initialize camera and lighting
    cam = init_camera()
    init_lighting()
    print('[INFO] Camera and lighting initialized.')

    # Override material
    # if arg.geo_mode:
    #     override_material()
    
    # Create a list of views
    to_export = {
        "aabb": [(np.array(bbox_min)).tolist(), (np.array(bbox_max)).tolist()],
        "scale": scale,
        # "offset": [offset.x, offset.y, offset.z],
        "frames": []
    }
    views = json.loads(arg.views)
    for i, view in enumerate(views):
        frame_index = arg.frame_offset + i
        cam.location = (
            view['radius'] * np.cos(view['yaw']) * np.cos(view['pitch']),
            view['radius'] * np.sin(view['yaw']) * np.cos(view['pitch']),
            view['radius'] * np.sin(view['pitch'])
        )
        cam.data.lens_unit = "FOV"
        cam.data.angle_x = view['fov']
        cam.data.angle_y = view['fov']
        cam.data.clip_start = 0.01
        cam.data.clip_end = 2.0 * scale * (math.sqrt(2) * 0.5) * 1.1 / np.sin(view['fov'] * 0.5)

        spec_nodes['depth_map'].inputs[1].default_value = 0.01
        spec_nodes['depth_map'].inputs[2].default_value = cam.data.clip_end
        
        # bpy.context.scene.render.filepath = os.path.join(arg.output_folder, f'{i:03d}_rgb.png')
        output = outputs['depth']
        output_prefix = os.path.join(arg.output_folder, f'{frame_index:03d}')
        # OPEN_EXR_MULTILAYER derives its file name from ``base_path`` and the
        # scene frame, rather than from individual file-slot paths.
        output.base_path = os.path.join(arg.output_folder, 'frame_')
        # The compositor caches a File Output slot path after its first render.
        # Advancing the scene frame gives each view a unique temporary filename.
        bpy.context.scene.frame_set(frame_index + 1)
        
        # Render the scene
        bpy.ops.render.render(write_still=True)
        bpy.context.view_layer.update()
        ext = EXT[output.format.file_format]
        path = glob.glob(os.path.join(arg.output_folder, f'frame_*.{ext}'))[0]
        os.rename(path, f'{output_prefix}.{ext}')
            
        # Save camera parameters
        metadata = {
            "file_path": f'{frame_index:03d}.{ext}',
            "camera_angle_x": view['fov'],
            "transform_matrix": get_transform_matrix(cam)
        }
        metadata['depth'] = {
            'min': 0.01,
            'max': cam.data.clip_end
        }
        metadata['normal'] = {
            'layer': 'normal',
            'space': 'world',
            'encoding': 'signed_unit_vector',
        }
        to_export["frames"].append(metadata)
    
    # Append batch renders by frame name; the normal Step4 path uses offset 0
    # and continues to write one complete transforms.json in a single run.
    transforms_path = os.path.join(arg.output_folder, 'transforms.json')
    if arg.frame_offset and os.path.isfile(transforms_path):
        with open(transforms_path, 'r') as f:
            existing = json.load(f)
        by_path = {frame['file_path']: frame for frame in existing.get('frames', [])}
        by_path.update({frame['file_path']: frame for frame in to_export['frames']})
        to_export['frames'] = [by_path[path] for path in sorted(by_path)]

    # Save the camera parameters
    with open(transforms_path, 'w') as f:
        json.dump(to_export, f, indent=4)
        
    if arg.save_mesh:
        # triangulate meshes
        unhide_all_objects()
        convert_to_meshes()
        triangulate_meshes()
        print('[INFO] Meshes triangulated.')
        
        # export ply mesh
        bpy.ops.wm.ply_export(filepath=os.path.join(arg.output_folder, 'mesh.ply'))

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
    parser.add_argument('--views', type=str, help='JSON string of views. Contains a list of {yaw, pitch, radius, fov} object.')
    parser.add_argument('--object', type=str, help='Path to the 3D model file to be rendered.')
    parser.add_argument('--output_folder', type=str, default='/tmp', help='The path the output will be dumped to.')
    parser.add_argument('--resolution', type=int, default=512, help='Resolution of the images.')
    parser.add_argument('--engine', type=str, default='CYCLES', help='Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')
    parser.add_argument('--geo_mode', action='store_true', help='Geometry mode for rendering.')
    parser.add_argument('--split_normal', action='store_true', help='Split the normals of the mesh.')
    parser.add_argument('--save_mesh', action='store_true', help='Save the mesh as a .ply file.')
    parser.add_argument('--frame-offset', type=int, default=0,
                        help='First output frame index; used to resume/render a view batch.')
    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    main(args)
    
