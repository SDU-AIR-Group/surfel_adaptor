import argparse, sys, os, math, re, glob
from typing import *
import bpy
TOOLKIT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TOOLKIT_ROOT not in sys.path:
    sys.path.insert(0, TOOLKIT_ROOT)
try:
    import range_scanner
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "The range_scanner Blender add-on is not installed. Run Blender in background with "
        "tools/install_range_scanner_addon.py -- --install-pyyaml first."
    ) from exc
if not hasattr(bpy.types.Scene, "scannerProperties"):
    range_scanner.register()
from mathutils import Vector, Matrix, Euler
import numpy as np
import random
import json
import glob
import sys
from tools.camera_planning import CameraPlanner
from tools.lidar_profiles import choose_lidar
from blender_scripts.basic import *


def set_scanner_material(meshes):
    """Give imported terrain a scanner-supported material."""
    material = bpy.data.materials.new(name="ScannerTerrainMaterial")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (0.5, 0.5, 0.5, 1.0)
    bsdf.inputs['Metallic'].default_value = 0.0
    material.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    for obj in meshes:
        obj.data.materials.clear()
        obj.data.materials.append(material)


def downsample_pcl(pcl_arr, voxel_size=0.05):
    """
    Downsample point cloud using Open3D VoxelGrid method and retain intensity channel.
    
    Parameters:
    - pcl_arr (np.array): Input point cloud array of shape (N, 4), where each row is [x, y, z, intensity].
    - voxel_size (float): Voxel size for downsampling. Smaller values preserve more detail, larger values reduce point count more.
    
    Returns:
    - np.array: Downsampled point cloud of shape (M, 4), where each row is [x, y, z, intensity].
    """
    grid = np.floor(pcl_arr[:, :3] / voxel_size).astype(np.int64)
    _, first = np.unique(grid, axis=0, return_index=True)
    return pcl_arr[np.sort(first)]

def save_xyzit_to_ply(arr: np.ndarray, ply_path: str, clamp=(None, None), binary=True):
    """Write XYZ and intensity to PLY with grayscale preview colors."""
    assert arr.ndim == 2 and arr.shape[1] == 4, "arr must be (N,4)"
    xyz = arr[:, :3].astype(np.float32)
    it = arr[:, 3:4].astype(np.float32)

    pcd = o3d.t.geometry.PointCloud(o3d.core.Tensor(xyz, o3d.core.Dtype.Float32))
    pcd.point["intensity"] = o3d.core.Tensor(it, o3d.core.Dtype.Float32)

    lo = np.min(it) if clamp[0] is None else clamp[0]
    hi = np.max(it) if clamp[1] is None else clamp[1]
    denom = max(hi - lo, 1e-12)
    g = np.clip((it - lo) / denom, 0.0, 1.0).astype(np.float32)  # (N,1)
    colors = np.repeat(g, 3, axis=1)  # (N,3)
    pcd.point["colors"] = o3d.core.Tensor(colors, o3d.core.Dtype.Float32)

    o3d.t.io.write_point_cloud(ply_path, pcd, write_ascii=not binary)

def convert_to_3d_path(terrain_obj, path_2d):
    path_3d = []
    path_2d_np = np.array(path_2d)
    if path_2d_np.shape[0] == 1:
        orientation = np.zeros((path_2d_np.shape[0], 3), dtype=float)
        rand_yaw = math.radians(random.uniform(-179.999, 179.999))
        fw_p = np.array([math.cos(rand_yaw), math.sin(rand_yaw)]).reshape(1, 2)
        orientation[0, :2] = path_2d_np - fw_p
        orientation = orientation / (np.linalg.norm(orientation, axis=1, keepdims=True) + 1e-8)
    else:
        orientation = np.zeros((path_2d_np.shape[0], 3), dtype=float)
        orientation[:-1, :2] = path_2d_np[1:, :] - path_2d_np[:-1, :]
        orientation[-1] = orientation[-2]
        orientation = orientation / (np.linalg.norm(orientation, axis=1, keepdims=True) + 1e-8)
    for id, p in enumerate(path_2d):
        point_2d = Vector((p[0], p[1], 100.0))
        normal, location = get_normal_at_point(terrain_obj, point_2d)
        if normal is not None or location is not None:
            yaw_axis = Vector(orientation[id].tolist())
            z_axis = normal.normalized()
            y_axis = z_axis.cross(yaw_axis)
            x_axis = y_axis.cross(z_axis)
            rot_mat = Matrix((x_axis, y_axis, z_axis)).transposed()
            trans = np.identity(4)
            trans[:3, :3] = (rot_mat)
            trans[:3, 3] = np.array([location])
            path_3d.append(trans)
    if len(path_3d) == len(path_2d):
        return path_3d
    else:
        return []

def perturb_scan_points(
    points: np.ndarray,
    lidar_origin: np.ndarray,
    *,
    position_std: float,
    attitude_std_deg: float,
    rng: Any = np.random,
) -> np.ndarray:
    """Return a copy with xyz roll/pitch perturbation about ``lidar_origin``.

    The fourth (intensity/label) column, and any subsequent columns, are left
    untouched.  Translation is expressed in world coordinates; yaw is not
    sampled by design.
    """
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must have shape (N, 3+)")
    origin = np.asarray(lidar_origin, dtype=float)
    if origin.shape != (3,):
        raise ValueError("lidar_origin must have shape (3,)")

    translation = rng.normal(0.0, position_std, size=3)
    roll, pitch = rng.normal(0.0, math.radians(attitude_std_deg), size=2)
    cos_roll, sin_roll = math.cos(roll), math.sin(roll)
    cos_pitch, sin_pitch = math.cos(pitch), math.sin(pitch)
    rotation = np.array([
        [cos_pitch, sin_pitch * sin_roll, sin_pitch * cos_roll],
        [0.0, cos_roll, -sin_roll],
        [-sin_pitch, cos_pitch * sin_roll, cos_pitch * cos_roll],
    ])

    perturbed = np.array(points, copy=True)
    perturbed[:, :3] = (perturbed[:, :3] - origin) @ rotation.T + origin + translation
    return perturbed

def scan_once(arg) -> bool:
    """Build one random scene and write a scan only when it is valid."""
    # Initialize context
    init_render()

    init_scene()

    # Randomly Load terrain model
    if arg.terrain_model_dir is not None:
        selected_terrrain = np.random.choice(os.listdir(arg.terrain_model_dir))
        print(selected_terrrain)
        if(selected_terrrain.endswith('.glb')):
            load_object(os.path.join(arg.terrain_model_dir, selected_terrrain))
            # Avoid invalid scanning without material
            set_scanner_material([obj for obj in bpy.context.scene.objects if obj.type == 'MESH'])
            print(f"Imported Terrain objects: {bpy.context.scene.objects.values()}")
            if arg.split_normal:
                split_mesh_normal()
            print('[INFO] Scene initialized.')
            
    terrain_obj = bpy.context.scene.objects.values()[0]
    terrain_box = object_bbox(terrain_obj)

    # Sample and place auxiliary objects.
    aux_models = [f for f in os.listdir(arg.aux_model_dir) if f.lower().endswith('.glb')]
    
    selected_num = random.randint(0, 10)
    selected_models = random.sample(aux_models, min(selected_num, len(aux_models)))
    aux_objs = []
    for model_name in selected_models:
        model_path = os.path.join(arg.aux_model_dir, model_name)
        
        load_object(model_path)
        
        aux_obj = bpy.context.view_layer.objects.active
        if aux_obj is None:
            continue

        position = random.uniform(terrain_box[0][0], terrain_box[1][0]), random.uniform(terrain_box[0][1], terrain_box[1][1]), random.uniform(terrain_box[0][2], terrain_box[1][2])
        scale = random.uniform(arg.scale_factor[0], arg.scale_factor[1]), random.uniform(arg.scale_factor[2], arg.scale_factor[3]), random.uniform(arg.scale_factor[4], arg.scale_factor[5])

        # If random rotation
        is_random = random.choices([0, 1], [1-arg.random_pose_rate, arg.random_pose_rate], k=1)
        if is_random:
            rand_rpy = np.random.uniform(0, 2*math.pi, size=(3,))
            trans = np.identity(4)
            trans[:3, :3] = (Euler(rand_rpy.tolist(), 'XYZ').to_matrix())
            trans[:3, 3] = np.array([position])
            set_transform_object(aux_obj, trans.tolist(), False)
            aux_obj.scale = scale
            aux_objs.append(aux_obj)
        else:
            # Scale setting
            aux_obj.scale = scale
            # Rotation setting
            normal, inter_position = None, None
            while normal is None:
                position_xy = random.uniform(terrain_box[0][0], terrain_box[1][0]), random.uniform(terrain_box[0][1], terrain_box[1][1])
                normal, inter_position = get_normal_at_point(terrain_obj, Vector((position_xy[0], position_xy[1], terrain_box[1][2]+1.0)))
            up_vector = Vector((0, 0, 1))
            align_rotation = up_vector.rotation_difference(normal)
            trans = np.identity(4)
            trans[:3, :3] = (align_rotation.to_euler('XYZ').to_matrix())
            trans[:3, 3] = np.array([position])
            set_transform_object(aux_obj, trans.tolist(), False)
            
            # Position setting
            aux_bbox = object_bbox(aux_obj)
            z_diff = (aux_bbox[1][2] - aux_bbox[0][2]) / 2.0
            pos_l = Vector((0., 0., z_diff))
            T_w = aux_obj.matrix_world
            T_w.translation = Vector(inter_position)
            pos_w = T_w @ pos_l       
            aux_obj.location = pos_w
            aux_objs.append(aux_obj)
    
    is_select_goal = random.choices([0, 1], [0.8, 0.2], k=1)
    # is_select_goal[0] = 0  ######
    print(f"Selected goal: {is_select_goal[0]}")
    if is_select_goal[0] == 0:
        start_is_free = False
        while not start_is_free:
            start_point = random.uniform(terrain_box[0][0], terrain_box[1][0]), random.uniform(terrain_box[0][1], terrain_box[1][1])
            random_eta = random.uniform(0.2, 1.0)
            cam_planner = CameraPlanner(terrain_obj, aux_objs, start_point, (0., 0.), eta=random_eta, inflate=0.2)
            start_is_free = cam_planner.is_free(*start_point)
        # print(f"Grid size: {cam_planner.W} x {cam_planner.H}")
        is_rand_path = random.choices([0, 1], [0.6, 0.4], k=1)
        # is_rand_path[0] = 0    ######
        if is_rand_path[0] == 0:
            print("scan_type: rrt_path_planning_mode")
            path_2d = list(cam_planner.rrt_path_planning())
            print()
            print(f"path_2d : {path_2d}")

        else:
            path_2d = []
            path_nodes_num = random.randint(2,15)
            while(len(path_2d) < path_nodes_num):
                path_node = random.uniform(terrain_box[0][0], terrain_box[1][0]), random.uniform(terrain_box[0][1], terrain_box[1][1])
                if cam_planner.is_free(*path_node):
                    path_2d.append(path_node)
            if cam_planner.is_free(0.0, 0.0):
                path_2d.append((0.0, 0.0))
            else:
                print("")
                print("-----!!!  cam_planner is not free  !!!------")
                print("")
            path_2d = list(path_2d)
            print("scan_type: random_nodes_mode")
    else: 
        path_2d = [[0., 0.]] # Goal is the start
        print("scan_type: origin_only_mode")
    # path_2d = np.array([start_point, [0., 0.]], dtype=float)
    
    if isinstance(path_2d, (list, tuple)) and len(path_2d) > 0:
        print(f"###################   {path_2d[-1]}   #####################")
    else:
        print("[WARN] path_2d is empty; retry this scene.")
        return False
    
    if path_2d[-1] == (0.0, 0.0) or path_2d[-1] == [0.0, 0.0]:
    # if isinstance(path_2d, (list, tuple)) and len(path_2d) > 0:
    # if len(path_2d) != 0 and path_2d[-1] == (0., 0.):
        path_3d = convert_to_3d_path(terrain_obj, path_2d)
        if len(path_3d) == 0:
            print("[WARN] 3D path projection failed; retry this scene.")
            return False
        position_std = random.uniform(0.005, 0.1) if is_select_goal[0] == 0 else None
        attitude_std_deg = random.uniform(0.1, 3.0) if is_select_goal[0] == 0 else None
        
        pc = None
        # Camera
        cam_obj = bpy.data.objects.new(f'Camera', bpy.data.cameras.new('Camera'))
        bpy.context.collection.objects.link(cam_obj)
        bpy.context.scene.camera = cam_obj
        print(cam_obj)
        lidar = choose_lidar()
        scan_kwargs = lidar.scan_kwargs()
        print(f"LiDAR: {lidar.manufacturer} {lidar.model} ({lidar.channels} channels)")
        for i, T in enumerate(path_3d):
            T_cam_px = np.identity(4)
            T_cam_px[:3, :3] = Euler((math.radians(-90.0), math.radians(180.0), math.radians(90.0)), 'XYZ').to_matrix()
            T_cam_px[:3, 3] = np.array([0., 0., arg.lidar_height])
            trans = T @ T_cam_px
            set_transform_object(cam_obj, trans.tolist(), False)

            scan_data = range_scanner.ui.user_interface.scan_rotating(
                bpy.context, 

                scannerObject= cam_obj, # bpy.context.scene.objects["Camera"],

                **scan_kwargs,

                reflectivityLower=0.0, reflectivityUpper=255.0, maxReflectionDepth=10,
                
                enableAnimation=False, frameStart=1, frameEnd=1, frameStep=1, frameRate=1,

                addNoise=True, noiseType='gaussian', mu=0.0, noiseAbsoluteOffset=0.0, noiseRelativeOffset=0.0, 

                simulateRain=False, rainfallRate=0.0, 

                addMesh=False,

                exportLAS=False, exportHDF=False, exportCSV=False, exportPLY=False, exportSingleFrames=False,
                dataFilePath="output", dataFileName="output_file",
                
                debugLines=False, debugOutput=False, outputProgress=False, measureTime=False, singleRay=False, destinationObject=None, targetObject=None
            )
            
            pc_pieces = np.array(list(map(lambda hit: np.array([
                    hit.categoryID, hit.partID,                                             # 0, 1
                    hit.location.x, hit.location.y, hit.location.z,                         # 2, 3, 4
                    hit.distance,                                                           # 5
                    hit.intensity,                                                          # 6
                    hit.color[0], hit.color[1], hit.color[2],                               # 7, 8, 9
                    hit.noiseLocation.x, hit.noiseLocation.y, hit.noiseLocation.z,          # 10, 11, 12
                    hit.noiseDistance                                                       # 13
                ]), scan_data)))[:, [10,11,12,0]]
            if i < len(path_3d)-1 and position_std is not None and attitude_std_deg is not None:
                pc_pieces = perturb_scan_points(
                    pc_pieces, trans[:3, 3], position_std=position_std,
                    attitude_std_deg=attitude_std_deg,
                )
            if pc is None:
                pc = pc_pieces
            else:
                pc = np.concatenate([pc, pc_pieces], axis=0)    
        if pc is not None:
            pc = downsample_pcl(pc, 0.02)
            np.save(os.path.join(arg.output_folder, 'scans', 'scan.npy'), pc)
            
            try:
                if is_select_goal[0] == 1:
                    scan_type = "origin_only"
                    print("scan_type: origin_only_mode")
                    eta_value = None
                    is_rand_path_val = None
                else:
                    scan_type = "rrt_path" if is_rand_path[0] == 0 else "random_path"
                    try:
                        eta_value = float(random_eta)
                    except Exception:
                        eta_value = None
                    is_rand_path_val = int(is_rand_path[0])

                try:
                    num_nodes = int(len(path_2d))
                except Exception:
                    num_nodes = 1

                scans_dir = os.path.join(arg.output_folder, "scans")
                os.makedirs(scans_dir, exist_ok=True)
                meta_path = os.path.join(scans_dir, "scan_meta.json")
                meta = {
                    "scan_type": scan_type,                    # "rrt_path" / "random_path" / "origin_only"
                    "eta": eta_value,
                    "num_nodes": num_nodes,
                    "is_select_goal": int(is_select_goal[0]),
                    "is_rand_path": is_rand_path_val,          # None / 0 / 1
                    "seed": getattr(arg, "seed", None),
                    "lidar_model": lidar.model,
                    "lidar_manufacturer": lidar.manufacturer,
                    "lidar_channels": lidar.channels,
                    "lidar_parameters": lidar.metadata(),
                }
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False)
            except Exception as e:
                print(f"[WARN] failed to write scan_meta.json: {e}")

            print(f"Scan data length: {pc.shape}")
            if arg.save_mesh:
                save_object(os.path.join(arg.output_folder, 'model.blend'))
            return True
        print("[WARN] Scanner returned no points; retry this scene.")
        return False
    else: 
        print("Collision with start or goal, no valid path found; retry this scene.")
        return False


def main(arg):
    """Retry independent random scenes up to five times before giving up."""
    if getattr(arg, "seed", None) is not None:
        random.seed(arg.seed)
        np.random.seed(arg.seed)
    os.makedirs(os.path.join(arg.output_folder, 'scans'), exist_ok=True)

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        print(f"[INFO] Scan attempt {attempt}/{max_attempts}")
        if scan_once(arg):
            return
        if attempt < max_attempts:
            print("[INFO] Rebuilding the scene and retrying.")

    print("[WARN] No valid 3D scan path after 5 attempts; no scan was saved.")

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
    parser.add_argument('--aux-model-dir', '--aux_model_dir', dest='aux_model_dir', type=str,
                        help='Directory containing preprocessed auxiliary GLB files.')
    parser.add_argument('--terrain-model-dir', '--terrain_model_dir', dest='terrain_model_dir', type=str,
                        help='Directory containing the terrain patch GLB file.')
    parser.add_argument('--output-folder', '--output_folder', dest='output_folder', type=str, default='/tmp',
                        help='Directory for scan.npy and scan_meta.json.')
    parser.add_argument('--scale-factor', '--scale_factor', dest='scale_factor', nargs='+', type=float,
                        help='Min/max random scale ranges for auxiliary x, y, and z.')
    parser.add_argument('--random-pose-rate', '--random_pose_rate', dest='random_pose_rate', type=float,
                        help='Probability of using a random path instead of RRT.')
    parser.add_argument('--split-normal', '--split_normal', dest='split_normal', action='store_true',
                        help='Split mesh normals before scanning.')
    parser.add_argument('--lidar-height', '--lidar_height', dest='lidar_height', type=float, default=0.83,
                        help='LiDAR mounting height above the terrain (metres).')
    parser.add_argument('--save-mesh', '--save_mesh', dest='save_mesh', action='store_true',
                        help='Export the prepared mesh with the scan.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for one scan sample.')
    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)
    main(args)
    
