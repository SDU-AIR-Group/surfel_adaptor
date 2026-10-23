# `basic.py` API Reference

Use these APIs only from Blender Python.

| Function | Input → output | Purpose |
| --- | --- | --- |
| `init_render` | engine, resolution, geometry mode → scene settings | Configure the render engine and resolution. |
| `init_nodes` | requested output passes → compositor nodes | Configure compositor outputs. |
| `init_scene` | none → empty scene | Reset the current Blender scene. |
| `init_camera` | none → camera object | Add and configure the default camera. |
| `init_lighting` | none → light objects | Add the default scene lights. |
| `load_object` | model path → imported objects | Import a supported 3D model. |
| `delete_invisible_objects` | scene objects → visible objects | Remove hidden render geometry. |
| `split_mesh_normal` | mesh objects → split-normal meshes | Split normals along sharp mesh edges. |
| `delete_custom_normals` | mesh objects → reset normals | Remove custom normal data. |
| `override_material` | mesh objects → simple materials | Replace materials with a neutral material. |
| `unhide_all_objects` | scene objects → visible objects | Make every object visible. |
| `convert_to_meshes` | scene objects → mesh objects | Convert supported objects to meshes. |
| `triangulate_meshes` | mesh objects → triangle meshes | Triangulate mesh faces. |
| `scene_bbox` | scene → min/max bounds | Return bounds across all mesh objects. |
| `object_bbox` | object → min/max bounds | Return world-space object bounds. |
| `normalize_scene` | scene → scale, offset | Center and scale scene geometry. |
| `get_transform_matrix` | object → 4×4 matrix | Read the object's transform matrix. |
| `orig_init` | objects → stored transforms | Store source transforms as custom properties. |
| `scale_objects` | objects, xyz scale → scaled objects | Apply independent object scaling. |
| `transform_objects` | objects, matrices → transformed objects | Restore or apply object transform matrices. |
| `set_transform_object` | object, matrix, apply flag → object | Assign and optionally apply one transform. |
| `split_mesh_to_grid` | mesh, grid size, cell size → patch meshes | Split terrain geometry into grid patches. |
| `get_normal_at_point` | terrain mesh, point → normal, hit point | Ray-cast the terrain surface at a point. |
| `save_object` | output path, optional object → GLB file | Export an object or selected geometry. |
