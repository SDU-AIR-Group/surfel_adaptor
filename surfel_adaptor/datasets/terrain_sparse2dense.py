import os
import json
from typing import Union
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import utils3d
from .components import StandardDatasetBase
from ..octree import DfsOctree as Octree
from ..octree import OctreeRenderer


class StructureDenser(StandardDatasetBase):
    """
    Sparse structure dataset

    Args:
        roots (str): path to the dataset
        resolution (int): resolution of the voxel grid
        min_aesthetic_score (float): minimum aesthetic score of the instances to be included in the dataset
    """

    def __init__(self,
        roots,
        resolution: int = 128,
        min_aesthetic_score: float = 5.0,
    ):
        self.resolution = resolution
        self.min_aesthetic_score = min_aesthetic_score
        self.value_range = (0, 1)

        super().__init__(roots)
        
    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[metadata[f'voxel_id'] != '']
        metadata = metadata[metadata[f'full_feat_id'] != '']
        stats['Valid data'] = len(metadata)
        # metadata = metadata[metadata[f'voxel_id'], metadata[f'full_feat_id']]
        return metadata, stats

    def get_instance(self, root, instance):
        feat_id = instance['feature']
        sparse_scan = np.load(os.path.join(root, 'full_features', f'{feat_id}.npz'), allow_pickle=True)['indices']
        sparse_scan = torch.tensor(sparse_scan, dtype=torch.long)
        ds = torch.zeros(1, self.resolution, self.resolution, self.resolution, dtype=torch.long)
        ds[:, sparse_scan[:, 0], sparse_scan[:, 1], sparse_scan[:, 2]] = 1
        
        voxel_id = instance['voxel']
        position = utils3d.io.read_ply(os.path.join(root, 'voxels', f'{voxel_id}.ply'))[0]
        coords = ((torch.tensor(position) + 0.5) * self.resolution).int().contiguous() # 映射到 (0, 128)
        ss = torch.zeros(1, self.resolution, self.resolution, self.resolution, dtype=torch.long)
        ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
        return {'ss': ss, 'ds': ds}

    @torch.no_grad()
    def visualize_sample(self, sample: Union[torch.Tensor, dict]):
        if isinstance(sample, torch.Tensor):
            sample = {'ss': sample}

        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.near = 0.8
        renderer.rendering_options.far = 1.6
        renderer.rendering_options.bg_color = (0, 0, 0)
        renderer.rendering_options.ssaa = 4
        renderer.pipe.primitive = 'voxel'
        
        # Build camera
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
        yaws = [y + yaws_offset for y in yaws]
        pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

        exts = []
        ints = []
        for yaw, pitch in zip(yaws, pitch):
            orig = torch.tensor([
                np.sin(yaw) * np.cos(pitch),
                np.cos(yaw) * np.cos(pitch),
                np.sin(pitch),
            ]).float().cuda() * 2
            fov = torch.deg2rad(torch.tensor(30)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            exts.append(extrinsics)
            ints.append(intrinsics)

        def _render_voxel_tensor(voxel_tensor: torch.Tensor) -> torch.Tensor:
            images = []
            voxel_tensor = voxel_tensor.cuda()
            for i in range(voxel_tensor.shape[0]):
                representation = Octree(
                    depth=10,
                    aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
                    device='cuda',
                    primitive='voxel',
                    sh_degree=0,
                    primitive_config={'solid': True},
                )
                coords = torch.nonzero(voxel_tensor[i, 0], as_tuple=False)
                representation.position = coords.float() / self.resolution
                representation.depth = torch.full((representation.position.shape[0], 1), int(np.log2(self.resolution)), dtype=torch.uint8, device='cuda')

                image = torch.zeros(3, 1024, 1024).cuda()
                tile = [2, 2]
                for j, (ext, intr) in enumerate(zip(exts, ints)):
                    res = renderer.render(representation, ext, intr, colors_overwrite=representation.position)
                    image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
                images.append(image)
            return torch.stack(images)

        vis_dict = {}
        if 'ss' in sample and sample['ss'] is not None:
            vis_dict['ss'] = _render_voxel_tensor(sample['ss'])
        if 'ds' in sample and sample['ds'] is not None:
            vis_dict['ds'] = _render_voxel_tensor(sample['ds'])

        return vis_dict
       
