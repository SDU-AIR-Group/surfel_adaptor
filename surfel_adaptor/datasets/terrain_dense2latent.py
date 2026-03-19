import json
import os
from typing import *
import numpy as np
import torch
import utils3d.torch
from .components import StandardDatasetBase, TextConditionedMixin, ImageConditionedMixin
from ..modules.sparse.basic import SparseTensor
from .. import models
from ..utils.render_utils import get_renderer
from ..utils.data_utils import load_balanced_group_indices


class LatentVisMixin:
    def __init__(
        self,
        *args,
        pretrained_latent_dec: Optional[str] = None,
        latent_dec_path: Optional[str] = None,
        latent_dec_ckpt: Optional[str] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.latent_dec = None
        self.pretrained_latent_dec = pretrained_latent_dec
        self.latent_dec_path = latent_dec_path
        self.latent_dec_ckpt = latent_dec_ckpt
        
    def _loading_latent_dec(self):
        if self.latent_dec is not None:
            return
        if self.latent_dec_path is not None:
            cfg = json.load(open(os.path.join(self.latent_dec_path, 'config.json'), 'r'))
            decoder = getattr(models, cfg['models']['decoder']['name'])(**cfg['models']['decoder']['args'])
            ckpt_path = os.path.join(self.latent_dec_path, 'ckpts', f'decoder_{self.latent_dec_ckpt}.pt')
            decoder.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        else:
            decoder = models.from_pretrained(self.pretrained_latent_dec)
        self.latent_dec = decoder.cuda().eval()

    def _delete_latent_dec(self):
        del self.latent_dec
        self.latent_dec = None

    @torch.no_grad()
    def decode_latent(self, z, batch_size=4):
        self._loading_latent_dec()
        reps = []
        if self.normalization is not None:
            z = z * self.std.to(z.device) + self.mean.to(z.device)
        for i in range(0, z.shape[0], batch_size):
            reps.append(self.latent_dec(z[i:i+batch_size]))
        reps = sum(reps, [])
        self._delete_latent_dec()
        return reps

    @torch.no_grad()
    def visualize_sample(self, x_0: Union[SparseTensor, dict]):
        x_0 = x_0 if isinstance(x_0, SparseTensor) else x_0['x_0']
        reps = self.decode_latent(x_0.cuda())
        
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
            fov = torch.deg2rad(torch.tensor(40)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            exts.append(extrinsics)
            ints.append(intrinsics)

        renderer = get_renderer(reps[0])
        images = []
        for representation in reps:
            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr)
                image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['render']
            images.append(image)
        images = torch.stack(images)
            
        return images
    
    
class StructureLatent(LatentVisMixin, StandardDatasetBase):
    """
    structured latent dataset
    
    Args:
        roots (str): path to the dataset
        latent_model (str): name of the latent model
        min_aesthetic_score (float): minimum aesthetic score
        max_num_voxels (int): maximum number of voxels
        normalization (dict): normalization stats
        pretrained_latent_dec (str): name of the pretrained latent decoder
        latent_dec_path (str): path to the latent decoder, if given, will override the pretrained_latent_dec
        latent_dec_ckpt (str): name of the latent decoder checkpoint
    """
    def __init__(self,
        roots: str,
        *,
        # latent_model: str,
        min_aesthetic_score: float = 5.0,
        max_num_voxels: int = 32768,
        normalization: Optional[dict] = None,
        pretrained_latent_dec: Optional[str] = None,
        latent_dec_path: Optional[str] = None,
        latent_dec_ckpt: Optional[str] = None,
    ):
        self.normalization = normalization
        # self.latent_model = latent_model
        self.min_aesthetic_score = min_aesthetic_score
        self.max_num_voxels = max_num_voxels
        self.value_range = (0, 1)
        
        super().__init__(
            roots,
            pretrained_latent_dec=pretrained_latent_dec,
            latent_dec_path=latent_dec_path,
            latent_dec_ckpt=latent_dec_ckpt,
        )
        self.loads = [self.metadata.loc[instance['index'], 'num_voxels'] for _, instance in self.instances]
        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(1, -1)
            self.std = torch.tensor(self.normalization['std']).reshape(1, -1)
      
    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[metadata[f'latent_id'] != '']
        stats['With latent'] = len(metadata)
        # metadata = metadata[metadata['num_voxels'] <= self.max_num_voxels]
        # stats[f'Num voxels <= {self.max_num_voxels}'] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        feat_id = instance['latent']
        data = np.load(os.path.join(root, 'latents', f'{feat_id}.npz'))
        coords = torch.tensor(data['coords']).int()
        feats = torch.tensor(data['feats']).float()
        if self.normalization is not None:
            feats = (feats - self.mean) / self.std
        return {
            'coords': coords,
            'feats': feats,
        }
        
    @staticmethod
    def collate_fn(batch, split_size=None):
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices([b['coords'].shape[0] for b in batch], split_size)
        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            pack = {}
            coords = []
            feats = []
            layout = []
            start = 0
            for i, b in enumerate(sub_batch):
                coords.append(torch.cat([torch.full((b['coords'].shape[0], 1), i, dtype=torch.int32), b['coords']], dim=-1))
                feats.append(b['feats'])
                layout.append(slice(start, start + b['coords'].shape[0]))
                start += b['coords'].shape[0]
            coords = torch.cat(coords)
            feats = torch.cat(feats)
            pack['x_0'] = SparseTensor(
                coords=coords,
                feats=feats,
            )
            pack['x_0']._shape = torch.Size([len(group), *sub_batch[0]['feats'].shape[1:]])
            pack['x_0'].register_spatial_cache('layout', layout)
            
            # collate other data
            keys = [k for k in sub_batch[0].keys() if k not in ['coords', 'feats']]
            for k in keys:
                if isinstance(sub_batch[0][k], torch.Tensor):
                    pack[k] = torch.stack([b[k] for b in sub_batch])
                elif isinstance(sub_batch[0][k], list):
                    pack[k] = sum([b[k] for b in sub_batch], [])
                else:
                    pack[k] = [b[k] for b in sub_batch]
                    
            packs.append(pack)
          
        if split_size is None:
            return packs[0]
        return packs
        
    
class TextConditionedSLat(TextConditionedMixin, StructureLatent):
    """
    Text conditioned structured latent dataset
    """
    pass


class ImageConditionedSLat(ImageConditionedMixin, StructureLatent):
    """
    Image conditioned structured latent dataset
    """
    pass
