from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...modules import sparse as sp
from ...utils.random_utils import hammersley_sequence
from .base import SparseTransformerBase
from ...gaussian import Gaussian
from ..sparse_elastic_mixin import SparseTransformerElasticMixin


class SLatGaussianDecoder(SparseTransformerBase):
    def __init__(
        self,
        resolution: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        representation_config: dict = None,
    ):
        super().__init__(
            in_channels=latent_channels,
            model_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
        )
        self.resolution = resolution
        self.rep_config = representation_config
        self.model_scale = self.rep_config['model_scale']
        self._calc_layout() # 初始化高斯的属性和模型的输出的通道, 一个通道对应一个属性的某一维度
        self.out_layer = sp.SparseLinear(model_channels, self.out_channels)
        self._build_perturbation()

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    def initialize_weights(self) -> None:
        super().initialize_weights()
        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def _build_perturbation(self) -> None:
        # 采样一组在[-1, 1]上均匀分布的噪声扰动
        perturbation = [hammersley_sequence(3, i, self.rep_config['num_gaussians']) for i in range(self.rep_config['num_gaussians'])]
        perturbation = torch.tensor(perturbation).float() * 2 - 1 # 映射到[-1, 1]
        perturbation = perturbation / self.rep_config['voxel_size']
        perturbation = torch.atanh(perturbation).to(self.device)
        self.register_buffer('offset_perturbation', perturbation)

    def _calc_layout(self) -> None:
        self.layout = {
            '_xyz' : {'shape': (self.rep_config['num_gaussians'], 3), 'size': self.rep_config['num_gaussians'] * 3},
            # '_features_dc' : {'shape': (self.rep_config['num_gaussians'], 1, 3), 'size': self.rep_config['num_gaussians'] * 3},
            '_scaling' : {'shape': (self.rep_config['num_gaussians'], 2), 'size': self.rep_config['num_gaussians'] * 2},
            '_rotation' : {'shape': (self.rep_config['num_gaussians'], 4), 'size': self.rep_config['num_gaussians'] * 4},
            '_opacity' : {'shape': (self.rep_config['num_gaussians'], 1), 'size': self.rep_config['num_gaussians']},
        }
        start = 0
        for k, v in self.layout.items():
            v['range'] = (start, start + v['size']) # 内存存储范围(相对于start)
            start += v['size']
        self.out_channels = start # 输出通道数 = 所有属性size的和
    
    def to_representation(self, x: sp.SparseTensor) -> List[Gaussian]:
        """
        Convert a batch of network outputs to 3D representations.

        Args:
            x: The [N x * x C] sparse tensor output by the network.

        Returns:
            list of representations
        """
        ret = []
        for i in range(x.shape[0]): # 分成几个batch处理
            # 初始化高斯表示
            representation = Gaussian(
                sh_degree=0,
                # aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0],
                mininum_kernel_size = self.rep_config['3d_filter_kernel_size'],
                scaling_bias = self.rep_config['scaling_bias'],
                opacity_bias = self.rep_config['opacity_bias'],
                scaling_activation = self.rep_config['scaling_activation']
            )
            xyz = (x.coords[x.layout[i]][:, 1:].float() + 0.5) / self.resolution # 稀疏栅格的中心坐标
            for k, v in self.layout.items():
                if k == '_xyz':
                    offset = x.feats[x.layout[i]][:, v['range'][0]:v['range'][1]].reshape(-1, *v['shape'])
                    # print(f"offset: {offset[:20, :]}")
                    offset = offset * self.rep_config['lr'][k]
                    # print(f"offset lr: {self.rep_config['lr'][k]}")
                    if self.rep_config['perturb_offset']:
                        offset = offset + self.offset_perturbation
                        # print(f"offset + perturb: {self.offset_perturbation}")
                    offset = torch.tanh(offset) / self.resolution * 0.5 * self.rep_config['voxel_size']
                    _xyz = xyz.unsqueeze(1) + offset # 体素中心坐标+偏移
                    # print(f"xyz: {xyz[:20, :]}")
                    # print(f"_xyz: {_xyz[:20, :]}")
                    # print(["********"]*3)
                    setattr(representation, k, _xyz.flatten(0, 1))
                else:
                    feats = x.feats[x.layout[i]][:, v['range'][0]:v['range'][1]].reshape(-1, *v['shape']).flatten(0, 1)
                    feats = feats * self.rep_config['lr'][k]
                    setattr(representation, k, feats)
            representation.zoom_gaussians(self.model_scale)
            ret.append(representation)
        # print(f"Generated gaussian num: {ret[0]._xyz.shape}")
        return ret

    def forward(self, x: sp.SparseTensor) -> List[Gaussian]:
        h = super().forward(x)
        h = h.type(x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)
        return self.to_representation(h)
    

class ElasticSLatGaussianDecoder(SparseTransformerElasticMixin, SLatGaussianDecoder):
    """
    Slat VAE Gaussian decoder with elastic memory management.
    Used for training with low VRAM.
    """
    pass
