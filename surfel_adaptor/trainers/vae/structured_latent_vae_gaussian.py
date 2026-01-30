from typing import *
import copy
import torch
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict
import utils3d.torch

from ..basic import BasicTrainer
from ...gaussian import Gaussian, GaussianRenderer
# from ...renderers import GaussianRenderer
from ...modules.sparse import SparseTensor
from ...utils.loss_utils import l1_loss, l2_loss, ssim, lpips, huber_log_loss, xyz_loss, normal_loss, scale_loss, knn_search
import matplotlib.pyplot as plt


class SLatVaeGaussianTrainer(BasicTrainer):
    """
    Trainer for structured latent VAE.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.
        
        loss_type (str): Loss type. Can be 'l1', 'l2'
        lambda_ssim (float): SSIM loss weight.
        lambda_lpips (float): LPIPS loss weight.
        lambda_kl (float): KL loss weight.
        regularizations (dict): Regularization config.
    """
    
    def __init__(
        self,
        *args,
        loss_type: str = 'l1',
        lambda_ssim: float = 0.2,
        lambda_lpips: float = 0.2,
        lambda_normal: float = 0.05,
        lambda_dist: float = 0.01,
        lambda_kl: float = 1e-6,
        lambda_gmm: float = 0.0, # 1e-4,
        regularizations: Dict = {},
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.lambda_ssim = lambda_ssim
        self.lambda_lpips = lambda_lpips
        self.lambda_normal = lambda_normal
        self.lambda_dist = lambda_dist
        self.lambda_kl = lambda_kl
        self.lambda_gmm = lambda_gmm
        self.regularizations = regularizations
        self.zoom_factor = self.models['decoder'].rep_config['model_scale']

        self._init_renderer()
        
    def _init_renderer(self):
        rendering_options = {"near" : 0.8,
                             "far" : 1.6,
                             "bg_color" : 'random'}
        self.renderer = GaussianRenderer(rendering_options)
        self.renderer.pipe.kernel_size = self.models['decoder'].rep_config['2d_filter_kernel_size']
        
    def _render_batch(self, reps: List[Gaussian], extrinsics: torch.Tensor, intrinsics: torch.Tensor, ground_image=None) -> torch.Tensor:
        """
        Render a batch of representations.

        Args:
            reps: The dictionary of lists of representations.
            extrinsics: The [N x 4 x 4] tensor of extrinsics.
            intrinsics: The [N x 3 x 3] tensor of intrinsics.
        """
        ret = None
        for i, representation in enumerate(reps):
            if ground_image is not None:
                render_pack = self.renderer.render(representation, extrinsics[i], intrinsics[i], ground_image[i])
            else:
                render_pack = self.renderer.render(representation, extrinsics[i], intrinsics[i])
            if ret is None:
                ret = {k: [] for k in list(render_pack.keys())} # + ['bg_color']}
            for k, v in render_pack.items():
                ret[k].append(v)
            # ret['bg_color'].append(self.renderer.bg_color)
        for k, v in ret.items():
            ret[k] = torch.stack(v, dim=0) 
        return ret
        
    @torch.no_grad()
    def _get_status(self, z: SparseTensor, reps: List[Gaussian]) -> Dict:
        xyz = torch.cat([g.get_xyz for g in reps], dim=0)
        xyz_base = (z.coords[:, 1:].float() + 0.5) / self.models['decoder'].resolution - 0.5
        xyz_base = xyz_base * self.models['decoder'].model_scale 
        offset = xyz - xyz_base.unsqueeze(1).expand(-1, self.models['decoder'].rep_config['num_gaussians'], -1).reshape(-1, 3)
        status = {
            'xyz': xyz,
            'offset': offset,
            'scale': torch.cat([g.get_scaling for g in reps], dim=0),
            'opacity': torch.cat([g.get_opacity for g in reps], dim=0),
        }

        for k in list(status.keys()):
            status[k] = {
                'mean': status[k].mean().item(),
                'max': status[k].max().item(),
                'min': status[k].min().item(),
            }
            
        return status
    
    def _get_regularization_loss(self, reps: List[Gaussian]) -> Tuple[torch.Tensor, Dict]:
        loss = 0.0
        terms = {}
        if 'lambda_vol' in self.regularizations and self.regularizations['lambda_vol'] != 0:
            # scale越小，loss越小
            # scales = torch.cat([g.get_scaling for g in reps], dim=0)   # [N x 2]
            # volume = torch.prod(scales, dim=1)  # [N]
            # terms[f'reg_vol'] = volume.mean()

            # Huber
            scales = torch.cat([g.get_scaling / self.zoom_factor for g in reps], dim=0)   # [N x 2]
            terms[f'reg_vol'] = huber_log_loss(scales, delta=0.5)

            loss = loss + self.regularizations['lambda_vol'] * terms[f'reg_vol']
        if 'lambda_opacity' in self.regularizations and self.regularizations['lambda_opacity'] != 0:
            opacity = torch.cat([g.get_opacity for g in reps], dim=0)
            terms[f'reg_opacity'] = (opacity).pow(2).mean()
            # terms[f'reg_opacity'] = (opacity - 0.7).pow(2).mean()
            # gauss越不透明，loss越小
            loss = loss + self.regularizations['lambda_opacity'] * terms[f'reg_opacity']
        return loss, terms

    def _get_gmm_loss(self, reps: List[Gaussian]) -> Tuple[torch.Tensor, Dict]:
        loss = 0.0
        terms = {}
        lambda_dis = 0.0
        lambda_norm = 0.8
        if self.lambda_gmm != 0:
            xyz = torch.cat([g.get_xyz for g in reps], dim=0)
            normal = torch.cat([g.get_normal for g in reps], dim=0)
            scales = torch.cat([g.get_scaling for g in reps], dim=0)
            trans = torch.cat([g.get_covariance() for g in reps], dim=0)
            knn_mask, _ = knn_search(xyz, 10)
            knn_pc = xyz[knn_mask]
            knn_n = normal[knn_mask]
            knn_s = scales[knn_mask]
            knn_trans = trans[knn_mask]
            pc_usq = torch.unsqueeze(xyz, 1)  # src_usq: (N x 1 x 3) knn: (N x k x 3)
            p_mu = pc_usq - knn_pc
            # RBF weight: (N x K)
            RBF_weights = torch.exp(-1 * torch.einsum('nki,nki->nk', p_mu, p_mu)/2.0) # 2 is assumed standard covariance of surfel

            # Position loss
            pos_loss = xyz_loss(xyz, knn_pc, knn_n, RBF_weights)
            # print(f"pos_loss = {pos_loss}")
            # Normal loss
            norm_loss = normal_loss(normal, knn_n, RBF_weights)
            # print(f"normal_loss = {norm_loss}")
            # Scale loss
            # covs: (N x 4 x 4), knn_vs: = [:, :, 0:3, 0:2] (N x k x 2 x 3), vs: [:, 0:3, 0:2] (N x 2 x 3)
            knn_vs = knn_trans[:, :, 0:3, 0:2].transpose(2, 3)
            src_vs = trans[:, 0:3, 0:2].transpose(1, 2)
            knn_vs = knn_pc.unsqueeze(2) + knn_s.unsqueeze(-1) * knn_vs
            src_vs = xyz.unsqueeze(1) + scales.unsqueeze(-1) * src_vs
            s_loss = scale_loss(scales, knn_s, src_vs, knn_vs, knn_n, RBF_weights)
            terms[f'reg_gmm'] = lambda_dis * pos_loss + lambda_norm * norm_loss + (1-lambda_dis-lambda_norm) * s_loss

            loss = loss + terms[f'reg_gmm']
        
        return loss, terms
    
    def training_losses(
        self,
        feats: SparseTensor,
        image: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        return_aux: bool = False,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses.

        Args:
            feats: The [N x * x C] sparse tensor of features.
            image: The [N x 3 x H x W] tensor of images.
            extrinsics: The [N x 4 x 4] tensor of extrinsics.
            intrinsics: The [N x 3 x 3] tensor of intrinsics.
            return_aux: Whether to return auxiliary information.

        Returns:
            a dict with the key "loss" containing a scalar tensor.
            may also contain other keys for different terms.
        """
        z, mean, logvar = self.training_models['encoder'](feats, sample_posterior=True, return_raw=True)
        reps = self.training_models['decoder'](z)
        gt_image = image
        
        # 渲染高斯表示为图片
        self.renderer.rendering_options.resolution = image.shape[-1]
        render_results = self._render_batch(reps, extrinsics, intrinsics, gt_image)     
        rec_image = render_results['render']
        # plt.imshow(gt_image[0].permute(1,2,0).cpu().detach().numpy())
        # plt.show()
        # plt.imshow(rec_image[0].permute(1,2,0).cpu().detach().numpy())
        # plt.show()
        # normal_img = (render_results['ground_normal'][0].permute(1,2,0).detach().cpu() + 1) * 127.5
        # normal_img = normal_img.clip(0, 255).numpy().astype(np.uint8)
        # plt.imshow(normal_img)
        # plt.show()
        
        terms = edict(loss = 0.0, rec = 0.0)
                
        if self.loss_type == 'l1':
            terms["l1"] = l1_loss(rec_image, gt_image)
            terms["rec"] = terms["rec"] + terms["l1"]
        elif self.loss_type == 'l2':
            terms["l2"] = l2_loss(rec_image, gt_image)
            terms["rec"] = terms["rec"] + terms["l2"]
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")
        if self.lambda_ssim > 0:
            terms["ssim"] = 1 - ssim(rec_image, gt_image)
            terms["rec"] = terms["rec"] + self.lambda_ssim * terms["ssim"]
        if self.lambda_lpips > 0:
            terms["lpips"] = lpips(rec_image, gt_image)
            terms["rec"] = terms["rec"] + self.lambda_lpips * terms["lpips"]
        terms["loss"] = terms["loss"] + terms["rec"]

        # Model loss
        normal_error = (1 - (render_results['render_normal'] * render_results['surf_normal']).sum(dim=0))[None]
        gt_normal_err = (1 - (render_results['render_normal'] * render_results['ground_normal']).sum(dim=0))[None]
        lambda_render_normal = 0.8
        terms["normal"] = lambda_render_normal*(normal_error).mean() + (1-lambda_render_normal)*(gt_normal_err).mean()
        terms["dist"] = (render_results['render_dist']).mean()
        terms["loss"] = terms["loss"] + self.lambda_normal * terms["normal"] + self.lambda_dist * terms["dist"]

        terms["kl"] = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1)
        terms["loss"] = terms["loss"] + self.lambda_kl * terms["kl"]

        # GMM loss
        gmm_loss, gmm_terms = self._get_gmm_loss(reps)
        terms.update(gmm_terms)
        terms["loss"] = terms["loss"] + self.lambda_gmm * gmm_loss

        reg_loss, reg_terms = self._get_regularization_loss(reps)
        terms.update(reg_terms)
        terms["loss"] = terms["loss"] + reg_loss
        
        status = self._get_status(z, reps)
        
        if return_aux:
            return terms, status, {'rec_image': rec_image, 'gt_image': gt_image}       
        return terms, status
    
    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )
        print(("***************************************"))
        # inference
        ret_dict = {}
        gt_images = []
        exts = []
        ints = []
        reps = []
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            args = {k: v[:batch].cuda() for k, v in data.items()}
            gt_images.append(args['image'])
            exts.append(args['extrinsics'])
            ints.append(args['intrinsics'])
            z = self.models['encoder'](args['feats'], sample_posterior=True, return_raw=False)
            reps.extend(self.models['decoder'](z))
        gt_images = torch.cat(gt_images, dim=0)
        ret_dict.update({f'gt_image': {'value': gt_images, 'type': 'image'}})
        # render single view
        exts = torch.cat(exts, dim=0)
        ints = torch.cat(ints, dim=0)
        # self.renderer.rendering_options.bg_color = (0, 0, 0)
        self.renderer.rendering_options.resolution = gt_images.shape[-1]
        render_results = self._render_batch(reps, exts, ints)
        ret_dict.update({f'rec_image': {'value': render_results['render'], 'type': 'image'}})

        # render multiview
        self.renderer.rendering_options.resolution = 512
        ## Build camera
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
        yaws = [y + yaws_offset for y in yaws]
        pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

        ## render each view
        miltiview_images = []
        for yaw, pitch in zip(yaws, pitch):
            orig = torch.tensor([
                np.sin(yaw) * np.cos(pitch),
                np.cos(yaw) * np.cos(pitch),
                np.sin(pitch),
            ]).float().cuda() * 2
            fov = torch.deg2rad(torch.tensor(30)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            extrinsics = extrinsics.unsqueeze(0).expand(num_samples, -1, -1)
            intrinsics = intrinsics.unsqueeze(0).expand(num_samples, -1, -1)
            render_results = self._render_batch(reps, extrinsics, intrinsics)
            miltiview_images.append(render_results['render'])

        ## Concatenate views
        miltiview_images = torch.cat([
            torch.cat(miltiview_images[:2], dim=-2),
            torch.cat(miltiview_images[2:], dim=-2),
        ], dim=-1)
        ret_dict.update({f'miltiview_image': {'value': miltiview_images, 'type': 'image'}})
        # self.renderer.rendering_options.bg_color = 'random'
                                    
        return ret_dict
