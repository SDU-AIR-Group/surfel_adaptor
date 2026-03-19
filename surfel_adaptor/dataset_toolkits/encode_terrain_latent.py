import os
import sys
import json
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from easydict import EasyDict as edict
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import surfel_adaptor.models as models
import surfel_adaptor.modules.sparse as sp


torch.set_grad_enabled(False)


def load_encoder(opt):
    if opt.enc_model is None:
        encoder = models.from_pretrained(opt.enc_pretrained).eval().cuda()
    else:
        cfg_path = os.path.join(opt.model_root, opt.enc_model, 'config.json')
        cfg = edict(json.load(open(cfg_path, 'r')))
        encoder = getattr(models, cfg.models.encoder.name)(**cfg.models.encoder.args).cuda()
        ckpt_path = os.path.join(opt.model_root, opt.enc_model, 'ckpts', f'encoder_{opt.ckpt}.pt')
        encoder.load_state_dict(torch.load(ckpt_path), strict=False)
        encoder.eval()
        print(f'Loaded model from {ckpt_path}')
    return encoder


def build_patch_id_list(df):
    patch_ids = []
    seen = set()
    for value in df['patch_id'].values.tolist():
        if pd.isna(value):
            continue
        patch_id = str(value)
        if patch_id in seen:
            continue
        seen.add(patch_id)
        patch_ids.append(patch_id)
    return patch_ids


def build_patch_to_feat(df):
    feat_lookup = df[['patch_id', 'full_feat_id']].dropna()
    if len(feat_lookup) == 0:
        return {}
    feat_lookup = feat_lookup.copy()
    feat_lookup['patch_id'] = feat_lookup['patch_id'].astype(str)
    feat_lookup['full_feat_id'] = feat_lookup['full_feat_id'].astype(str)
    feat_lookup = feat_lookup.drop_duplicates(subset=['patch_id'])
    return dict(zip(feat_lookup['patch_id'], feat_lookup['full_feat_id']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', '--output_dir', dest='dataset_root', type=str, required=True,
                        help='Dataset root directory containing metadata.csv')
    parser.add_argument('--enc_pretrained', type=str,
                        default='microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16',
                        help='Pretrained encoder model')
    parser.add_argument('--model_root', type=str, default='results',
                        help='Root directory of models')
    parser.add_argument('--enc_model', type=str, default=None,
                        help='Encoder model. If specified, use this model instead of pretrained model')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Checkpoint to load when using --enc_model')
    parser.add_argument('--instances', type=str, default=None,
                        help='Optional file listing patch_id values to process')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing latents if present')
    opt = parser.parse_args()
    opt = edict(vars(opt))

    metadata_path = os.path.join(opt.dataset_root, 'metadata.csv')
    if not os.path.exists(metadata_path):
        raise ValueError('metadata.csv not found')

    metadata = pd.read_csv(metadata_path)
    required_cols = {'patch_id', 'full_feat_id'}
    if not required_cols.issubset(set(metadata.columns)):
        missing = required_cols - set(metadata.columns)
        raise ValueError(f'metadata.csv missing columns: {sorted(missing)}')

    if opt.instances is not None:
        with open(opt.instances, 'r') as f:
            patch_ids_filter = [line.strip() for line in f if line.strip()]
        scope = metadata[metadata['patch_id'].isin(patch_ids_filter)]
    else:
        scope = metadata

    patch_ids = build_patch_id_list(scope)
    if len(patch_ids) == 0:
        print('No patch_id entries to process.')
        sys.exit(0)

    patch_to_feat = build_patch_to_feat(metadata)

    dup_feat = metadata.groupby('patch_id')['full_feat_id'].nunique()
    inconsistent = dup_feat[dup_feat > 1].index.tolist()
    inconsistent = [pid for pid in inconsistent if pid in patch_ids]
    if len(inconsistent) != 0:
        print(f'Warning: patch_id has multiple full_feat_id values, using the first occurrence: {len(inconsistent)}')

    latents_dir = os.path.join(opt.dataset_root, 'latents')
    os.makedirs(latents_dir, exist_ok=True)

    existing_patch_ids = set()
    missing_inputs = set()
    to_process = []
    for patch_id in patch_ids:
        output_path = os.path.join(latents_dir, f'latent_{patch_id}.npz')
        if not opt.overwrite and os.path.exists(output_path):
            existing_patch_ids.add(patch_id)
            continue
        full_feat_id = patch_to_feat.get(patch_id)
        if full_feat_id is None or pd.isna(full_feat_id):
            missing_inputs.add(patch_id)
            continue
        input_path = os.path.join(opt.dataset_root, 'full_features', f'{full_feat_id}.npz')
        if not os.path.exists(input_path):
            missing_inputs.add(patch_id)
            continue
        to_process.append((patch_id, input_path, output_path))

    print(f'Total unique patch_ids in scope: {len(patch_ids)}')
    print(f'Existing latents: {len(existing_patch_ids)}')
    print(f'To process: {len(to_process)}')
    if len(missing_inputs) != 0:
        print(f'Missing inputs: {len(missing_inputs)}')

    generated_patch_ids = set()
    failed_patch_ids = set()
    feature_sum = None
    feature_sq_sum = None
    feature_count = 0
    if len(to_process) != 0:
        encoder = load_encoder(opt)
        for patch_id, input_path, output_path in tqdm(to_process, desc='Extracting terrain latents'):
            try:
                with np.load(input_path) as feats:
                    if 'patchtokens' not in feats or 'indices' not in feats:
                        raise KeyError('Missing patchtokens/indices in full feature file')
                    feats_tensor = sp.SparseTensor(
                        feats=torch.from_numpy(feats['patchtokens']).float(),
                        coords=torch.cat([
                            torch.zeros(feats['patchtokens'].shape[0], 1).int(),
                            torch.from_numpy(feats['indices']).int(),
                        ], dim=1),
                    ).cuda()
                latent = encoder(feats_tensor, sample_posterior=False)
                assert torch.isfinite(latent.feats).all(), "Non-finite latent"
                latent_feats = latent.feats.detach().cpu().float()
                latent_feats_sum = latent_feats.sum(dim=0, dtype=torch.float64)
                latent_feats_sq_sum = latent_feats.pow(2).sum(dim=0, dtype=torch.float64)
                if feature_sum is None:
                    feature_sum = latent_feats_sum
                    feature_sq_sum = latent_feats_sq_sum
                else:
                    feature_sum += latent_feats_sum
                    feature_sq_sum += latent_feats_sq_sum
                feature_count += latent_feats.shape[0]
                pack = {
                    'feats': latent_feats.numpy().astype(np.float32),
                    'coords': latent.coords[:, 1:].cpu().numpy().astype(np.uint8),
                }
                np.savez_compressed(output_path, **pack)
                generated_patch_ids.add(patch_id)
            except Exception as e:
                print(f'Error processing patch_id {patch_id}: {e}')
                failed_patch_ids.add(patch_id)

    if len(failed_patch_ids) != 0:
        print(f'Failed to process: {len(failed_patch_ids)}')

    if feature_count > 0:
        feature_mean = feature_sum / feature_count
        feature_var = torch.clamp_min(feature_sq_sum / feature_count - feature_mean.pow(2), 0.0)
        feature_std = torch.sqrt(feature_var)
        print(f'Global latent mean: {feature_mean.tolist()}')
        print(f'Global latent std: {feature_std.tolist()}')

    success_patch_ids = existing_patch_ids | generated_patch_ids
    if 'latent_id' not in metadata.columns:
        metadata['latent_id'] = pd.NA
    if 'latent_path' not in metadata.columns:
        metadata['latent_path'] = pd.NA

    if len(success_patch_ids) != 0:
        mask = metadata['patch_id'].isin(success_patch_ids)
        metadata.loc[mask, 'latent_id'] = metadata.loc[mask, 'patch_id'].apply(
            lambda x: f'latent_{x}'
        )
        metadata.loc[mask, 'latent_path'] = metadata.loc[mask, 'patch_id'].apply(
            lambda x: f'latents/latent_{x}.npz'
        )

    metadata.to_csv(metadata_path, index=False)
