import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm



def resolve_npz_path(dataset_root: str, full_feat_path: str) -> str:
    full_feat_path = str(full_feat_path).strip()
    if os.path.isabs(full_feat_path):
        return full_feat_path
    return os.path.join(dataset_root, full_feat_path)



def count_voxels(npz_path: str) -> int:
    with np.load(npz_path, allow_pickle=True) as data:
        if 'indices' not in data:
            raise KeyError(f"Missing 'indices' in {npz_path}")
        return int(data['indices'].shape[0])



def main() -> None:
    parser = argparse.ArgumentParser(
        description='Populate metadata.csv num_voxels from full feature npz files.'
    )
    parser.add_argument(
        '--dataset_root',
        type=str,
        required=True,
        help='Dataset root directory containing metadata.csv',
    )
    args = parser.parse_args()

    metadata_path = os.path.join(args.dataset_root, 'metadata.csv')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f'metadata.csv not found: {metadata_path}')

    metadata = pd.read_csv(metadata_path)
    if 'full_feat_path' not in metadata.columns:
        raise ValueError("metadata.csv missing required column: 'full_feat_path'")

    num_voxels = pd.Series([pd.NA] * len(metadata), dtype='Int64')
    failed = []

    for idx, rel_path in tqdm(
        metadata['full_feat_path'].items(),
        total=len(metadata),
        desc='Counting voxels',
    ):
        if pd.isna(rel_path) or str(rel_path).strip() == '':
            failed.append((idx, 'empty full_feat_path'))
            continue

        npz_path = resolve_npz_path(args.dataset_root, rel_path)
        try:
            if not os.path.exists(npz_path):
                raise FileNotFoundError(npz_path)
            num_voxels.iloc[idx] = count_voxels(npz_path)
        except Exception as exc:
            failed.append((idx, str(exc)))

    metadata['num_voxels'] = num_voxels
    metadata.to_csv(metadata_path, index=False)

    print(f'Updated: {metadata_path}')
    print(f'Total rows: {len(metadata)}')
    print(f'Success: {len(metadata) - len(failed)}')
    print(f'Failed: {len(failed)}')
    if failed:
        print('First failed rows:')
        for idx, reason in failed[:10]:
            print(f'  row={idx}: {reason}')


if __name__ == '__main__':
    main()
