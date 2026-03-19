#!/bin/bash
# export CUDA_VISIBLE_DEVICES=1
python train.py --config /root/wyc/terrain_recon/terrain_model_train/surfel_adaptor/configs/train/denser_conv3d_16l8_fp16.json --num_gpu 2 --ckpt latest --output_dir /root/wyc/terrain_recon/terrain_dataset/outputs/denser_init_260312 --data_dir /root/wyc/terrain_recon/terrain_dataset/