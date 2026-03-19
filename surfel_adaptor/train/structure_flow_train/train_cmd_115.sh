#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python train.py --config /home/users/wyc/surfel_adaptor/surfel_adaptor/configs/train/latent_flow_dit_L_64l8p2_fp16.json --num_gpu 1 --ckpt latest --output_dir /home/users1/wychuan/terrain_dataset/outputs/flow_init_260312 --data_dir /home/users1/wychuan/terrain_dataset 