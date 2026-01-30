#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python train.py --config /home/users/wyc/surfel_adaptor/surfel_adaptor/configs/train/slat_vae_enc_dec_gs_swin8_B_64l8_fp16.json --num_gpu 1 --ckpt latest --output_dir /home/users1/wychuan/terrain_dataset/outputs/vae_dec_gs_gmm_260127 --data_dir /home/users1/wychuan/terrain_dataset 