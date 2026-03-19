#!/bin/bash
# export CUDA_VISIBLE_DEVICES=1
python train.py --config /root/wyc/terrain_recon/terrain_model_train/surfel_adaptor/configs/train/slat_vae_enc_dec_gs_swin8_B_64l8_fp16.json --num_gpu 2 --ckpt latest --output_dir /root/wyc/terrain_recon/terrain_dataset/outputs/vae_dec_gs_log_lr_260124 --data_dir /root/wyc/terrain_recon/terrain_dataset/