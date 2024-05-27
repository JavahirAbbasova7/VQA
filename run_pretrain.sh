#!/bin/bash

export CUDA_VISIBLE_DEVICES=3,5,6,7

python -m torch.distributed.launch --nproc_per_node=4 --use_env pretrain.py \
--config ./config_pretrain.yaml \
--output_dir output/pretrain/4M_caption_27May \
--checkpoint data/ALBEF_4M.pth