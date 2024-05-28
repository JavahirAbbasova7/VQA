#!/bin/bash

export CUDA_VISIBLE_DEVICES=6

python pretrain.py \
--config ./config_pretrain.yaml \
--output_dir output/pretrain/4M_caption_27May \
--checkpoint data/ALBEF_4M.pth