#!/bin/bash

export CUDA_VISIBLE_DEVICES=4,5,6,7

python -m torch.distributed.launch --nproc_per_node=4 --use_env pretrain.py \
--config ./config_pretrain.yaml \
--output_dir output/ \
--checkpoint data/ALBEF_14M.pth