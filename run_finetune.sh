#!/bin/bash

export CUDA_VISIBLE_DEVICES=4,5,6,7

python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py \
--config ./config_VQA.yaml \
--output_dir output/ \
--checkpoint data/ALBEF_14M.pth