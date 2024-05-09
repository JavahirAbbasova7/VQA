#!/bin/bash

export CUDA_VISIBLE_DEVICES=4,5,6,7

python -m torch.distributed.launch --nproc_per_node=4 --use_env finetune.py \
--config ./config_VQA.yaml \
--output_dir output/finetune \
--checkpoint output/pretrain/may8_10epochs/checkpoint_09.pth