#!/bin/bash

export CUDA_VISIBLE_DEVICES=6

python -m torch.distributed.launch --nproc_per_node=1 --use_env finetune.py \
--config ./config_VQA.yaml \
--output_dir output/finetune/4M_caption_25May \
--checkpoint output/pretrain/4M_caption_25May/checkpoint.pth