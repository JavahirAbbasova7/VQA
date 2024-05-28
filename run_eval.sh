#!/bin/bash

export CUDA_VISIBLE_DEVICES=4

python -m torch.distributed.launch --nproc_per_node=1 --use_env finetune.py \
--config ./config_VQA.yaml \
--output_dir output/eval/4M_caption_27May \
--evaluate \
--checkpoint output/finetune/4M_caption_27May/checkpoint.pth