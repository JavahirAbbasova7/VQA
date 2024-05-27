#!/bin/bash

export CUDA_VISIBLE_DEVICES=4,5,6,7

python finetune.py \
--config ./config_VQA.yaml \
--output_dir output/eval \
--evaluate \
--checkpoint output/finetune/checkpoint_07.pth