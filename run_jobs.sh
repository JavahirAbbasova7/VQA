export CUDA_VISIBLE_DEVICES=4

python -m torch.distributed.launch --nproc_per_node 1 --rdzv-backend c10d \
--rdzv-endpoint localhost:29401 --use_env pretrain.py \
--config ./config_pretrain.yaml \
--output_dir output/pretrain/4M_caption_27May \
--checkpoint data/ALBEF_4M.pth

python -m torch.distributed.launch --nproc_per_node=1 --rdzv-backend c10d \
--rdzv-endpoint localhost:29401 --use_env finetune.py \
--config ./config_VQA.yaml \
--output_dir output/finetune/4M_caption_27May \
--checkpoint output/pretrain/4M_caption_27May/checkpoint.pth