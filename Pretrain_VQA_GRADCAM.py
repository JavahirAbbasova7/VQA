'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_pretrain import ALBEF
from visual_encoders.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer

from functools import partial
from visual_encoders.vit import VisionTransformer
from text_encoders.xbert import BertConfig, BertModel
from models.tokenization_bert import BertTokenizer
import utils

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms


from PIL import Image

import cv2
import numpy as np

from skimage import transform as skimage_transform
from scipy.ndimage import filters
from matplotlib import pyplot as plt

def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
    
    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for i, (image, text) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        optimizer.zero_grad()
  
        image = image.to(device,non_blocking=True) 

        text_input = tokenizer(text, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(device)  
        
        if epoch>0:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader)) 
        
        loss_mlm, loss_ita, loss_itm = model(image, text_input, alpha = alpha)  
            
        loss = loss_mlm + loss_ita + loss_itm    
          
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])         
        
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} 


def getAttMap(img, attMap, blur = True, overlap = True):
    attMap -= attMap.min()
    if attMap.max() > 0:
        attMap /= attMap.max()
    attMap = skimage_transform.resize(attMap, (img.shape[:2]), order = 3, mode = 'constant')
    if blur:
        attMap = filters.gaussian_filter(attMap, 0.02*max(img.shape[:2]))
        attMap -= attMap.min()
        attMap /= attMap.max()
    cmap = plt.get_cmap('jet')
    attMapV = cmap(attMap)
    attMapV = np.delete(attMapV, 3, 2)
    if overlap:
        attMap = 1*(1-attMap**0.7).reshape(attMap.shape + (1,))*img + (attMap**0.7).reshape(attMap.shape+(1,)) * attMapV
    return attMap

import re

def pre_caption(caption,max_words=30):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])            
    return caption
def patch_alignment(visual_patch_proj, text_cls_proj): # shapes =  [B, 196, 768], [B, 1, 768]

    # normalize visual patch tokens and then permute
    normalized_visual_patch_proj = F.normalize(visual_patch_proj, dim=-1)
    normalized_visual_patch_proj = normalized_visual_patch_proj.transpose(-2,-1) # shapes =  [B, 768, 196]
    # normalize text cls token and unsqueeze (required for matmul)

    normalized_text_cls_proj = text_cls_proj.unsqueeze(1)

    # compute dot product
    patch_activations = normalized_text_cls_proj @ normalized_visual_patch_proj # shapes =  [B, 1, 196]
    patch_activations = patch_activations.squeeze() # shapes =  [B, 196]
    # because of dot product, the range is between -1 (least similar) to +1 (most similar)
    # multiply by 10 and apply sigmoid function. this squashes the range from 0 to 1 for every element (not necessarily sums to 1 like that of a softmax function)
    return F.sigmoid(patch_activations*10)

def order_to_coordinates(order):
    x = order % 16
    y = order // 16
    return x * 16, y * 16

def val(model, data_loader, tokenizer, device, block_num):
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    print_freq = 50
     
    for i, (image, question) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    # for i, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device)
        print("TEXT: ", question)
        print(len(question))
        text_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)  
        print(len(text_input))
 
        image_embeds = model.visual_encoder(image, register_blk=block_num) 
        image_feat = model.vision_proj(image_embeds[:,1:,:])
        print(image_feat.size())

        text_output = model.text_encoder.bert(text_input.input_ids, attention_mask = text_input.attention_mask,                 
                                            return_dict = True, mode = 'text') 
        # print(text_output)           
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(model.text_proj(text_embeds[:,0,:]),dim=-1)     
        # sim = image_feat@text_feat.t()/model.temp
        # loss = sim.diag().sum()

        patch_activations = patch_alignment(image_feat, text_feat)
        # most_important_patch = torch.argmax(patch_activations, dim=-1)
        _, most_important_patches = torch.topk(patch_activations, 3, dim=-1)
        print("IMPORTANT ", most_important_patches.size())
        patch_pooled_visual_projections = torch.sum(image_feat * patch_activations.unsqueeze(-1), dim=1) # [B, 768]
        norm_img_feat = F.normalize(patch_pooled_visual_projections, dim=-1)
        print(norm_img_feat.size())

        sim = norm_img_feat @ text_feat.t() / model.temp 
        print(sim.size())

        loss = sim.diag().sum()

        # image_embeds = model.visual_encoder(image, register_blk=block_num) 
        # print(image_embeds.size())
        # image_feat = F.normalize(model.vision_proj(image_embeds[:,0,:]),dim=-1) 
        # print(image_feat.size())
        # text_output = model.text_encoder.bert(text_input.input_ids, attention_mask = text_input.attention_mask,                 
        #                                     return_dict = True, mode = 'text')            
        # text_embeds = text_output.last_hidden_state
        # print(text_embeds.size())

        # text_feat = F.normalize(model.text_proj(text_embeds[:,0,:]),dim=-1)  
        # print(text_feat.size())   
        # sim = image_feat@text_feat.t()/model.temp
        # print(sim.size())
        # loss = sim.diag().sum()
        
        model.zero_grad()
        loss.backward()  
        
        # with torch.no_grad():
        #     grad = model.visual_encoder.blocks[block_num].attn.get_attn_gradients().detach()
        #     cam = model.visual_encoder.blocks[block_num].attn.get_attention_map().detach()
        #     cam = cam[:, :, 0, 1:].reshape(image.size(0), -1, 16, 16)
        #     grad = grad[:, :, 0, 1:].reshape(image.size(0), -1, 16, 16).clamp(0)
        #     gradcam = (cam * grad).mean(1).to("cpu")

        # for b in range (image.size()[0]):
        #     rgb_image = image[b].squeeze().permute(1, 2, 0).to("cpu")
        #     rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
        #     rgb_image = np.float32(rgb_image)
        #     fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        #     gradcam_np = gradcam[b].numpy().astype(np.float32)
        #     gradcam_image = getAttMap(rgb_image, gradcam_np)
        #     ax.imshow(gradcam_image)
        #     ax.set_yticks([])
        #     ax.set_xticks([])
        #     ax.set_xlabel(question[b])
        #     path_save_image = f"Gradcam_PACL_COCO_CLS/TEXT_{i}_{b}.png"
        #     fig.savefig(path_save_image, bbox_inches='tight')

        with torch.no_grad():
            grad = model.visual_encoder.blocks[block_num].attn.get_attn_gradients().detach()
            cam = model.visual_encoder.blocks[block_num].attn.get_attention_map().detach()

        for b in range (image.size()[0]):
            rgb_image = image[b].squeeze().permute(1, 2, 0).to("cpu")
            rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
            rgb_image = np.float32(rgb_image)
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            print(most_important_patches[b])

            # final_cam = cam[:, :, most_important_patch[b]+1, 1:].reshape(image.size(0), -1, 16, 16)
            # final_grad = grad[:, :, most_important_patch[b]+1, 1:].reshape(image.size(0), -1, 16, 16).clamp(0)
            # final_cam = torch.mean(cam[:, :, most_important_patches[b]+1, 1:], dim=2).reshape(image.size(0), -1, 16, 16)
            # final_grad = torch.mean(grad[:, :, most_important_patches[b]+1, 1:], dim=2).reshape(image.size(0), -1, 16, 16).clamp(0)
            # gradcam = (final_cam * final_grad).mean(1).to("cpu")
            final_cam = (cam[:, :, 1:, 1:]*patch_activations[b].view(1, 1, 256, 1)).sum(dim=2).reshape(image.size(0), -1, 16, 16)  
            final_grad = (grad[:, :, 1:, 1:]*patch_activations[b].view(1, 1, 256, 1)).sum(dim=2).reshape(image.size(0), -1, 16, 16).clamp(0)
            print("FINAL CAM: ", final_cam.size())
            gradcam = (final_cam * final_grad).mean(1).to("cpu").detach()
            print(gradcam.size())
            gradcam_np = gradcam[b].numpy().astype(np.float32)
            gradcam_image = getAttMap(rgb_image, gradcam_np)

            ax.imshow(gradcam_image)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_xlabel(question[b])
            path_save_image = f"Gradcam_PACL_COCO/TEXT_{i}_{b}.png"
            fig.savefig(path_save_image, bbox_inches='tight')

    
    
def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']    

    #### Dataset #### 
    print("Creating dataset")
    datasets = create_dataset('pretrain', config)
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)         
    else:
        samplers = [None]

    data_loader, test_loader = create_loader(datasets,samplers,
                                            batch_size=[config['batch_size'],config['batch_size']],
                                            num_workers=[4,4],is_trains=[True, False], 
                                            collate_fns=[None,None]) 
                                            

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model #### 
    print("Creating model")
    model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, init_deit=True)
    
    model = model.to(device)   
        
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  

    
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']                       
        if args.resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch']+1         
        else:
            pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)   
            m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)  
            state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped       
            state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped               
        model.load_state_dict(state_dict)    
        print('load checkpoint from %s'%args.checkpoint)
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    

    if not args.evaluate:
    
        print("Start training")
        start_time = time.time()

        for epoch in range(start_epoch, max_epoch):
            
            if epoch>0:
                lr_scheduler.step(epoch+warmup_steps)  
                
            train_stats = train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config) 
            if utils.is_main_process():  
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch,
                            }                     
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))  
                
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            dist.barrier()  
                    
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str)) 
    else:
        val(model_without_ddp, data_loader, tokenizer, device, 8)
    
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain.yaml')
    parser.add_argument('--checkpoint', default='output/Pretrain/checkpoint_14.pth') 
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--evaluate', action='store_true') 
    parser.add_argument('--output_dir', default='Pretrain/')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    yml = yaml.YAML(typ='rt')
    config = yml.load(open(args.config, 'r'))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))  

    main(args, config)