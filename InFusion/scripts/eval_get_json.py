import yaml
import os
import datetime
from PIL import Image
import requests
import json

import numpy as np 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.vqa_dataset import vqa_dataset
from dataset.randaugment import RandomAugment

from torch.utils.tensorboard import SummaryWriter

from models.infusion import InFusionModel

save_dir = './saved_models/21-May_08-34-30/'
load_path = save_dir + 'epoch_1_iter_1200.pt'
config_path = save_dir + 'config.yaml'
results_path = save_dir + 'results.json'

json_path = save_dir + 'test.json'

config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)

infusion_model = InFusionModel(
    num_hidden_layers=config['num_hidden_layers'],
    num_attention_heads = config['num_attention_heads'],
    max_position_embeddings = config['max_position_embeddings']
)

load_dict = torch.load(load_path)
infusion_model.load_state_dict(load_dict['model_state_dict'])
infusion_model.eval()

config = yaml.load(open('./configs/config_VQA.yaml', 'r'), Loader=yaml.Loader)

# normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

# test_transform = transforms.Compose([
#     transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
#     transforms.ToTensor(),
#     normalize,
#     ])   

# test_dataset = vqa_dataset(config['test_file'], test_transform, config['vqa_root'], vg_root=None, split='test') 

# def collate_fn(batch):
#     image_list, question_list, answers_list, weights_list = [], [], [], []
#     for item in batch:
#         image_list.append(transforms.functional.to_pil_image(item[0]))
#         question_list.append(item[1])
#         answers_list.append(item[2][0])     ## TODO: just taking the first answer for now!!
#         weights_list.append(item[3])
        
#     return image_list, question_list, answers_list, weights_list

# test_loader = DataLoader(test_dataset, batch_size=config['batch_size_train'], collate_fn = collate_fn)

# test_json = config['test_file'][0]
test_json = '../../data/val_subset.json'
vqa_root = config['vqa_root']
with open(test_json, 'r') as f:
    test_data = json.load(f)

def process_sentence(sentence, start_token='<|startoftext|>', end_token='<|endoftext|>'):
    return sentence.replace(start_token, '').replace(end_token, '').rstrip()

results = []
with torch.no_grad():
    i=0
    for test_sample in test_data:
        print(f'\rEvaluating {i}/{len(test_data)} sample', end='')
        q_id, question, image_path = test_sample['question_id'], test_sample['question'], test_sample['image']
        pil_image = Image.open(vqa_root+image_path)
        outputs = infusion_model([pil_image], [question])
        output_ids = torch.argmax(outputs, dim=2)
        output_sentence = infusion_model.processor.batch_decode(output_ids)[0]
        output_sentence = process_sentence(output_sentence)
        result_sample = {'question_id':q_id, 'answer':output_sentence}
        results.append(result_sample)
        i+=1

with open(results_path, 'w') as f:
    json.dump(results, f)