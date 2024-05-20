import yaml
import os
import datetime
from PIL import Image
import requests

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.vqa_dataset import vqa_dataset
from dataset.randaugment import RandomAugment

from torch.utils.tensorboard import SummaryWriter

from models.infusion import InFusionModel


load_path = './saved_models/19-May_19-05-46/iter_600.pt'
config_path = './saved_models/19-May_18-51-31/config.yaml'

config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)

infusion_model = InFusionModel(
    num_hidden_layers=config['num_hidden_layers'],
    num_attention_heads = config['num_attention_heads'],
    max_position_embeddings = config['max_position_embeddings']
)

load_dict = torch.load(load_path)
infusion_model.load_state_dict(load_dict['model_state_dict'])

config = yaml.load(open('./configs/config_VQA.yaml', 'r'), Loader=yaml.Loader)

normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

train_transform = transforms.Compose([                        
        transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                            'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
        transforms.ToTensor(),
        normalize,
    ])  
test_transform = transforms.Compose([
    transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    normalize,
    ])   

train_dataset = vqa_dataset(config['train_file'], train_transform, config['vqa_root'], vg_root=None, split='train') 

def collate_fn(batch):
    image_list, question_list, answers_list, weights_list = [], [], [], []
    for item in batch:
        image_list.append(transforms.functional.to_pil_image(item[0]))
        question_list.append(item[1])
        answers_list.append(item[2][0])     ## TODO: just taking the first answer for now!!
        weights_list.append(item[3])
        
    return image_list, question_list, answers_list, weights_list

train_loader = DataLoader(train_dataset, batch_size=config['batch_size_train'], collate_fn = collate_fn)

batch = next(iter(train_loader))
outputs = infusion_model(batch[0], batch[1])

output_ids = torch.argmax(outputs, dim=2)
output_sentences = infusion_model.processor.batch_decode(output_ids)
