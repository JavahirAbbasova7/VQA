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

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

images = [image]
texts =  ["describe the photo"]

outputs = infusion_model(images, texts)

output_ids = torch.argmax(outputs, dim=2)
output_sentences = infusion_model.processor.batch_decode(output_ids)
