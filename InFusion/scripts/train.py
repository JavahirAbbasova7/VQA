import yaml
import os
import datetime
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

from dataset.vqa_dataset import vqa_dataset
from dataset.randaugment import RandomAugment

from torch.utils.tensorboard import SummaryWriter

from models.infusion import InFusionModel

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

infusion_model = InFusionModel(
    device,
    num_hidden_layers=config['num_hidden_layers'],
    num_attention_heads = config['num_attention_heads'],
    max_position_embeddings = config['max_position_embeddings']
).to(device)

total_params = sum(p.numel() for p in infusion_model.parameters())
print(f"Number of parameters: {total_params}")
trainable_params = sum(p.numel() for p in infusion_model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {trainable_params}")


epochs = config['epochs']
learning_rate = config['learning_rate']

# output_ids = torch.argmax(outputs, dim=2)
# output_sentences = infusion_model.processor.batch_decode(output_ids)
        
loss_fct = nn.CrossEntropyLoss(reduction='mean')

optimizer = optim.Adam(infusion_model.parameters(), lr=learning_rate)

# if not return_dict:
#     output = (reshaped_logits,) + outputs[2:]
#     return ((loss,) + output) if loss is not None else output

save_dir = config['save_dir']
save_path = save_dir+datetime.datetime.now().strftime("%d-%b_%H-%M-%S")
os.makedirs(save_path)
writer = SummaryWriter(log_dir=save_path)

if config['load_path']:
    print(f'loading weights from path {config["load_path"]}...')
    a = torch.load(config['load_path'])
    infusion_model.load_state_dict(a['model_state_dict'])
    optimizer.load_state_dict(a['optimizer_state_dict'])

with open(save_path+'/config.yaml', 'w') as f:
    yaml.dump(config, f)

for epoch in range(epochs):
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = infusion_model(batch[0], batch[1], )
        reshaped_logits = outputs.view(-1, outputs.shape[-1])
        
        labels = infusion_model.processor(text=batch[2], max_length=outputs.shape[1], return_tensors='pt', padding='max_length')
        labels_tensor = torch.tensor(labels.data['input_ids']).view(-1).to(device)

        loss = loss_fct(reshaped_logits, labels_tensor)
        loss.backward()
        optimizer.step()

        if (i+1)%10==0:
            print(f'Epoch {epoch} ,  iter {i} | Loss: {loss.item()}')
        writer.add_scalar("Loss/train", loss.item(), i)

        if (i+1)%config['save_freq']==0:
            torch.save({
                'epoch': epoch,
                'iter': i,
                'model_state_dict': infusion_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, save_path+f'/epoch_{epoch}_iter_{i+1}.pt')
