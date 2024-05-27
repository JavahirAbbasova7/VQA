import os
import json
from PIL import Image
from torch.utils.data import Dataset
from data_utils import pre_question

class pretrain_dataset(Dataset):
    def __init__(self, ann_file, transform, file_root_dir, max_words=30):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.max_words = max_words
        self.file_root_dir = file_root_dir
        
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        ann = self.ann[index]
        caption = pre_question(ann['caption'],self.max_words)  
        image = Image.open(os.path.join(self.file_root_dir, ann['image'])).convert('RGB')   
        image = self.transform(image)
                
        return image, caption