import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# from text_encoders.xbert import BertModel, BertConfig, BertLMHeadModel
# from visual_encoders.vit import VisionTransformer    

class AddBlock(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, embedding1, embedding2):
        return embedding1+embedding2

class ConcatResizeBlock(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.emb_size = emb_size
        self.linear = nn.Linear(2*emb_size, emb_size)
    def forward(self, embedding1, embedding2):
        concat = torch.cat((embedding1, embedding2), dim=1)
        resized = self.linear(concat)
        return resized

class ProductBlock(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, embedding1, embedding2):
        return embedding1*embedding2

def get_fusion_blocks(fusion_type, **kwargs):
    if fusion_type=='add':
        return AddBlock()
    elif fusion_type=='concat_resize':
        return ConcatResizeBlock(emb_size=kwargs['emb_size'])
    elif fusion_type=='product':
        return ProductBlock()
    else:
        print('Enter valid fusion type')
        raise NotImplementedError

class InFusionBlockv1(nn.Module):
    def __init__(self, emb_size=768, output_emb_size=768, fusion_types=['add', 'concat_resize', 'product']):
        super().__init__()
        self.fusion_blocks = [get_fusion_blocks(fusion_type, emb_size=emb_size) for fusion_type in fusion_types]
        self.linear = nn.Linear(len(fusion_types)*emb_size, output_emb_size)

    def forward(self, image_embedding, text_embedding):
        # img_embedding and text embedding both oare of shape (N, emb_dim)
        fusion_outputs = [fusion_block(image_embedding, text_embedding) for fusion_block in self.fusion_blocks]
        fusion_concat = torch.cat(tuple(fusion_outputs), dim=1)
        infusion_output = self.linear(fusion_concat)
        return infusion_output
    
# class InFusionModelVQA(nn.Module):
#     def __init__(self):
#         self.visual_encoder = VisionTransformer()   
#         self.text_encoder = BertModel.from_pretrained('bert-base-uncased', config='./configs/config_bert.json')
#         config_decoder = BertConfig.from_json_file(config['bert_config'])
#         config_decoder.fusion_layer = 0
#         config_decoder.num_hidden_layers = 6
#         self.text_decoder = BertLMHeadModel.from_pretrained('bert-base-uncased', config=config_decoder)    
#         # TODO: Load Albef weights

#         self.infusion_block = InFusionBlockv1()

#     def forward(self, inputs):
#         img, question, answers, weights = inputs
#         img_embs = self.visual_encoder(img.unsqueeze(dim=0))
#         text_embs = self.text_encoder(question.unsqueeze(dim=0))

#         img_cls_emb = img_embs[:,0]
#         text_cls_emb = text_embs[0][:,0]
#         fusion_output = self.infusion_block(img_cls_emb, text_cls_emb)

#         ## add fusion to every emb (currently selected)
#         ## or concat
#         ## or have it as separate embeddings in the sequence (that is, increase the sequence length by one)
#         # img_embs

#     def forward(self, image, quesiton, answer=None, alpha=0, k=None, weights=None, train=True):
        
#         image_embeds = self.visual_encoder(image) 
#         image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        
#         if train:               
#             '''
#             k: number of answers for each question
#             weights: weight for each answer
#             '''          
#             answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)      

#             question_output = self.text_encoder(quesiton.input_ids, 
#                                                 attention_mask = quesiton.attention_mask, 
#                                                 # encoder_hidden_states = image_embeds,
#                                                 # encoder_attention_mask = image_atts,                             
#                                                 return_dict = True)    

#             question_states = []                
#             question_atts = []  
#             for b, n in enumerate(k):
#                 question_states += [question_output.last_hidden_state[b]]*n
#                 question_atts += [quesiton.attention_mask[b]]*n 
#             question_states = torch.stack(question_states,0)    
#             question_atts = torch.stack(question_atts,0)     

#             if self.distill:                    
#                 with torch.no_grad():
#                     self._momentum_update()
#                     image_embeds_m = self.visual_encoder_m(image) 
#                     question_output_m = self.text_encoder_m(quesiton.input_ids, 
#                                                             attention_mask = quesiton.attention_mask, 
#                                                             encoder_hidden_states = image_embeds_m,
#                                                             encoder_attention_mask = image_atts,                             
#                                                             return_dict = True)    

#                     question_states_m = []                
#                     for b, n in enumerate(k):
#                         question_states_m += [question_output_m.last_hidden_state[b]]*n
#                     question_states_m = torch.stack(question_states_m,0)    

#                     logits_m = self.text_decoder_m(answer.input_ids, 
#                                                    attention_mask = answer.attention_mask, 
#                                                    encoder_hidden_states = question_states_m,
#                                                    encoder_attention_mask = question_atts,                                  
#                                                    return_logits = True,
#                                                   )                       

#                 answer_output = self.text_decoder(answer.input_ids, 
#                                                   attention_mask = answer.attention_mask, 
#                                                   encoder_hidden_states = question_states,
#                                                   encoder_attention_mask = question_atts,                  
#                                                   labels = answer_targets,
#                                                   return_dict = True,   
#                                                   soft_labels = F.softmax(logits_m,dim=-1),
#                                                   alpha = alpha,
#                                                   reduction = 'none',
#                                                  )   
#             else:
#                 answer_output = self.text_decoder(answer.input_ids, 
#                                                   attention_mask = answer.attention_mask, 
#                                                   encoder_hidden_states = question_states,
#                                                   encoder_attention_mask = question_atts,                  
#                                                   labels = answer_targets,
#                                                   return_dict = True,   
#                                                   reduction = 'none',
#                                                  )                      
#             loss = weights * answer_output.loss         
#             loss = loss.sum()/image.size(0)

#             return loss
            

#         else: 
#             question_output = self.text_encoder(quesiton.input_ids, 
#                                                 attention_mask = quesiton.attention_mask, 
#                                                 encoder_hidden_states = image_embeds,
#                                                 encoder_attention_mask = image_atts,                                    
#                                                 return_dict = True)                    
#             topk_ids, topk_probs = self.rank_answer(question_output.last_hidden_state, quesiton.attention_mask, 
#                                                     answer.input_ids, answer.attention_mask, k) 
#             return topk_ids, topk_probs
    

if __name__=='__main__':
    # import ruamel_yaml as yaml
    import yaml
    import os

    from torchvision import transforms
    from PIL import Image

    from dataset.vqa_dataset import vqa_dataset
    from dataset.randaugment import RandomAugment

    batch_size = 10
    img_size = 224
    text_seq = 17
    img = torch.randn((batch_size, 3, img_size, img_size))
    text = torch.randint(1000, size=(batch_size, text_seq))

    config_bert = {
    "architectures": [
        "BertForMaskedLM"
    ],
    "attention_probs_dropout_prob": 0.1,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "layer_norm_eps": 1e-12,
    "max_position_embeddings": 512,
    "model_type": "bert",
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "pad_token_id": 0,
    "type_vocab_size": 2,
    "vocab_size": 30522,
    "fusion_layer": 6,
    "encoder_width": 768
    }

    bert = BertModel.from_pretrained('bert-base-uncased', config='./configs/config_bert.json')
    vit = VisionTransformer()
    infusion = InFusionBlockv1()

    img_embs = vit(img)
    text_embs = bert(text)

    img_cls_emb = img_embs[:,0]
    text_cls_emb = text_embs[0][:,0]
    fusion_output = infusion(img_cls_emb, text_cls_emb)

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
    # vqa_test_dataset = vqa_dataset(config['test_file'], test_transform, config['vqa_root'], vg_root=None, split='test', answer_list=config['answer_list'])       
    
        
    # train_loader = DataLoader(train_dataset, batch_size=2)
    # batch = next(iter(train_loader))

    img, question, answers, weights = train_dataset[0]
    img_embs = vit(img.unsqueeze(dim=0))
    ## need to tokenize first
    text_embs = bert(question.unsqueeze(dim=0))

    img_cls_emb = img_embs[:,0]
    text_cls_emb = text_embs[0][:,0]
    fusion_output = infusion(img_cls_emb, text_cls_emb)
