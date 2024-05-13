from functools import partial
from visual_encoders.vit import VisionTransformer

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.BLIP2 import LayerNorm, Blip2Base, disabled_train
import numpy as np


class ALBEF(Blip2Base):
    def __init__(self,                 
                 text_encoder = None,
                 text_decoder = None,
                 tokenizer = None,
                 config = None,     
                 ):
        super().__init__()

        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        
        self.tokenizer = self.init_tokenizer(truncation_side="left")
        self.distill = config['distill']
        num_query_token = 32
        qformer_text_input = True

        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))    
        
        self.ln_vision = LayerNorm(self.visual_encoder.num_features)  
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )

        if self.distill:
            self.visual_encoder_m = VisionTransformer(
                img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))             
            self.Qformer_m, self.query_tokens_m = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
            )

            self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                                [self.Qformer, self.Qformer_m]
                               ]
            self.copy_params() 
            self.momentum = 0.995


        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None


        self.llm_tokenizer = AutoTokenizer.from_pretrained("llm/model", truncation_side="left")
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            "llm/model", torch_dtype=torch.float16, device_map = 'auto'
        )
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        # self.llm_tokenizer.pad_token = self.llm_tokenizer.unk_token

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        # self.eos_token_id = self.llm_tokenizer(
        #     self.llm_tokenizer.eos_token, add_special_tokens=False
        # ).input_ids[0]

        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False

        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
        )

        self.max_txt_len = 128
        self.max_output_txt_len = 256
        self.prompt = ""
        prompt_tokens = self.llm_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)
        

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len
        

    def forward(self, image, quesiton, answer=None, alpha=0, k=None, weights=None, train=True):
        
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        bs = image.size(0)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        
        if train:               
            '''
            k: number of answers for each question
            weights: weight for each answer
            '''          

            # do not apply loss to the padding

            text_Qformer = self.tokenizer(
            quesiton,
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)

            self.llm_tokenizer.padding_side = "right"
            self.llm_tokenizer.truncation_side = 'left'
            text_input_tokens = self.llm_tokenizer(
                quesiton,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)

            self.llm_tokenizer.truncation_side = 'right'
            # EOS already added
            text_output_tokens = self.llm_tokenizer(
                [t + self.llm_tokenizer.eos_token for t in answer],
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_output_txt_len,
            ).to(image.device)

            llm_tokens, input_part_targets_len = self.concat_text_input_output(
                text_input_tokens.input_ids,
                text_input_tokens.attention_mask,
                text_output_tokens.input_ids,
                text_output_tokens.attention_mask,
            )
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
            targets = llm_tokens['input_ids'].masked_fill(
                llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
            )

            # do not apply loss to the text input (i.e., instruction)
            for i, l in enumerate(input_part_targets_len):
                targets[i][:l] = -100

            # answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)      

            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            ) 

            question_states = []                
            question_atts = []  
            for b, n in enumerate(k):
                question_states += [query_output.last_hidden_state[b]]*n
                question_atts += [quesiton.attention_mask[b]]*n 
            question_states = torch.stack(question_states,0)    
            question_atts = torch.stack(question_atts,0)   

            inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)

            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

            # do not apply loss to the query tokens
            empty_targets = (
                torch.ones(atts_llm.size(), dtype=torch.long).to(image.device).fill_(-100)
            )
            targets = torch.cat([empty_targets, targets], dim=1)  

            if self.distill:                    
                with torch.no_grad():
                    self._momentum_update()
                    image_embeds_m = self.visual_encoder_m(image) 
                    query_output_m = self.Qformer_m.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=image_embeds_m,
                        encoder_attention_mask=image_atts,
                        return_dict=True,
                    )  

                    question_states_m = []                
                    for b, n in enumerate(k):
                        question_states_m += [query_output_m.last_hidden_state[b]]*n
                    question_states_m = torch.stack(question_states_m,0)    

                    inputs_llm_m = self.llm_proj(query_output_m.last_hidden_state[:,:query_tokens.size(1),:])

                    inputs_embeds_m = torch.cat([inputs_llm_m, inputs_embeds], dim=1)
                    attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)

                    logits_m = self.llm_model(
                        inputs_embeds=inputs_embeds_m,
                        attention_mask=attention_mask,
                        encoder_hidden_states = question_states_m,
                        encoder_attention_mask = question_atts,  
                        return_logits = True,
                    )                    

                answer_output = self.llm_model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        encoder_hidden_states = question_states,
                        encoder_attention_mask = question_atts,     
                        return_dict = True,  
                        soft_labels = F.softmax(logits_m,dim=-1), 
                        alpha = alpha,
                        reduction = 'none',
                        labels=targets,
                    )    
                                                 
            else:
                answer_output = self.llm_model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        encoder_hidden_states = question_states,
                        encoder_attention_mask = question_atts,     
                        return_dict = True,  
                        reduction = 'none',
                        labels=targets,
                    )    

            loss = weights * answer_output.loss         
            loss = loss.sum()/image.size(0)      

        else: 

            if isinstance(quesiton, str):
                quesiton = [quesiton]


            prompt = "Question: {} Answer:"
            prompt = [prompt.format(question) for question in quesiton]
            if isinstance(prompt, str):
                prompt = [prompt] * bs

            text_Qformer = self.tokenizer(
                prompt,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

            query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            
            inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)


            llm_tokens = self.llm_tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt"
            ).to(image.device)

            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=False,
                top_p=0.9,
                temperature=1,
                num_beams=5,
                max_length=256,
                min_length=1,
                # eos_token_id=self.eos_token_id,
                repetition_penalty=1.5,
                length_penalty=1,
                num_return_sequences=1,
            )

            outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
            output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_text = [text.strip() for text in output_text]

            return output_text


    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
                
    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):
        
        num_ques = question_states.size(0)
        start_ids = answer_ids[0,0].repeat(num_ques,1) # bos token

        answer_output = self.llm_model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        encoder_hidden_states = question_states,
                        encoder_attention_mask = question_atts,     
                        return_dict = True,  
                        reduction = 'none',
                    ) 
        
        start_output = self.text_decoder(start_ids, 
                                         encoder_hidden_states = question_states,
                                         encoder_attention_mask = question_atts,                                      
                                         return_dict = True,
                                         reduction = 'none')              
        logits = start_output.logits[:,0,:] # first token's logit
        
        # topk_probs: top-k probability 
        # topk_ids: [num_question, k]        
        answer_first_token = answer_ids[:,1]
        prob_first_token = F.softmax(logits,dim=1).index_select(dim=1, index=answer_first_token) 
        topk_probs, topk_ids = prob_first_token.topk(k,dim=1) 
        
        # answer input: [num_question*k, answer_len]                 
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids,dim=0)  
        input_atts = torch.cat(input_atts,dim=0)  

        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)
        
        output = self.text_decoder(input_ids, 
                                   attention_mask = input_atts, 
                                   encoder_hidden_states = question_states,
                                   encoder_attention_mask = question_atts,     
                                   labels = targets_ids,
                                   return_dict = True, 
                                   reduction = 'none')                 

        answer_loss = output.loss 
        answer_loss = answer_loss.view(input_ids.size(0),-1)
        
        # topk_prob: first token probability
        topk_probs = topk_probs.view(-1,1)
        log_probs = torch.cat([topk_probs.log(), -answer_loss],dim=1)

        # re-calculate log probabilities for the answer sequences using chain rule
        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques,k)

        topk_probs = F.softmax(log_probs_sum, dim=-1)
        # get top-k after re-ranking
        topk_probs, rerank_id = topk_probs.topk(k,dim=1) 
        topk_ids = torch.gather(topk_ids, 1, rerank_id)    

        return topk_ids, topk_probs
    
def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))    