from models.infusion_block import InFusionBlockv1

import torch
import torch.nn as nn

from transformers import CLIPProcessor, CLIPModel, VisualBertModel, VisualBertConfig
from transformers.modeling_outputs import SequenceClassifierOutput

class InFusionModel(nn.Module):
    def __init__(
            self,
            num_hidden_layers = 4,
            num_attention_heads = 4,
            max_position_embeddings = 128
            ):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        ## freezing clip
        for param in self.clip.parameters():
            param.requires_grad = False

        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        vocab_size = self.clip.config.text_config.vocab_size
        hidden_size = self.clip.config.text_config.projection_dim
        visual_embedding_dim = hidden_size
        intermediate_size = 4*hidden_size

        self.infusion_block = InFusionBlockv1(emb_size=hidden_size, output_emb_size=hidden_size)

        config = VisualBertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            visual_embedding_dim=visual_embedding_dim,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings
        )

        visualbert = VisualBertModel(config)

        # self.projection_layer = nn.Linear(512, 768)

        # visualbert = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
        # # visualbert_enc = visualbert.encoder

        for param in visualbert.parameters():
            param.requires_grad = False
        
        self.visualbert_enc = visualbert.encoder
        
        self.linear = nn.Linear(visualbert.config.hidden_size, vocab_size)


    def forward(self, images, texts, labels=None):
        '''
        args:
            images: list of PIL images
            text: list of strings
        '''
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        outputs = self.clip(**inputs)

        text_emb = outputs.text_embeds
        img_emb = outputs.image_embeds

        text_seq = outputs.text_model_output.last_hidden_state
        img_seq = self.clip.visual_projection(outputs.vision_model_output.last_hidden_state)
        # print(text_seq.shape, img_seq.shape)

        fused_emb = self.infusion_block(img_emb, text_emb)

        ## having a new element in the sequence
        all_seq = torch.concat((fused_emb.unsqueeze(1), img_seq, text_seq), dim=1)

        # ## option 1
        # all_seq = self.projection_layer(all_seq)

        outputs = self.visualbert_enc(all_seq)

        last_hidden_states = outputs.last_hidden_state

        logits = self.linear(last_hidden_states)

        return logits
