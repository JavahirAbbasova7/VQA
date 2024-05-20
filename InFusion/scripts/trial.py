from models.infusion_block import InFusionBlockv1

import torch
import torch.nn as nn
# from transformers import AutoTokenizer, CLIPTextModel

from torchsummary import summary

from PIL import Image
import requests
# from transformers import AutoProcessor, CLIPVisionModel



from transformers import CLIPProcessor, CLIPModel, VisualBertModel, VisualBertConfig


clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
inputs = processor(text=["a photo of a cat", "a photo of a cat"], images=[image, image], return_tensors="pt", padding=True)

outputs = clip(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

text_emb = outputs.text_embeds
img_emb = outputs.image_embeds

text_seq = outputs.text_model_output.last_hidden_state
img_seq = clip.visual_projection(outputs.vision_model_output.last_hidden_state)




# model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
# tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

# outputs = model(**inputs)
# last_hidden_state_text = outputs.last_hidden_state
# pooled_output_text = outputs.pooler_output  # pooled (EOS token) states


# model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# inputs = processor(images=image, return_tensors="pt")

# outputs = model(**inputs)
# last_hidden_state_img = outputs.last_hidden_state
# pooled_output_img = outputs.pooler_output  # pooled CLS states

infusion_block = InFusionBlockv1(emb_size=512, output_emb_size=512)

fused_emb = infusion_block(img_emb, text_emb)


## having a new element in the sequence
all_seq = torch.concat((fused_emb.unsqueeze(1), img_seq, text_seq), dim=1)

# ### OPTION 1
# ## projection
# projection_layer = nn.Linear(512, 768)
# all_seq = projection_layer(all_seq)

# visualbert = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
# visualbert_enc = visualbert.encoder

# outputs = visualbert_enc(all_seq)

# last_hidden_states = outputs.last_hidden_state


### OPTION 2
num_hidden_layers = 2
num_attention_heads = 4

vocab_size = clip.config.text_config.vocab_size
hidden_size = clip.config.text_config.projection_dim
visual_embedding_dim = hidden_size

intermediate_size = 4*hidden_size

max_position_embeddings = 128

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
visualbert_enc = visualbert.encoder


outputs = visualbert_enc(all_seq)

last_hidden_states = outputs.last_hidden_state

summary(visualbert_enc, (58, 512))



###