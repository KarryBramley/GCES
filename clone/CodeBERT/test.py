from transformers import BertModel
import torch

model = BertModel.from_pretrained("./bert-base-uncased")
# last_n_layers = 4  # 最后4层
# param_count_last_n_layers = sum(p.numel() for n, p in model.encoder.layer[-last_n_layers:].named_parameters())

# print("Parameters in the last 4 layers of bert-base-uncased:", param_count_last_n_layers)


frozen = ['embeddings.',
                  'encoder.layer.0.',
                  'encoder.layer.1.',
                  'encoder.layer.2.',
                  'encoder.layer.3.',
                  'encoder.layer.4.',
                  'encoder.layer.5.',
                  'encoder.layer.6.',
                  'encoder.layer.7.',
                 ] # *** change here to filter out params we don't want to track ***

param_optimizer = list(model.named_parameters())
param_influence = []
for n, p in param_optimizer:
    print(n)
for n, p in param_optimizer:
    if (not any(fr in n for fr in frozen)):
        param_influence.append(p)
    elif 'bert.embeddings.word_embeddings.' in n:
        pass # need gradients through embedding layer for computing saliency map
    else:
        p.requires_grad = False
        
param_size = 0
for p in param_influence:
    tmp_p = p.clone().detach()
    param_size += torch.numel(tmp_p)

print(param_size)