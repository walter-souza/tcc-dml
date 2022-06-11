"""
The network architectures and weights are adapted and used from the great https://github.com/Cadene/pretrained-models.pytorch.
"""
import torch, torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig




"""============================================================="""
class Network(torch.nn.Module):
    def __init__(self, opt, use_all_tokens):
        super(Network, self).__init__()

        self.pars  = opt
        self.name = opt.arch
        self.use_all_tokens = use_all_tokens
        
        # Initializing a DistilBERT
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.configuration = self.model.config

        if self.use_all_tokens:
            self.last_linear = torch.nn.Linear(self.configuration.dim * opt.token_max_length, opt.embed_dim)
        else:
            self.last_linear = torch.nn.Linear(self.configuration.dim, opt.embed_dim)

        self.layer_blocks = self.model.transformer.layer

    # def forward(self, opt, inputs):
    #     encodings = opt.tokenizer(inputs, return_tensors='pt', truncation=True, padding='max_length')
    #     input_ids = encodings['input_ids'].to(opt.device)
    #     attention_mask = encodings['attention_mask'].to(opt.device)


    #     out = self.model(input_ids, attention_mask=attention_mask)
        
    #     # print("FORWARD")
    #     # print(out.last_hidden_state.shape)
    #     # print(self.last_linear)


    #     if self.use_all_tokens:
    #         out = out.last_hidden_state.view(out.last_hidden_state.shape[0], -1)
    #     else:
    #         out = out.last_hidden_state[:,0,:]
        
    #     # print(out.shape)
    #     out = self.last_linear(out)
    #     # print(out.shape)
    #     if 'normalize' in self.pars.arch:
    #         out = torch.nn.functional.normalize(out, dim=-1)

    #     return out

    def forward(self, input_ids, attention_mask):
        # print("FORWARD", input_ids.shape, attention_mask.shape)
        out = self.model(input_ids, attention_mask=attention_mask)
        
        # print("FORWARD")
        # print(out.last_hidden_state.shape)
        # print(self.last_linear)


        if self.use_all_tokens:
            out = out.last_hidden_state.view(out.last_hidden_state.shape[0], -1)
        else:
            out = out.last_hidden_state[:,0,:]
        
        # print(out.shape)
        out = self.last_linear(out)
        # print(out.shape)
        if 'normalize' in self.pars.arch:
            out = torch.nn.functional.normalize(out, dim=-1)

        return out
