'''
Author: kk
Date: 2021-01-22 09:27:14
LastEditTime: 2021-01-30 13:19:41
LastEditors: Please set LastEditors
Description: kk
FilePath: /pytorch-AAPR/model/GRU.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from args import *

if torch.cuda.is_available() and opt.gpu >= 0:
    device = torch.device('cuda:' + str(opt.gpu))
else:
    device = torch.device('cpu')


def cr(shape):
    return torch.zeros(shape)


class GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = INPUT_SIZE
        self.output_size = OUTPUT_SIZE
        self.hidden_size = opt.hidden_size
        self.batch_size = opt.batch_size
        self.num_layers = opt.num_layers
        # layer
        self.text_embedding_layer = nn.Embedding(opt.vocab_size + 3, self.input_size).to(device)
        self.gru_layer = nn.ModuleDict({key: nn.GRU(self.input_size, hidden_size, self.num_layers).to(device) for key, hidden_size in self.hidden_size.items()})
        # self.gru_layer = [nn.GRU(self.input_size, hidden_size, self.num_layers, dropout=opt.dropout).to(device) for hidden_size in self.hidden_size] # , batch_first=True
        self.dropout = nn.Dropout(opt.dropout_rate).to(device)
        self.fc_layer1 = nn.Linear(sum(self.hidden_size.values()), 128).to(device)
        self.fc_layer2 = nn.Linear(128, 2).to(device)

    def forward(self, text):
        embeds = {item: self.text_embedding_layer(text[item]) for item in opt.text_section}
        # embeds = [self.text_embedding_layer(text[item]) for item in opt.text_section]
        h0s = {key: torch.zeros(self.num_layers, self.batch_size, hidden_size).to(device) for key, hidden_size in self.hidden_size.items()}
        # h0s = [torch.zeros(self.num_layers, self.batch_size, hidden_size).to(device) for hidden_size in self.hidden_size]
        hidden_states = {}
        for section in opt.text_section:
            hidden_states[section] = self.gru_layer[section](embeds[section], h0s[section])[0]
        mlp_inputs = [hidden_state[-1, :, :] for hidden_state in hidden_states.values()]
        mlp_input = torch.cat(mlp_inputs, dim=1)
        mlp_hidden = self.dropout(F.relu(self.fc_layer1(mlp_input)))
        mlp_output = self.fc_layer2(mlp_hidden)
        return mlp_output
