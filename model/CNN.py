'''
Author: kk
Date: 2021-01-22 09:27:14
LastEditTime: 2021-02-04 18:19:42
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /pytorch-AAPR/model/CNN.py
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


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = INPUT_SIZE
        self.output_size = OUTPUT_SIZE
        self.hidden_size = opt.hidden_size
        self.batch_size = opt.batch_size

        self.text_embedding_layer = nn.Embedding(opt.vocab_size + 3, self.input_size).to(device)
        # conv : [input_channel(=1), output_channel(=3), (filter_height, filter_width), bias_option]
        self.cnn_layer = nn.ModuleDict({key: nn.ModuleList([nn.Conv2d(1, nf, (fs, self.input_size)).to(device) for fs in opt.filter_size]) for key, nf in opt.num_filters.items()})
        self.mp_layer = nn.ModuleDict({key: nn.ModuleList([nn.MaxPool2d((sl - fs + 1, 1)).to(device) for fs in opt.filter_size]) for key, sl in opt.seq_lens.items()})
        self.dropout = nn.Dropout(opt.dropout_rate).to(device)
        self.fc_layer1 = nn.Linear(len(opt.filter_size) * sum(opt.num_filters.values()), 8).to(device)
        self.fc_layer2 = nn.Linear(8, 2).to(device)

    def forward(self, text):
        embeds = {item: self.text_embedding_layer(text[item]) for item in opt.text_section}
        embeds = {key: embed.permute(1, 0, 2).unsqueeze(1) for key, embed in embeds.items()}
        # batch_size, channels, seq_len, voc_dim
        outer = []
        for section in opt.text_section:
            inner = []
            for cnn, mp in zip(self.cnn_layer[section], self.mp_layer[section]):
                hidden_state = F.relu(cnn(embeds[section]))
                max_pool = mp(hidden_state)
                inner.append(max_pool)
            outer.extend(inner)
        mlp_input = torch.cat(outer, dim=1).squeeze()
        mlp_hidden = self.dropout(F.relu(self.fc_layer1(mlp_input)))
        mlp_output = self.fc_layer2(mlp_hidden)
        return mlp_output
