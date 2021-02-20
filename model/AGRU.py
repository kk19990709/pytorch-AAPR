'''
Author: your name
Date: 2021-01-26 14:41:24
LastEditTime: 2021-02-04 20:59:46
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /pytorch-AAPR/model/AGRU.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from args import *

if torch.cuda.is_available() and opt.gpu >= 0:
    device = torch.device('cuda:' + str(opt.gpu))
else:
    device = torch.device('cpu')


class GRU(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.vocab_dims = INPUT_SIZE
        self.hidden_size = hidden_size
        self.batch_size = opt.batch_size
        self.num_layers = opt.num_layers
        self.gru = nn.GRU(self.vocab_dims, self.hidden_size, self.num_layers)

    def forward(self, input):
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)
        output, hidden = self.gru(input, h0)
        return output, hidden


class Attn(nn.Module):
    def __init__(self, seq_len, hidden_size):
        super().__init__()
        self.vocab_dims = INPUT_SIZE
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.batch_size = opt.batch_size
        self.num_layers = opt.num_layers

        self.attn = nn.Linear(self.hidden_size + self.vocab_dims, self.seq_len)
        self.attn_comb = nn.Linear(self.hidden_size + self.vocab_dims, self.hidden_size)

    def forward(self, input, encoder_outputs, hidden):
        input = input.view(1, opt.batch_size, -1)

        embed_with_hidden = torch.cat((input[0], hidden[0]), dim=1)  # [batch_size, +]
        attn_weights = F.softmax(self.attn(embed_with_hidden))
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs.transpose(0, 1)).transpose(0, 1)

        output = torch.cat((input, attn_applied), dim=2)
        output = F.relu(self.attn_comb(output))

        return output


class AGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_dims = INPUT_SIZE
        self.seq_lens = opt.seq_lens
        self.hidden_size = opt.hidden_size
        self.batch_size = opt.batch_size
        self.num_layers = opt.num_layers
        self.text_embedding_layer = nn.Embedding(opt.vocab_size + 3, self.vocab_dims).to(device)
        self.gru = nn.ModuleDict({key: GRU(hidden_size).to(device) for key, hidden_size in self.hidden_size.items()})
        self.attn = nn.ModuleDict({key: Attn(seq_len, hidden_size).to(device) for (key, seq_len), hidden_size in zip(self.seq_lens.items(), self.hidden_size.values())})
        self.hidden = nn.Linear(sum(opt.hidden_size.values()), 128)
        self.out = nn.Linear(128, 2)

    def forward(self, inputs, attn_input):
        gru_inputs = dict()
        for input_key, input_value in inputs.items():
            gru_inputs[input_key] = self.text_embedding_layer(input_value)

        gru_output_hiddens = dict()

        # for gru_input_key, gru_input_value in gru_inputs.items():
        #     gru_output_hiddens[gru_input_key] = self.gru[gru_input_key](gru_input_value)
        for key in opt.text_section:
            gru_output_hiddens[key] = self.gru[key](gru_inputs[key])

        attn_input = self.text_embedding_layer(attn_input)
        attn_outputs = dict()
        for gru_output_hidden_key, (gru_output_value, gru_hidden_value) in gru_output_hiddens.items():
            attn_outputs[gru_output_hidden_key] = self.attn[gru_output_hidden_key](attn_input, gru_output_value, gru_hidden_value)

        attn_applied = torch.cat([val[0] for val in attn_outputs.values()], dim=1)
        hidden = self.hidden(attn_applied)
        out = self.out(hidden)
        return out
