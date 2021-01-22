import torch
import torch.nn as nn
from args import *

if torch.cuda.is_available() and opt.gpu >= 0:
    device = torch.device('cuda:' + str(opt.gpu))
else:
    device = torch.device('cpu')


def cr(shape):
    return torch.zeros(shape)


class Scorer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.input_size = INPUT_SIZE
        self.output_size = OUTPUT_SIZE
        self.hidden_size = opt.hidden_size
        self.batch_size = opt.batch_size
        self.num_layers = opt.num_layers
        # layer
        self.text_embedding_layer = nn.Embedding(opt.vocab_size+2, self.input_size)
        self.lstm_layer = [nn.LSTM(self.input_size, hidden_size, self.num_layers) for hidden_size in self.hidden_size] # , batch_first=True
        self.fc_layer1 = nn.Linear(sum(self.hidden_size), 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(opt.dropout_rate)
        self.fc_layer2 = nn.Linear(128, 2)

    def forward(self, text):
        embeds = [self.text_embedding_layer(text[item]) for item in opt.text_section]
        h0s = [torch.zeros(self.num_layers, self.batch_size, hidden_size) for hidden_size in self.hidden_size]
        c0s = [torch.zeros(self.num_layers, self.batch_size, hidden_size) for hidden_size in self.hidden_size]
        # hidden_states = [lstm_layer(embed, (h0, c0))[0] for lstm_layer, embed, h0, c0 in zip(self.lstm_layer, embeds, h0s, c0s)]
        hidden_states = []
        for lstm_layer, embed, h0, c0 in zip(self.lstm_layer, embeds, h0s, c0s):
            hidden_states.append(lstm_layer(embed, (h0, c0))[0])
        mlp_inputs = [hidden_state[-1, :, :] for hidden_state in hidden_states]
        mlp_input = torch.cat(mlp_inputs, dim=1)
        mlp_hidden = self.dropout(self.relu(self.fc_layer1(mlp_input)))
        mlp_output = self.fc_layer2(mlp_hidden)
        return mlp_output
