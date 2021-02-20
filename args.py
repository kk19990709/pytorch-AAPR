'''
Author: your name
Date: 2021-01-22 09:27:13
LastEditTime: 2021-02-05 11:22:36
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /pytorch-AAPR/args.py
'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--nt', dest='notrain', action='store_true', default=False)
parser.add_argument('--md', dest='makedata', action="store_true", default=False)
parser.add_argument('--mn', dest='minidata', action="store_true", default=False)
parser.add_argument('--lg', dest='largedata', action="store_true", default=False)
parser.add_argument('--ds', dest='datasize', type=int, default=-1)
parser.add_argument('--bs', dest='batch_size', type=int, default=32)
parser.add_argument('--hs', dest='hidden_size', nargs='+', type=int, default={'abstr': 64, 'title': 32, 'intro': 64, 'relat': 64, 'metho': 64, 'concl': 64})
# parser.add_argument('--nf', dest='num_filters', nargs='+', type=int, default={'abstr': 5, 'title': 5, 'intro': 5, 'relat': 5, 'metho': 5, 'concl': 5})
# parser.add_argument('--fs', dest='filter_size', nargs='+', type=int, default=[5, 5, 5])
parser.add_argument('--sl', dest='seq_lens', nargs='+', type=int, default={'abstr': 160, 'title': 16, 'intro': 512, 'relat': 300, 'metho': 800, 'concl': 240})
parser.add_argument('--ts', dest='text_section', nargs='+', type=str, default=['abstr', 'title', 'intro', 'relat', 'metho', 'concl'])
parser.add_argument('--ne', dest='num_epochs', type=int, default=100)
parser.add_argument('--nl', dest='num_layers', type=int, default=2)
parser.add_argument('--lr', dest='learning_rate', type=float, default=0.004)
parser.add_argument('--dr', dest='dropout_rate', type=float, default=0.1)
parser.add_argument('--vs', dest='vocab_size', type=int, default=40000)
parser.add_argument('--gpu', dest='gpu', type=int, default=2)
parser.add_argument('--wghdp', dest='weight_datapath', type=str, default='./weight/')
parser.add_argument('--model', dest='model', type=str, default='LSTM')
parser.add_argument('--optim', dest='optimizer', type=str, default='AdamW')
# opt = parser.parse_args()
opt = parser.parse_args(args=["--ne", "20", "--bs", "48", "--model", "AGRU", "--optim", "Adam"])
# opt = parser.parse_args(args=["--ne", "3", "--bs", "48", "--model", "SharedAGRU", "--optim", "Adam"])
# opt = parser.parse_args(args=["--mn", "--ne", "300", "--model", "SharedAGRU", "--optim", "Adam"])
print(vars(opt))

INPUT_SIZE = 300
OUTPUT_SIZE = 2
