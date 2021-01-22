import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--nt', dest='notrain', action='store_true', default=False)
parser.add_argument('--md', dest='makedata', action="store_true", default=False)
parser.add_argument('--bs', dest='batch_size', type=int, default=32)
parser.add_argument('--hs', dest='hidden_size', nargs='+', type=int, default=[64, 32, 64, 64, 64, 64])
parser.add_argument('--ts', dest='text_section', nargs='+', type=str, default=['abstr', 'title', 'intro', 'relat', 'metho', 'concl'])
parser.add_argument('--ne', dest='num_epochs', type=int, default=100)
parser.add_argument('--nl', dest='num_layers', type=int, default=2)
# parser.add_argument('--sl', dest='seq_len', type=int, default=200)
parser.add_argument('--lr', dest='learning_rate', type=float, default=0.01)
parser.add_argument('--dr', dest='dropout_rate', type=float, default=0.1)
parser.add_argument('--vs', dest='vocab_size', type=int, default=40000)
parser.add_argument('--gpu', dest='gpu', type=int, default=-1)
# parser.add_argument('--loss', dest='loss', type=str, default='Focal')
# parser.add_argument('--logdp', dest='log_datapath', type=str, default='./log/')
parser.add_argument('--wghdp', dest='weight_datapath', type=str, default='./weight/')
opt = parser.parse_args(args=["--nt", "--md"])
# opt = parser.parse_args(args=["--hs", "32", "64", "64", "64", "64", "64"])
# opt = parser.parse_args()
print(vars(opt))

INPUT_SIZE = 300
OUTPUT_SIZE = 1