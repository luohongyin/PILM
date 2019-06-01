import argparse
import time
import math
import torch
import torch.nn as nn
# from torch.autograd import Variable
import torch.optim as optim
import sys
import hashlib
# sys.path.append("../../")
# from TCN.word_cnn.utils import *
# from TCN.word_cnn.model import *
import pickle
import numpy as np
from random import randint

import tcn
import data
import model

from utils import batchify, get_batch, repackage_hidden

parser = argparse.ArgumentParser(description='Sequence Modeling - Word-level Language Modeling')

parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size (default: 16)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.45,
                    help='dropout applied to layers (default: 0.45)')
parser.add_argument('--emb_dropout', type=float, default=0.25,
                    help='dropout applied to the embedded layer (default: 0.25)')
parser.add_argument('--clip', type=float, default=0.35,
                    help='gradient clip, -1 means no clip (default: 0.35)')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--ksize', type=int, default=3,
                    help='kernel size (default: 3)')
parser.add_argument('--data', type=str, default='data/penn',
                    help='location of the data corpus (default: ./data/penn)')
parser.add_argument('--emsize', type=int, default=600,
                    help='size of word embeddings (default: 600)')
parser.add_argument('--levels', type=int, default=4,
                    help='# of levels (default: 4)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100)')
parser.add_argument('--lr', type=float, default=4,
                    help='initial learning rate (default: 4)')
parser.add_argument('--nhid', type=int, default=600,
                    help='number of hidden units per layer (default: 600)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--tied', action='store_false',
                    help='tie the encoder-decoder weights (default: True)')
parser.add_argument('--optim', type=str, default='SGD',
                    help='optimizer type (default: SGD)')
parser.add_argument('--validseqlen', type=int, default=40,
                    help='valid sequence length (default: 40)')
parser.add_argument('--seq_len', type=int, default=80,
                    help='total sequence length, including effective history (default: 80)')
parser.add_argument('--corpus', action='store_true',
                    help='force re-make the corpus (default: False)')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
args = parser.parse_args()

def process_sentence(s, corpus):
    words = s.split()# + ['<eos>']
    num_tokens = len(words)
    ids = torch.LongTensor(num_tokens)
    for i, word in enumerate(words):
        ids[i] = corpus.dictionary.word2idx[word]
    return ids.view(-1, 1).cuda()

print(args)

import os
import hashlib
fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
corpus = torch.load(fn)
# corpus = data_generator(args)
s1 = open('test_input.txt').readline()
s1 = process_sentence(s1, corpus)

print(s1)
# print(s2)

with open(args.save, 'rb') as f:
    # with open("models/candidate.pt", 'rb') as f:
    model, model_r, model_mlp, _, _ = torch.load(f)

if args.cuda:
    model.cuda()
    model_r.cuda()
    model_mlp.cuda()

model.eval()
model_r.eval()

seq_len = s1.size(0)

input_emb = model.encoder(s1)
attention, seq_len, reg_len, heights = model_r(input_emb, seq_len)

print(attention.squeeze())
print(heights.squeeze())

# model.eval()
# output2, _, _, A_outputs = model(s2)

# print(output1)
# print(output2)

# A_outputs = [A.cpu().detach().numpy()[0] for A in A_outputs]
# A_outputs.reverse()
# A_last_word = []

attention = attention.squeeze().cpu().detach().numpy()
heights = heights.squeeze().cpu().detach().numpy()

np.set_printoptions(precision=5)

print(attention)
print(heights)

# for A in A_outputs:
#     print(A)
#     print('-' * 89)
#     A_last_word.append(A[-2].tolist())

# print(A_last_word)

np.savetxt('pair_matrix.np', attention)
np.savetxt('heights.np', heights)
# np.save('A.npy', A_outputs[0], allow_pickle=False)
# np.save('A.npy', A_last_word, allow_pickle=False)
# pickle.dump(np.array(A_last_word), open('A.pkg', 'wb'))

