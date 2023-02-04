# Reference: https://youtu.be/EoGUlvhRYpk?list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

from torch.utils.tensorboard import SummaryWritter

import numpy as np
import spacy
import random

from model import Encoder, Decoder, Seq2Seq


spacy_ger = spacy.load('de')
spacy_eng = spacy.load('en')

def tokernizer_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokernizer_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


german = Field(tokenize=tokernizer_ger, lower=True,
               init_token='<sos>', eos_token='<eos>')

english = Field(tokenize=tokernizer_eng, lower=True,
               init_token='<sos>', eos_token='<eos>')


train_data, val_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                         fields=(german, english))

german.build_vocab(train_data, max_size=10_000, min_freq=2)
english.build_vocab(train_data, max_size=10_000, min_freq=2)


# Training hyperparameters
num_epochs = 20
learning_rate = 0.001
batch_size = 64

# Model hyperparameters
load_model = False
device = 'gpu' if torch.cuda.is_available() else 'cpu'
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 2
encoder_dropout = 0.5
decoder_dropout = 0.5

# Tensorboard
writter = SummaryWritter(f'runs/loss_plot')
step = 0

train_iterator, val_iterator, test_iterator = BucketIterator.splits(
    (train_data, val_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device
)

encoder_net = Encoder(input_size_encoder, encoder_embedding_size,
                      hidden_size, num_layers, encoder_dropout).to(device)

decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, 
                      output_size, num_layers, decoder_dropout).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)

pad_idx = english.vocab.stoi['<pad>']
loss_criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# if load_model:
#     load_checkpoint(torch.load())
