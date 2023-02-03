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


train_data, validation_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                         fields=(german, english))

german.build_vocab(train_data, max_size=10_000, min_freq=2)
english.build_vocab(train_data, max_size=10_000, min_freq=2)


class Encoder(nn.Module):
    
    def __init__(self, input_size, embedding_size, 
                 hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):
        # x.shape: (seq_len, N)

        embedding = self.dropout(self.embedding(x))
        # embedding.shape: (seq_len, N, embedding_size)

        _, (hidden, cell) = self.rnn(embedding)
        
        return hidden, cell

class Decored(nn.Module):
    pass


class Seq2Seq(nn.Module):
    pass


