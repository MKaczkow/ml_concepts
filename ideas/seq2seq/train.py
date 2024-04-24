# Reference: https://youtu.be/EoGUlvhRYpk?list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from torchtext.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

from torch.utils.tensorboard import SummaryWritter

import numpy as np
import spacy
import random

from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from model import Encoder, Decoder, Seq2Seq


spacy_ger = spacy.load("de")
spacy_eng = spacy.load("en")


def tokernizer_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokernizer_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


# Create an iterator to feed taining loop
# def collate_batch(batch):
#    label_list, text_list = [], []
#    for (_label, _text) in batch:
#         label_list.append(label_transform(_label))
#         processed_text = torch.tensor(text_transform(_text))
#         text_list.append(processed_text)
#    return torch.tensor(label_list), pad_sequence(text_list, padding_value=3.0)

# train_iter = IMDB(split='train')
# train_dataloader = DataLoader(list(train_iter), batch_size=8, shuffle=True,
#                               collate_fn=collate_batch)

german = Field(
    tokenize=tokernizer_ger, lower=True, init_token="<sos>", eos_token="<eos>"
)

english = Field(
    tokenize=tokernizer_eng, lower=True, init_token="<sos>", eos_token="<eos>"
)


train_data, val_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(german, english)
)

german.build_vocab(train_data, max_size=10_000, min_freq=2)
english.build_vocab(train_data, max_size=10_000, min_freq=2)


# Training hyperparameters
num_epochs = 20
learning_rate = 0.001
batch_size = 64

# Model hyperparameters
load_model = False
device = "gpu" if torch.cuda.is_available() else "cpu"
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
writter = SummaryWritter(f"runs/loss_plot")
step = 0

train_iterator, val_iterator, test_iterator = BucketIterator.splits(
    (train_data, val_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device,
)

encoder_net = Encoder(
    input_size_encoder, encoder_embedding_size, hidden_size, num_layers, encoder_dropout
).to(device)

decoder_net = Decoder(
    input_size_decoder,
    decoder_embedding_size,
    hidden_size,
    output_size,
    num_layers,
    decoder_dropout,
).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = english.vocab.stoi["<pad>"]
loss_criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.ptar"), model, optimizer)

for epoch in range(num_epochs):
    print(f"Epoch [{epoch} / {num_epochs}]")

    checkpoint = {
        "state_dict:": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    for batch_idx, batch in enumerate(train_iterator):
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        output = model(inp_data, target)
        # output shape: (trg_len, batch_size, output_dim)

        # (N, 10) and targets would be (N)

        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        writter.add_scalar("Training loss", loss, global_step=step)
        step += 1
