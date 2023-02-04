import random
import spacy
import torch
import torch.nn as nn
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator


device = 'gpu' if torch.cuda.is_available() else 'cpu'

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

class Decoder(nn.Module):
    
    def __init__(self, input_size, embedding_size,
                 hidden_size, output_size, num_layers, p):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden, cell):
        # x.shape: (N)
        x = x.unsqueeze(0)

        # x.shape: (1, N)
        embedding = self.dropout(self.embedding(x))

        # embedding.shape: (1, N, embedding_size)
        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))

        # outputs.shape: (1, N, hidden_size)
        predictions = self.fc(outputs)

        # predictions.shape: (1, N, vocab_len)
        predictions = predictions.squeeze()

        return predictions, hidden, cell
    

class Seq2Seq(nn.Module):
    
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        hidden, cell = self.encoder(source)

        # Get start token
        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output

            # output.shape: (N, english_vocab_size)
            best_guess = output.argmax(1)

            x = target[t] if random.random() < teacher_force_ratio else best_guess
        
        return outputs