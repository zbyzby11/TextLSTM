"""
model层
"""
import torch as t
from torch import nn
from torch.optim import Adam


class TextLSTM(nn.Module):
    def __init__(self, voca_size, embed_size, hidden_size, number_layers):
        super(TextLSTM, self).__init__()
        self.voca_size = voca_size
        self.embedding_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = number_layers
        self.embedding = nn.Embedding(self.voca_size, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size,
                            self.hidden_size,
                            self.num_layers,
                            batch_first=True)
        self.liner = nn.Linear(self.hidden_size, 2)


