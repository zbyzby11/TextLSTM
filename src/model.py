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

    def forward(self, x):
        # x = [batch_zise, seq_len] => input_x = [batch_zise, seq_len, embedding_size]
        input_x = self.embedding(x)
        # input_x = [batch_zise, seq_len, embedding_size]
        # output = [batch_size, seq_len, num_directions(1) * hidden_size]
        # ht,hc = [batch_size, num_layers * num_directions(1),  hidden_size]
        # ht表示在最后一个时刻（即t = seq_len的hidden的状态）
        # hc表示细胞在最后一个时刻（即t = seq_len的状态）
        output, (ht, hc) = self.lstm(input_x, None)
        # output取出最后一个时刻的hidden的状态
        # output = [batch_size, num_directions(1) * hidden_size]
        output = output[:,-1,:]
        output = self.liner(output)
        return output
