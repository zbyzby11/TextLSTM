"""
主要功能层
"""
import torch as t
import datetime
from torch import optim
from torch.nn import functional as F
from model import TextLSTM
from data_processing import InputData


class classification(object):
    def __init__(self,
                 embedding_size=128,
                 hidden_size=100,
                 number_layers=1,
                 max_length=500,
                 lr=0.1,
                 batch_size=1000,
                 training_times=500):
        self.device = t.device('cpu' if t.cuda.is_available() else 'cpu')
        self.max_len = max_length
        self.batch_size = batch_size
        self.input_data = InputData('../data/reviews.txt',
                                    '../data/labels.txt',
                                    max_length=self.max_len)
        self.train_iter, self.val_iter, self.test_iter, self.voca_size = self.input_data.create_iter(
            split_ratio=[0.8, 0.1, 0.1],
            batch_size=self.batch_size,
            device=self.device)
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = number_layers
        self.lr = lr
        self.training_times = training_times
        self.model = TextLSTM(self.voca_size,
                              self.embedding_size,
                              self.hidden_size,
                              self.num_layers).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self):
        for epoch in range(self.training_times):
            for index, i in enumerate(self.train_iter):
                x = i.text
                y = i.label
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                # print(output.size())
                # print(y.size())
                loss = F.cross_entropy(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print('epoch:{} || step:{} loss is:{}'.format(str(epoch), str(index), round(loss.item(), 5)))
                if index % 10 == 0:
                    val = next(iter(self.val_iter))
                    val_x = val.text
                    val_y = val.label.data.numpy()
                    val_x = val_x.to(self.device)
                    val_output = self.model(val_x)
                    # t.max:torch.return_types.max(
                    #       values=tensor([0.0162, 0.1896, 0.1317,  ..., 1.5449, 1.5449, 1.5449],
                    #                                                   grad_fn=<MaxBackward0>),
                    #       indices=tensor([0, 1, 1,  ..., 1, 1, 1]))
                    # [1]表示取indices
                    pre_y = t.max(val_output, dim=1)[1].data.numpy()
                    acc = sum(pre_y == val_y) / len(val_y)
                    print('验证集上的准确率: ', acc)


if __name__ == '__main__':
    textlstm = classification()
    textlstm.train()
