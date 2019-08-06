"""
数据预处理
"""
from torchtext import data
from torchtext.data import Field, Example, TabularDataset, Dataset
from torchtext.data import BucketIterator
from string import punctuation


class InputData(object):
    def __init__(self, review_file, label_file, max_length=500):
        """
        初始化输入数据类，输入两个文件，转换为两个list
        :param review_file: 评论信息，一行一条评论
        :param label_file: 标签信息，一行一个标签
        :param max_length: 一个输入的最大长度
        """
        self.stopword = [f for f in punctuation]
        self.reviews = [line.strip() for line in open(review_file)]
        self.labels = [line.strip() for line in open(label_file)]
        self.TEXT = data.Field(sequential=True, use_vocab=True, batch_first=True, stop_words=self.stopword,
                               fix_length=max_length)
        self.LABEL = data.Field(sequential=False, use_vocab=True, batch_first=True)

    def create_iter(self, split_ratio, batch_size=1000, device=-1):
        fields = [("text", self.TEXT), ("label", self.LABEL)]
        examples = []
        for review, label in zip(self.reviews, self.labels):
            item = [review, label]
            examples.append(data.Example().fromlist(item, fields))
        train, valid, test = Dataset(examples=examples, fields=fields).split(split_ratio=split_ratio)
        self.TEXT.build_vocab(train)
        self.LABEL.build_vocab(train)
        train_iter, val_iter, test_iter = data.Iterator.splits(
            (train, valid, test), sort_key=lambda x: len(x.text),
            batch_sizes=(batch_size, len(valid), len(test)), device=device)
        return train_iter, val_iter, test_iter


def main():
    data = InputData('../data/reviews.txt', '../data/labels.txt', max_length=500)
    train_iter, val_iter, test_iter = data.create_iter(split_ratio=[0.8, 0.1, 0.1], batch_size=500)
    print(len(train_iter))
    print(len(val_iter))
    print(len(test_iter))


if __name__ == '__main__':
    main()
