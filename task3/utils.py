from torchtext import data
from torchtext.data import Iterator, BucketIterator
from torchtext.vocab import Vectors

import torch.nn as nn

def show_example(premise, hypothesis, label, TEXT, LABEL):
    print('premise:', ' '.join([TEXT.vocab.itos[i] for i in premise]))
    print('hypothesis:', ' '.join([TEXT.vocab.itos[i] for i in hypothesis]))
    print('label:', LABEL.vocab.itos[label])

def load_dataset(data_dir, word_dir, batch_size, device):
    TEXT = data.Field(batch_first=True, include_lengths=True, lower=True)
    LABEL = data.LabelField(batch_first=True)
    
    fields = {'sentence1':  ('premise', TEXT),
              'sentence2':  ('hypothesis', TEXT),
              'gold_label': ('label', LABEL)}
    
    trainDataset, valDataset, testDataset = data.TabularDataset.splits(
                                                path=data_dir,
                                                format='json',
                                                train='snli_1.0_train.jsonl',
                                                validation='snli_1.0_dev.jsonl',
                                                test='snli_1.0_test.jsonl',
                                                fields=fields,
                                                filter_pred=lambda x: x.label != '-'
                                            )
    
    vectors = Vectors('glove.6B.200d.txt', word_dir)
    
    TEXT.build_vocab(trainDataset, vectors=vectors, unk_init=nn.init.xavier_uniform)
    LABEL.build_vocab(trainDataset)
    
    train_iter, val_iter = BucketIterator.splits(
                                                 datasets=(trainDataset, valDataset),
                                                 batch_sizes=(batch_size, batch_size),
                                                 device=device,
                                                 sort_key=lambda x: len(x.premise) + len(x.hypothesis),
                                                 sort_within_batch=True,
                                                 repeat=False,
                                                 shuffle=True
                                                )
    
    test_iter = Iterator(
                         dataset=testDataset,
                         batch_size=batch_size,
                         device=device,
                         sort=False,
                         repeat=False,
                         shuffle=False
                        )
    
    return TEXT, LABEL, train_iter, val_iter, test_iter

class Average(object):
    def __init__(self):
        self.count = 0
        self.sum = 0.0
        self.ave = 0.0
    
    def update(self, val):
        self.count += 1
        self.sum += val
        self.ave = self.sum / self.count
    
    def clear(self):
        self.count = 0
        self.sum = 0.0
        self.ave = 0.0

