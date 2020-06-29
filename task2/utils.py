import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, text, label=None, seq_length=48, tokenizer=None):
        self.text = text
        self.label = label
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        if tokenizer is None:
            self.tokenizer = Tokenizer(text)
    
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):
        text = self.text[index]
        vocab_id = self.tokenizer.convert(text, self.seq_length)
        if self.label is None:
            return torch.tensor(vocab_id)
        else:
            label = self.label[index]
            return torch.tensor(vocab_id), torch.tensor(label)

# import re
class Tokenizer(object):
    def __init__(self, text=None, words=None):
        if text is not None:
            self.vocabulary_ = self.build_dict(self.tokenize(' '.join(text)), offset=2)
        if words is not None:
            self.vocabulary_ = self.build_dict(words, offset=2)
        self.vocabulary_['[PAD]'] = 0
        self.vocabulary_['[UNK]'] = 1

        
    def build_dict(self, words, offset=0):
        return {word: offset + i for i, word in enumerate(set(words))}
    
    def tokenize(self, text):
        # return list(filter(None, re.split('[\W]', text.lower())))
        return text.lower().split(' ')
    
    def convert(self, text, max_seq_length=48):
        ids = []
        tokens = self.tokenize(text)
        for token in tokens:
            ids.append(self.vocabulary_[token] if token in self.vocabulary_ \
                          else self.vocabulary_['[UNK]'])
        ids = ids[:max_seq_length]
        ids += [self.vocabulary_['[PAD]']] * (max_seq_length - len(tokens))
        return ids

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


def load_word2vec(path):
    res = dict()
    with open(path, encoding='utf8') as f:
        for line in f.readlines():
            word, vec = line.split(' ', 1)
            res[word] = list(map(float, vec.split()))
    return res
