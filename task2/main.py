import argparse
import logging

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import *
from utils import MyDataset, Tokenizer, Average


logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
logger.addHandler(console)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


embedding_dim = 200
seq_length = 48
num_classes = 5


def eval(net, val_dl, device):
    net.eval()
    loss_func = nn.CrossEntropyLoss()
    
    total_loss = Average()
    total_acc = Average()
    
    with torch.no_grad():
        for data in val_dl:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            
            out = net(x)
            loss = loss_func(out, y)
            total_loss.update(loss.item())
            total_acc.update(100.0 * (out.argmax(1) == y).sum() / len(y))
           
            
    logger.info('validation:')
    logger.info('loss = {}, Acc = {}%'.format(total_loss.ave, total_acc.ave))

def train(net, train_dl, val_dl, num_epochs, device):

    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=3e-4)
    loss_func = nn.CrossEntropyLoss()
    
    total_loss = Average()
    
    for i in range(1, num_epochs + 1):
        net.train()
        
        total_loss.clear()
        
        for data in train_dl:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            
            out = net(x)
            loss = loss_func(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss.update(loss.item())
            
        
        logger.info('Epoch {}: loss = {}'.format(i, total_loss.ave))
        
        if i % 5 == 0:
            eval(net, val_dl, device)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', dest='epochs', type=int, default=10)
    parser.add_argument('--output', dest='output', type=str, default='./nn_result.csv')
    parser.add_argument('--log', dest='log', type=str)
    parser.add_argument('--w2v', dest='w2v', type=str)
    parser.add_argument('--freeze', dest='freeze', action='store_true', default=False)
    parser.add_argument('--model', dest='model', type=str, choices=['CNN', 'RNN'], default='CNN')
    args = parser.parse_args()
    
    if args.log is not None:
        handler = logging.FileHandler(args.log)
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.info(args)
    logger.info('device = {}'.format(device))

    tsv_train = pd.read_csv('./data/train.tsv', sep='\t')
    num = len(tsv_train)
    num_val = int(num * 0.1)
    num_train = num - num_val
    
    tsv_val = tsv_train[num_train:]
    tsv_train = tsv_train[:num_train]
    
    x_train = tsv_train['Phrase'].values
    y_train = tsv_train['Sentiment'].values
    x_val = tsv_val['Phrase'].values
    y_val = tsv_val['Sentiment'].values

    if args.w2v is None:
        tokenizer = Tokenizer(text=x_train)
    else:
        from utils import load_word2vec
        w2v = load_word2vec(args.w2v)
        words = list(w2v.keys())
        tokenizer = Tokenizer(words=words)
        vecs = list(w2v.values())
        vecs.insert(0, [.0] * embedding_dim)
        vecs.insert(1, [.0] * embedding_dim)
    
    vocab_size = len(tokenizer.vocabulary_)
    logger.info('vocab_size = {}'.format(vocab_size))
    
    train_dataset = MyDataset(x_train, y_train, seq_length, tokenizer)
    train_dl = DataLoader(train_dataset, shuffle=True, batch_size=128)
    
    val_dataset = MyDataset(x_val, y_val, seq_length, tokenizer)
    val_dl = DataLoader(val_dataset, shuffle=False, batch_size=128)
    
    net = TextCNN(seq_length, vocab_size, embedding_dim, num_classes) if args.model == 'CNN' \
        else TextRNN(seq_length, vocab_size, embedding_dim, num_classes)

    if args.w2v is not None:
        net.embed.from_pretrained(torch.tensor(vecs), freeze=args.freeze)
    net.to(device)
    
    train(net, train_dl, val_dl, args.epochs, device)
    
    tsv_test = pd.read_csv('./data/test.tsv', sep='\t')
    x_test = tsv_test['Phrase'].values
    test_dataset = MyDataset(x_test, seq_length=seq_length, tokenizer=tokenizer)
    test_dl = DataLoader(test_dataset, shuffle=False, batch_size=128)
    y_test = []
    for data in test_dl:
        x = data.to(device)
        out = net(x)
        y_test += out.argmax(1).cpu().tolist()
        
    tsv_test['Sentiment'] = y_test
    tsv_test[['PhraseId', 'Sentiment']].to_csv(args.output, index=False)

if __name__ == "__main__":
    main()
