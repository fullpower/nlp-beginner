import argparse
import logging

import torch
import torch.nn as nn

from models import *
from utils import *

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
logger.addHandler(console)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def eval(model, val_iter, device):
    model.eval()
    loss_func = nn.CrossEntropyLoss()
    
    total_loss = Average()
    total_acc = Average()
    
    with torch.no_grad():
        for data in val_iter:
            premise, premise_length = data.premise
            hypothesis, hypothesis_length = data.hypothesis
            label = data.label
            premise = premise.to(device)
            hypothesis = hypothesis.to(device)
            label = label.to(device)
            
            out = model(premise, hypothesis)
            
            loss = loss_func(out, label)

            total_loss.update(loss.item())
            total_acc.update(100.0 * (out.argmax(1) == label).sum().float() / len(label))
            
    logger.info('validation:')
    logger.info('loss = {}, Acc = {}%'.format(total_loss.ave, total_acc.ave))


def train(model, train_iter, val_iter, epochs, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=3e-5)
    loss_func = nn.CrossEntropyLoss()
    
    total_loss = Average()

    n_iter = 0
    
    for i in range(1, epochs + 1):
        model.train()
        
        total_loss.clear()
        
        for data in train_iter:
            premise, premise_length = data.premise
            hypothesis, hypothesis_length = data.hypothesis
            label = data.label
            premise = premise.to(device)
            hypothesis = hypothesis.to(device)
            label = label.to(device)
            
            out = model(premise, hypothesis)
            
            loss = loss_func(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss.update(loss.item())

            n_iter += 1
            if n_iter % 1000 == 0:
                logger.info('Iter {}: loss = {}'.format(n_iter, total_loss.ave))
        
        logger.info('Epoch {}: loss = {}'.format(i, total_loss.ave))
        
        if i % 5 == 0:
            eval(model, val_iter, device)
            torch.save(model.state_dict(), './esim.pth')
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', dest='log', type=str, default='esim.log')
    parser.add_argument('--epochs', dest='epochs', type=int, default=20)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='./snli_1.0')
    parser.add_argument('--word_dir', dest='word_dir', type=str, default='./.vector_cache')
    args = parser.parse_args()
    
    if args.log is not None:
        handler = logging.FileHandler(args.log)
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.info(args)
    logger.info('device = {}'.format(device))
    
    TEXT, LABEL, train_iter, val_iter, test_iter = load_dataset(args.data_dir, args.word_dir, args.batch_size, device)
    
    embedding_dim = 200
    vocab_size = len(TEXT.vocab)
    model = ESIM(vocab_size, embedding_dim, hidden_size=64, num_layers=2)
    model.load_embedding(TEXT.vocab.vectors)
    model.to(device)
    
    train(model, train_iter, val_iter, args.epochs, device)
    
    eval(model, test_iter, device)
    

if __name__ == "__main__":
    main()
