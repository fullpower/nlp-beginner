import torch
import torch.nn as nn


class TextRNN(nn.Module):
    def __init__(self, seq_length, vocab_size, embedding_dim, num_classes,
                 hidden_size=128, num_layers=2, dropout=0.2):
        super(TextRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
                          batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_layers * 2 * hidden_size, num_classes)


    def forward(self, x):
        y = self.embed(x)
        output, (hn, cn) = self.rnn(y)
        y = hn.transpose(0, 1).reshape(x.size(0), -1)
        y = self.fc(self.dropout(y))
        return y

class TextCNN(nn.Module):
    def __init__(self, seq_length, vocab_size, embedding_dim, num_classes, num_filters=100, kernel_size=[3, 4, 5], keep_prob=0.5):
        super(TextCNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
                        nn.Sequential(nn.Conv1d(in_channels=embedding_dim,
                                                out_channels=num_filters,
                                                kernel_size=k),
                                      nn.ReLU(),
                                      nn.MaxPool1d(kernel_size=seq_length - k + 1))
                        for k in kernel_size
                    ])
        self.dropout = nn.Dropout(1.0 - keep_prob)
        self.fc = nn.Linear(in_features=num_filters * len(kernel_size),
                            out_features=num_classes)
        
    
    def forward(self, x):
        embed = self.embed(x).permute(0, 2, 1)
        conv_out = [conv(embed).squeeze() for conv in self.convs]
        conv_out = torch.cat(conv_out, 1)
        out = self.fc(self.dropout(conv_out))
        return out
