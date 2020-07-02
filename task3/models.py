import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.5):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers=num_layers, dropout=dropout,
                            batch_first=True, bidirectional=True)
    
    def forward(self, x):
        self.lstm.flatten_parameters()
        output, _ = self.lstm(x)
        return output

class LocalInferenceModel(nn.Module):
    def __init__(self):
        super(LocalInferenceModel, self).__init__()
    
    def forward(self, a, mask_a, b, mask_b):
        e = torch.matmul(a, b.transpose(1, 2))
        mask_e = torch.matmul(mask_a.unsqueeze(2).float(), mask_b.unsqueeze(1).float())
        e = e.masked_fill(mask_e < 0.5, -1e5)
        
        t_a = torch.matmul(F.softmax(e, dim=2), b)
        t_b = torch.matmul(F.softmax(e, dim=1).transpose(1, 2), a)
        
        m_a = torch.cat((a, t_a, a - t_a, a * t_a), dim=-1)
        m_b = torch.cat((b, t_b, b - t_b, b * t_b), dim=-1)
        return m_a, m_b

class InferenceComposition(nn.Module):
    def __init__(self, input_size, mapping_dim, hidden_size, num_layers=1):
        super(InferenceComposition, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.F = nn.Linear(input_size, mapping_dim)
        self.relu = nn.ReLU()
        self.bilstm = BiLSTM(mapping_dim, hidden_size, num_layers=num_layers)
    
    def forward(self, x, mask):
        y = self.relu(self.F(self.dropout(x)))
        y = self.bilstm(y)
        
        mask_expand = mask.unsqueeze(-1).expand(y.shape).float()
        
        y_ = y * mask_expand
        y_avg = y_.sum(1) / mask_expand.sum(1)
        
        y_ = y.masked_fill(mask_expand == 0, -1e5)
        y_max = y_.max(1)[0]
        
        return torch.cat((y_avg, y_max), dim=-1)

    
class ESIM(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1, output_size=3):
        super(ESIM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        
        self.input_encoding = BiLSTM(embedding_dim, hidden_size, num_layers=num_layers)
        
        self.local_inference = LocalInferenceModel()
        
        self.inference_composition = InferenceComposition(hidden_size * 8, hidden_size * 2, hidden_size, num_layers=num_layers)
        
        self.mlp = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size * 8, hidden_size * 2),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size * 2, output_size)
        )
        
    def load_embedding(self, vectors):
        self.embed.weight.data.copy_(vectors)
        
    def freeze_embedding(self):
        self.embed.weight.requires_grad = False
    
    def forward(self, a, b):
        embed_a = self.embed(a) # (batch, seq_len, embedding_dim)
        embed_b = self.embed(b)
        
        encoding_a = self.input_encoding(embed_a) # (batch, seq_len, hidden_size * 2)
        encoding_b = self.input_encoding(embed_b)
        
        mask_a = (a != 1).long()
        mask_b = (b != 1).long()
        
        m_a, m_b = self.local_inference(encoding_a, mask_a, encoding_b, mask_b) # (batch, seq_len, hidden_size * 8)
        
        v_a = self.inference_composition(m_a, mask_a) # (batch, hidden_size * 4)
        v_b = self.inference_composition(m_b, mask_b)
        v = torch.cat((v_a, v_b), dim=-1) # (batch, hidden_size * 8)
        
        out = self.mlp(v)
        
        return out
