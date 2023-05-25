import torch
import torch.nn as nn

vocab_size = 10000
d_model = 512
nhead = 8
num_layers = 6
dropout_rate = 0.1


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dropout_rate):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers, dropout=dropout_rate)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.transformer(x, x, src_mask=mask, tgt_mask=mask)
        x = self.fc(x)
        return x

model = TransformerModel(vocab_size, d_model, nhead, num_layers, dropout_rate)

