import torch
import torch.nn as nn

num_layers = 4
d_model = 128
num_heads = 8
dff = 512
input_input_vocab_size = 10000
maximum_position_encoding = 1000
dropout_rate = 0.1


class TransformerModel(nn.Module):
    def __init__(self, input_vocab_size, d_model, nhead, num_layers, dropout_rate):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers, dropout=dropout_rate)
        self.fc = nn.Linear(d_model, input_vocab_size)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.transformer(x, x, src_mask=mask, tgt_mask=mask)
        x = self.fc(x)
        return x

model = TransformerModel(input_vocab_size, d_model, nhead, num_layers, dropout_rate)

