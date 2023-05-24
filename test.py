import numpy as np
import torch
import torch.nn as nn

#hyperparameters

num_layers = 4
d_model = 128
num_heads = 8
dff = 512
input_vocab_size = 10000
maximum_position_encoding = 1000
dropout_rate = 0.1


