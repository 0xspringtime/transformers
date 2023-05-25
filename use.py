import torch
import torch.nn as nn
import nltk
from vocab import vocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from simpletransformer import model as loaded_model

loaded_model.load_state_dict(torch.load("trained_transformer_model.pth"))
loaded_model.eval()
loaded_model.to(device)

input_text = "Your input text here"
tokenized_input = nltk.word_tokenize(input_text)
input_indices = torch.tensor([vocab.stoi[token] for token in tokenized_input]).unsqueeze(0).to(device)

