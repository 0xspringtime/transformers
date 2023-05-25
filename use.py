import torch
import torch.nn as nn
import nltk
from collections import Counter
from vocab import vocab, vocab_size, input_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from simpletransformer import model as loaded_model

loaded_model.load_state_dict(torch.load("trained_transformer_model.pth"))
loaded_model.eval()
loaded_model.to(device)


def preprocess_input(input_text):
    # Convert to lowercase
    input_text = input_text.lower()

    # Tokenize
    tokens = nltk.word_tokenize(input_text)

    # Convert tokens to indices using the existing vocabulary
    input_indices = [vocab.index(token) if token in vocab else vocab_size for token in tokens]

    return input_indices

input_text = "I was angry with my friend:"
input_indices = preprocess_input(input_text)
input_tensor = torch.tensor(input_indices).unsqueeze(0).to(device)

with torch.no_grad():
    output = loaded_model(input_tensor)

output_indices = torch.argmax(output, dim=-1).squeeze().cpu().numpy()
output_text = " ".join([vocab[idx] if idx < vocab_size else "<UNK>" for idx in output_indices])

print("Output text:", output_text)
