import nltk
from collections import Counter

blake = nltk.corpus.gutenberg.words('blake-poems.txt')

text = ' '.join(blake).lower()

tokens = nltk.word_tokenize(text)

#tokenize
vocab_counter = Counter(tokens)
vocab = sorted(vocab_counter, key=vocab_counter.get, reverse=True)
vocab_size = len(vocab)

#map each unique token to integer for a vocabulary
token_to_id = {token: idx for idx, token in enumerate(vocab)}
id_to_token = {idx: token for token, idx in token_to_id.items()}

#convert tokens to integers
token_ids = [token_to_id[token] for token in tokens]

#create input sequences by dividing token_ids into fixed length sequences
sequence_length = 50
input_sequences = []

for i in range(0, len(token_ids) - sequence_length, 1):
    seq = token_ids[i:i + sequence_length]
    input_sequences.append(seq)

#convert input sequences into input and target data
input_data = []
target_data = []

for seq in input_sequences:
    input_data.append(seq[:-1])
    target_data.append(seq[1:])

input_data = torch.tensor(input_data, dtype=torch.long)
target_data = torch.tensor(target_data, dtype=torch.long)

