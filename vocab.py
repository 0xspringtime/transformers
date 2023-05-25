import nltk
from collections import Counter

blake = nltk.corpus.gutenberg.words('blake-poems.txt')

text = ' '.join(blake).lower()

tokens = nltk.word_tokenize(text)

#tokenize
vocab_counter = Counter(tokens)
vocab = sorted(vocab_counter, key=vocab_counter.get, reverse=True)
vocab_size = len(vocab)
