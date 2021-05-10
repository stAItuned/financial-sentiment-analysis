from keras_preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils.np_utils import to_categorical
import numpy as np
from constants.config import MAX_WORD_SENTENCE
from core.preprocessing.tokenizers import CustomTokenizer
import pandas as pd


def tokenize_dataset(x, y):

    tokenizer = CustomTokenizer()
    tokenizer.fit(x, y)

    length_samples = len(x)

    x_sentence, y_sentence = [], []

    for i in range(length_samples):
        sentence = x[i]

        x_sentence.append([tokenizer.word_to_index(word) for word in sentence.split() if word != ''])

    x_dataset = pad_sequences(sequences=x_sentence, maxlen=MAX_WORD_SENTENCE,
                              padding='post', value=tokenizer.word_to_index('PAD'))

    return x_dataset, y, tokenizer
