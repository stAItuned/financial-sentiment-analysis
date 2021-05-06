from tqdm import tqdm
from typing import Dict, Text
from constants.config import EMBEDDING_DIM
from constants.paths import GLOVE_PATH
import numpy as np

from core.preprocessing.tokenizers import MyTokenizer


def load_pretrained_glove_embeddings(tokenizer: MyTokenizer,
                                     embedding_path: Text =GLOVE_PATH):
    embeddings_index = {}

    f = open(embedding_path)

    for line in tqdm(f, desc='> Loading Embeddings'):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((tokenizer.n_words+1, EMBEDDING_DIM))
    for word, i in tqdm(tokenizer.vocab().items(), total=tokenizer.n_words):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


