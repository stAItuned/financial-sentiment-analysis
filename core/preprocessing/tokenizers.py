from typing import List, Text, Dict
from keras_preprocessing.text import Tokenizer
from pandas import Series


def init_tokenizer(tokenizer_type: Text):
    if tokenizer_type == 'custom':
        return CustomTokenizer()
    elif tokenizer_type == 'keras':
        return KerasTokenizer()
    else:
        raise AttributeError(f'No valid tokenizer type: {tokenizer_type}')


class MyTokenizer:

    def __init__(self):
        self.n_words = None
        self.n_labels = None

        self._word2idx = {}
        self._label2idx = {}
        self._idx2word = {}
        self._idx2label = {}

    def fit(self,
            sentences: Series,
            targets: Series):
        pass

    def word_to_index(self,
                      word: Text):
        return self._word2idx.get(word)

    def index_to_word(self,
                      index: int):
        return self._idx2word.get(index)

    def label_to_index(self,
                       label: Text):
        return self._label2idx.get(label)

    def index_to_label(self,
                       index: int):
        return self._idx2label.get(index)

    def sentence_to_sequence(self,
                             sentence):

        return [self.word_to_index(word) for word in sentence.split() if word != '']

    def vocab(self) -> Dict:
        return self._word2idx


class CustomTokenizer(MyTokenizer):

    def __init__(self):
        super().__init__()

    def fit(self, sentences, targets):
        all_words = [word for sentence in sentences for word in sentence.split(sep=' ')]
        unique_words = set(all_words)

        unique_targets = set(targets)

        self.n_words = len(unique_words)
        self.n_labels = len(unique_targets)

        self._word2idx = {w: i + 2 for i, w in enumerate(unique_words)}
        self._word2idx['UNK'] = 1
        self._word2idx['PAD'] = 0

        self._idx2word = {i: w for w, i in self._word2idx.items()}


class KerasTokenizer(MyTokenizer):

    def __init__(self):
        super().__init__()

    def fit(self, sentences, targets):
        all_words = [word for sentence in sentences for word in sentence.split(sep=' ')]
        unique_words = set(all_words)

        all_targets = [label for target in targets for label in target.split(sep=' ')]
        unique_targets = set(all_targets)

        self.n_words = len(unique_words)
        self.n_labels = len(unique_targets)

        self._label2idx = {t: i for i, t in enumerate(unique_targets)}
        self._idx2label = {i: t for t, i in self._label2idx.items()}

        self._tokenizer = Tokenizer(self.n_words, oov_token='UNK')
        self._tokenizer.fit_on_texts(sentences)
        self._tokenizer.word_index['PAD'] = 0
        self._tokenizer.index_word[0] = 'PAD'

        self._word2idx = self._tokenizer.word_index
        self._idx2word = self._tokenizer.index_word

    def word_to_index(self,
                      word: Text):

        if word not in self._tokenizer.word_index.keys():
            index = self._tokenizer.word_index['UNK']
        else:
            index = self._tokenizer.word_index.get(word)

        return index

    def index_to_word(self,
                      index: int):
        return self._tokenizer.index_word.get(index)


