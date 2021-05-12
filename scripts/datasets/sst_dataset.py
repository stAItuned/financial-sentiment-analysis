import torch

import numpy as np
import tensorflow_datasets as tfds
from scripts.data.preprocessing import data_preprocessing
from scripts.datasets.dataset import MyDataset, NN_Dataset
from transformers import BertTokenizer
import pandas as pd

from scripts.networks.pretrained_bert import PRE_TRAINED_MODEL_NAME

MAX_LEN = 100


class SSTDataset(MyDataset):

    def __init__(self, filepath):
        super().__init__(filepath)

    def load_data(self, filepath):
        if filepath is not None:
            return pd.read_csv(filepath)

        data_df = pd.DataFrame()
        for x in ['train', 'validation', 'test']:
            data = tfds.as_numpy(tfds.load('glue/sst2', split=x, batch_size=-1))
            data_df = data_df.append(pd.DataFrame({'sentiment': data['label'],
                                                   'text': data['sentence']}, index=data['idx']))
            data_df['text'] = data_df['text'].apply(lambda x: str(x).replace('b', ''))

        return data_df

    def get_x(self, data=None):
        return data['text'].to_list() if data is not None else self.data['text'].to_list()

    def get_y(self, data=None):
        return data['sentiment'].to_list() if data is not None else self.data['sentiment'].to_list()

    def training_preprocessing(self):
        prep_data = data_preprocessing(self.data,
                                       feature='text',
                                       norm_contractions=True,
                                       norm_charsequences=True,
                                       norm_whitespaces=True,
                                       norm_punctuation=True,
                                       punctuations=True,
                                       lowering=True,
                                       lemmatization=True,
                                       stop_words=True, )

        # Remove empty phrase
        prep_data = prep_data.drop(prep_data[prep_data['text'].str.isspace() & prep_data['text'] == ''].index)

        # Remove duplicated phrases
        prep_data = prep_data.drop(prep_data[prep_data.duplicated()].index)

        self.prep_data = prep_data

        return prep_data

    def test_preprocessing(self, data=None):
        d = self.data if data is None else data
        prep_data = data_preprocessing(d,
                                       punctuations=True,
                                       lowering=True,
                                       stemming=False,
                                       lemmatization=True,
                                       stop_words=True, )

        return prep_data

    def postprocessing(self, prediction, model_name):
        return [np.round(x) for x in prediction]


class Bert_NN_Dataset(NN_Dataset):

    def __init__(self, x, y=None):
        self.x = x
        self.y = y
        self.tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    def __getitem__(self, item):
        sentence = self.x[item]
        target = self.y[item] if self.y is not None else None

        encoding = self.tokenizer.encode_plus(sentence,
                                              add_special_tokens=True,
                                              max_length=MAX_LEN,
                                              return_token_type_ids=False,
                                              pad_to_max_length=True,
                                              return_attention_mask=True,
                                              return_tensors='pt', )

        return {
            'review_text': sentence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long) if target is not None else None}

    def __len__(self):
        return len(self.x)
