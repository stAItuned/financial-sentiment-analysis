from constants.config import SPACY
import tensorflow_datasets as tfds
from scripts.data.preprocessing import data_preprocessing
from scripts.datasets.dataset import MyDataset
import numpy as np
import pandas as pd


class SSTDataset(MyDataset):

    def __init__(self, filepath):
        super().__init__(filepath)

    def load_data(self, filepath):
        data_df = pd.DataFrame()
        for x in ['train', 'validation', 'test']:
            data = tfds.as_numpy(tfds.load('glue/sst2', split=x, batch_size=-1))
            data_df = data_df.append(pd.DataFrame({'sentiment': data['label'],
                                                   'text': data['sentence']}, index=data['idx']))
            data_df['text'] = data_df['text'].apply(lambda x: str(x).replace('b', ''))

        self.data = data_df

    def get_x(self, data=None):
        return data['text'].to_list() if data is not None else self.data['text'].to_list()

    def get_y(self, data=None):
        return data['sentiment'].to_list() if data is not None else self.data['sentiment'].to_list()

    def training_preprocessing(self):
        prep_data = data_preprocessing(self.data,
                                       feature='text',
                                       punctuations=True,
                                       lowering=True,
                                       stemming=False,
                                       lemmatization=True,
                                       stop_words=True, )

        # Remove empty phrase
        prep_data = prep_data.drop(prep_data[prep_data['text'].str.isspace() & prep_data['text'] == ''].index)

        # Remove duplicated phrases
        prep_data = prep_data.drop(prep_data[prep_data.duplicated()])

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
        pass