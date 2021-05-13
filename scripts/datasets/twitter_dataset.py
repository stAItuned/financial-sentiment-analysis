from constants.config import SPACY
from scripts.data.extraction import extract_dataset
from scripts.data.preprocessing import data_preprocessing
from scripts.datasets.dataset import MyDataset
import numpy as np


class TwitterDataset(MyDataset):
    def __init__(self, filepath):
        super().__init__(filepath)

    def load_data(self, filepath):
        return extract_dataset(filepath)

    def get_x(self, data=None):
        return data['text'] if data is not None else self.data['text']

    def get_y(self, data=None):
        data_test = self.get_x()
        return [0 for i in range(len(data_test))]

    def training_preprocessing(self):
        pass

    def test_preprocessing(self):
        prep_data = data_preprocessing(self.data,
                                       'text',
                                       norm_contractions=False,
                                       norm_charsequences=False,
                                       twitter=True,
                                       links=True,
                                       norm_whitespaces=True,
                                       punctuations=False,
                                       lowering=False,
                                       stemming=False,
                                       lemmatization=False,
                                       stop_words=True)

        return prep_data

    def postprocessing(self, prediction, model_name):
        return prediction