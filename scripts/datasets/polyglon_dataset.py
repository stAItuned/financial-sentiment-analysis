from typing import Text

from core.utils.polyglon_news import get_news
from scripts.data.extraction import extract_dataset
from scripts.data.preprocessing import data_preprocessing
from scripts.datasets.dataset import MyDataset
import numpy as np

from datetime import datetime

import os


def load_polyglon_data(ticker, window=14):
    date = str(datetime.today()).split(" ")[0]
    path = f'news/{ticker}-{date}_{window}.csv'

    # if the dataset is not already present, scrape it
    if not os.path.exists(path):
        get_news(ticker, window)

    return extract_dataset(path)


class PolyglonDataset(MyDataset):
    def __init__(self, filepath, ticker):
        self.ticker = ticker
        super().__init__(filepath)


    def load_data(self, filepath):
        return load_polyglon_data(self.ticker)

    def get_x(self, data=None):
        print(data)
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
                                       twitter=False,
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