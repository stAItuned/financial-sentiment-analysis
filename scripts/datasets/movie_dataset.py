from constants.config import SPACY, TRANSFOMER_MODEL
from scripts.data.extraction import extract_dataset
from scripts.data.preprocessing import data_preprocessing
from scripts.datasets.dataset import MyDataset
import numpy as np


class MovieDataset(MyDataset):

    def __init__(self, filepath):
        super().__init__(filepath)
        self.n_labels = 5

    def load_data(self, filepath):
        return extract_dataset(filepath)

    def get_x(self, data=None):
        return data['Phrase'].to_list() if data is not None else self.data['Phrase'].to_list()

    def get_y(self, data=None):
        result = data['Sentiment'] if data is not None else self.data['Sentiment']
        result = result.apply(lambda x: -1 if x == 0 or x == 1 else x)
        result = result.apply(lambda x: 0 if x == 2 else x)
        result = result.apply(lambda x: 1 if x == 3 or x == 4 else x)

        return result.to_list()

    def training_preprocessing(self):
        prep_data = data_preprocessing(self.data,
                                       feature='Phrase',
                                       punctuations=True,
                                       lowering=True,
                                       stemming=False,
                                       lemmatization=True,
                                       stop_words=True, )

        # Remove empty phrase
        prep_data = prep_data.drop(prep_data[prep_data['Phrase'].str.isspace() & prep_data['Phrase'] == ''].index)

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
        if model_name == SPACY:
            scaled_pred = (self.n_labels - 1) * ((np.array(prediction) + 1) / 2)
            return scaled_pred.round()
        elif model_name == TRANSFOMER_MODEL:
            post_pred = []
            for x in prediction:
                if x > 0.3:
                    post_pred.append(1)
                elif x < -0.3:
                    post_pred.append(-1)
                else:
                    post_pred.append(0)

            return post_pred
        else:
            return prediction
