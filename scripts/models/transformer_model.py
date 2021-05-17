import logging
from typing import Iterable, List

from tqdm import tqdm

from scripts.models.model_class import Model
from transformers import pipeline

logger = logging.getLogger()


class Transformer_Model(Model):

    def __init__(self, name, params):
        super().__init__(name, params)
        self.model = pipeline('sentiment-analysis')

    def fit(self, x=None, y=None):
        pass

    def predict(self, x):

        if isinstance(x, List):
            prediction = []
            for sentence in tqdm(x, desc=' > Predict'):
                pred = self.model(sentence)[0]
                polarity = 1 if pred['label'] == 'POSITIVE' else -1 if pred['label'] == 'NEGATIVE' else 0
                sentiment = polarity * pred['score']
                prediction.append(sentiment)

        elif isinstance(x, str):
            pred = self.model(x)[0]
            polarity = 1 if pred['label'] == 'POSITIVE' else -1 if pred['label'] == 'NEGATIVE' else 0
            sentiment = polarity * pred['score']
            prediction = sentiment

        else:
            raise AttributeError('No valid input!')

        return prediction

    def postprocessing(self, x):
        post_pred = []
        for value in x:
            if value > 0.9:
                post_pred.append(1)
            elif value < -0.9:
                post_pred.append(-1)
            else:
                post_pred.append(0)

        return post_pred

    @staticmethod
    def load(model_path):
        return Transformer_Model
