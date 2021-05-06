import logging

from tqdm import tqdm

from core.decorators.time_decorator import timing
from scripts.models.model_class import Model
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

logger = logging.getLogger()


class Spacy_Model(Model):

    def __init__(self, name, params):
        super().__init__(name, params)
        self.model = spacy.load('en_core_web_lg')
        self.model.add_pipe('spacytextblob')

    def fit(self, x=None, y=None):
        pass

    @timing
    def predict(self, x):
        pred = []

        for sentence in tqdm(x, desc='Spacy Prediction'):
            nlp_sentence = self.model(sentence)
            polarity = nlp_sentence._.polarity
            pred.append(polarity)

        return pred

    @staticmethod
    def load(model_path):
        model_class = Model.load(model_path)
        model_class.model = spacy.load('en_core_web_lg')
        model_class.model.add_pipe('spacytextblob')
        return model_class



