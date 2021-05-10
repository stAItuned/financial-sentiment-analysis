from scripts.models.model_class import Model
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from core.utils.sentiment_utils import getSentiment


class Vader_Model(Model):
    def __init__(self, name, params):
        super().__init__(name, params)
        self.model = SentimentIntensityAnalyzer()

    def fit(self, x=None, y=None):
        pass

    def predict(self, x):

        y = x.apply(lambda text : getSentiment(self, self.model.polarity_scores(text)['compound']))

        return y


    @staticmethod
    def load(model_path):
        pass
