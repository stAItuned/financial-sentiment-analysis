import logging

from sklearn.linear_model import LogisticRegression

from scripts.models.model_class import Model

logger = logging.getLogger()


class Logistic_Regression(Model):

    def __init__(self, name, params):

        super().__init__(name, params)
        self.log_reg = LogisticRegression(penalty=params['penalty'],
                                          C=params['C'],
                                          fit_intercept=params['fit_intercept'],
                                          random_state=params['random_state'],
                                          solver=params['solver'],
                                          max_iter=params['max_iter'],
                                          multi_class=params['multi_class'],
                                          verbose=params['verbose'])

    def fit(self, x=None, y=None):
        self.log_reg.fit(x, y)

    def predict(self, x):
        return self.log_reg.predict(x)

    @staticmethod
    def load(model_path):
        model_class = Model.load(model_path)
        return model_class




