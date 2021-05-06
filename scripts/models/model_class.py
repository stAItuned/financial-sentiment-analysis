import logging

from core.decorators.time_decorator import timing
from core.file_manager.loadings import pickle_load

logger = logging.getLogger()


class Model:

    def __init__(self, name, params):
        self.name = name
        self.model = None

    def fit(self, x=None, y=None):
        pass

    def predict(self, x):
        pass

    @staticmethod
    def load(model_path):
        return pickle_load(model_path)