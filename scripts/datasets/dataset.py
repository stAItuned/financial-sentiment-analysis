from typing import Text


class MyDataset:

    def __init__(self, filepath: Text):
        self.data = self.load_data(filepath)
        self.prep_data = None

    def load_data(self, filepath):
        pass

    def get_x(self, data=None):
        pass

    def get_y(self, data=None):
        pass

    def training_preprocessing(self):
        pass

    def test_preprocessing(self, data=None):
        pass

    def postprocessing(self, prediction, model_name):
        pass



