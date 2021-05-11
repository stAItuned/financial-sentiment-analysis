from typing import Text, Dict

import torch
from torch.utils.data import Dataset


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


class NN_Dataset(Dataset):

    def __init__(self, x, y):

        self.x = torch.IntTensor(x)
        self.y = torch.IntTensor(y)

    def __getitem__(self, index):
        torch.FloatTensor()
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)






