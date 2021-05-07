from torch.utils.data import DataLoader

from scripts.datasets.dataloader import generate_dataloader
from scripts.datasets.dataset import MyDataset, NN_Dataset
from scripts.networks.network_class import MyNetwork


class NetworkModel:

    def __init__(self,
                 network: MyNetwork,
                 dataloader: DataLoader,
                 loss,
                 optimizer):

        self.network = network
        self.dataloader = dataloader
        self.loss = loss
        self.optimizer = optimizer

    def train(self):
        pass

    def _train_one_epoch(self):
        pass

    def _validate_one_epoch(self):
        pass

    def predict(self, x):
        pass
