from typing import Text, Dict

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from core.decorators.nn_decorator import training_decorator
from scripts.networks.network_class import MyNetwork
from scripts.visualization.training_plots import plot_training_step

import logging

logger = logging.getLogger()


class NetworkModel:

    def __init__(self,
                 network: MyNetwork,
                 dataloader: Dict[Text, DataLoader],
                 loss,
                 optimizer: Optimizer):

        self.network = network
        self.dataloader = dataloader
        self.loss = loss
        self.optimizer = optimizer

        self.epoch_count = []
        self.train_losses, self.valid_losses = [], []

    def _init_optimizer(self, lr):
        self.optimizer = self.optimizer(self.network.parameters(), lr=lr)

    def train(self, epochs):
        # self.epoch_count = []
        # self.train_losses, self.valid_losses = [], []

        for epoch in range(epochs):
            logger.info(f'   Epoch {epoch}/{epochs}')
            self.epoch_count.append(epoch)
            train_loss, valid_loss = self._training_step()
            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)
            plot_training_step(self.train_losses, self.valid_losses)

    @training_decorator
    def _training_step(self):
        train_loss = self._train_one_epoch()
        valid_loss = self._validate_one_epoch()

        return train_loss, valid_loss

    def _train_one_epoch(self):
        losses = []

        for x, y in self.dataloader['train']:

            out = self.network.forward(x)
            out = out.squeeze()

            loss = self.loss(out, y)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

        return np.mean(losses)

    def _validate_one_epoch(self):
        losses = []

        for x, y in self.dataloader['valid']:

            out = self.network.forward(x)
            out = out.squeeze()

            loss = self.loss(out, y)

            losses.append(loss.item())

        return np.mean(losses)

    def predict(self, x):
        pass
