from typing import Text, Dict

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from core.decorators.nn_decorator import training_decorator
from core.file_manager.os_utils import ensure_folder
from core.utils.time_utils import timestamp
from scripts.networks.network_class import MyNetwork
from scripts.utils.early_stopping import EarlyStopping
from scripts.visualization.training_plots import plot_training_step

import logging

logger = logging.getLogger('Network Model')


class NetworkModel:

    def __init__(self,
                 network: MyNetwork,
                 dataloader: Dict[Text, DataLoader],
                 loss,
                 optimizer: Optimizer,
                 save_dir: Text):

        self.name = None
        self.network = network
        self.dataloader = dataloader
        self.loss = loss
        self.optimizer = optimizer
        self.timestamp = timestamp()
        self.save_dir = save_dir

        self.epoch_count = []
        self.train_losses, self.valid_losses = [], []

    def _init_optimizer(self, lr):
        self.optimizer = self.optimizer(self.network.parameters(), lr=lr)

    def _save_path(self, save_dir):
        save_dir = f'{save_dir}/' if save_dir[-1] != '/' else save_dir
        ensure_folder(save_dir)
        model_path = f'{save_dir}{self.name}_{self.timestamp}.pth'
        return model_path

    def train(self,
              epochs: int,
              patience: int = None,
              min_delta: float = 0):

        es = self.early_stopping(patience, min_delta)

        for epoch in range(epochs):
            logger.info(f'   Epoch {epoch+1}/{epochs}')
            self.epoch_count.append(epoch)
            train_loss, valid_loss = self._training_step()
            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)

            if len(self.valid_losses) > 1:
                stop = es.check_stopping(self.valid_losses[-1], self.valid_losses[-2]) if es is not None else False
                if not stop:
                    self.save()

        plot_training_step(self.train_losses,
                           self.valid_losses,
                           self.save_dir,
                           self.timestamp)

    @training_decorator
    def _training_step(self):
        train_loss = self._train_one_epoch()
        valid_loss = self._validate_one_epoch()

        return train_loss, valid_loss

    def _train_one_epoch(self):
        pass

    def _validate_one_epoch(self):
        pass

    @staticmethod
    def early_stopping(patience: int = None,
                       min_delta: float = 0):
        if patience is not None:
            return EarlyStopping(patience, min_delta)
        else:
            return None

    def predict(self, x):
        pass

    def save(self):
        torch.save(self, self.model_path)

    @staticmethod
    def load(model_path: Text):
        return torch.load(model_path)
