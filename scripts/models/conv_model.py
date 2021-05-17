import logging
import numpy as np
from scripts.models.model_class import Model
from scripts.models.nn_conv_model import ConvModel
from scripts.networks.conv_lstm_network import Conv1D_Network

logger = logging.getLogger()


class Conv_Model(Model):

    def __init__(self, name, params):
        self.training_params = params['training']
        self.network_params = params['network']

        self.model = ConvModel(Conv1D_Network(self.network_params).to(self.network_params['device']),
                               params['dataloader'],
                               params['loss'],
                               params['optimizer'],
                               self.training_params['save_dir'],
                               self.network_params['device'])
        self.model._init_optimizer(self.training_params['lr'])
        self.name = self.model.name

    def fit(self, x=None, y=None):
        epochs = self.training_params['epochs']
        patience = self.training_params.get('patience')
        min_delta = self.training_params.get('min_delta')
        self.model.train(epochs, patience, min_delta)

    def predict(self, x):
        prediction = self.model.predict(x[0])
        result = np.argmax(prediction, axis=1)
        return result

    @staticmethod
    def load(model_path):
        return ConvModel.load(model_path)
