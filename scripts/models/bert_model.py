import logging

from scripts.models.model_class import Model
from scripts.models.nn_pretrained_model import Pretrained_Bert_Model

logger = logging.getLogger()


class Bert_Model(Model):

    def __init__(self, name, params):
        self.model = Pretrained_Bert_Model(params['network'],
                                           params['dataloader'],
                                           params['loss'],
                                           params['optimizer'],
                                           params['save_dir'],
                                           params['device'])
        self.name = self.model.name
        self.training_params = params['training_params']

    def fit(self, x=None, y=None):
        epochs = self.training_params['epochs']
        patience = self.training_params.get('patience')
        min_delta = self.training_params.get('min_delta')
        self.model.train(epochs, patience, min_delta)

    def predict(self, x):
        out, target = self.model.predict(x)

        return out.numpy(), target.numpy()

    @staticmethod
    def load(model_path):
        return Pretrained_Bert_Model.load(model_path)