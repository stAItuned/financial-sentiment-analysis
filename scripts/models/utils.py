from constants.config import TRANSFOMER_MODEL
from scripts.models.transformer_model import Transformer_Model


def init_model(model_type):

    if model_type == TRANSFOMER_MODEL:
        return Transformer_Model

    else:
        raise AttributeError(f'No valid model_type found')