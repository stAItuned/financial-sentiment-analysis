from constants.config import LOG_REG, VADER, SPACY
from scripts.models.logistic_regression import Logistic_Regression
from scripts.models.spacy_model import Spacy_Model


def init_model(model_type):

    if model_type == LOG_REG:
        return Logistic_Regression

    elif model_type == VADER:
        pass

    elif model_type == SPACY:
        return Spacy_Model

    else:
        raise AttributeError(f'No valid model_type found')