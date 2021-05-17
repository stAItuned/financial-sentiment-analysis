from constants.config import LOG_REG, VADER, SPACY, TRANSFOMER_MODEL,  CONV_MODEL, BERT_MODEL
from scripts.models.bert_model import Bert_Model
from scripts.models.conv_model import Conv_Model
from scripts.models.logistic_regression import Logistic_Regression
from scripts.models.spacy_model import Spacy_Model
from scripts.models.transformer_model import Transformer_Model
from scripts.models.vader_model import Vader_Model


def init_model(model_type):

    if model_type == LOG_REG:
        return Logistic_Regression

    elif model_type == VADER:
        return Vader_Model

    elif model_type == SPACY:
        return Spacy_Model

    elif model_type == CONV_MODEL:
        return Conv_Model

    elif model_type == BERT_MODEL:
        return Bert_Model

    elif model_type == TRANSFOMER_MODEL:
        return Transformer_Model

    else:
        raise AttributeError(f'No valid model_type found')