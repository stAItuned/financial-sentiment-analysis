import json

from constants.paths import CONTRACTION_DICT_PATH


def load_contractions_dict():
    with open(CONTRACTION_DICT_PATH, 'r') as f:
        contr_dict = json.load(f)
        f.close()

    return contr_dict
