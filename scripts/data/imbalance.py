from typing import Dict, Text

from constants.config import SMOTE_IMBALANCE
from core.preprocessing.imbalance import smote_oversampling
import numpy as np


def fix_imbalance(x,
                  y,
                  params: Dict,
                  imbalance_type: Text = None):
    if imbalance_type == SMOTE_IMBALANCE:
        x, y = np.array(x), np.array(y)

        x_smote, y_smote = smote_oversampling(x, y,
                                              random_state=params['random_state'],
                                              k_neighbors=params['k_neighbors'])

        return x_smote, y_smote

    elif imbalance_type is None:
        return np.array(x), np.array(y)

    else:
        raise AttributeError(f'No valid imbalance type: {imbalance_type}')
