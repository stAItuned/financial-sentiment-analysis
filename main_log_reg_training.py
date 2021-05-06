import logging
from typing import Text, Dict

from sklearn.model_selection import train_test_split

from constants.config import TDIDF_EMBEDDING, SMOTE_IMBALANCE, MOVIE_DATASET, LOG_REG, SPACY
from constants.paths import RESULT_DIR
from scripts.pipelines.training_pipeline import model_training

logging.basicConfig(level='INFO')
logger = logging.getLogger()


def main(model_name: Text,
         data_params: Dict,
         model_params: Dict,
         seed: int,
         save_dir: Text = None):

    model_training(model_name, data_params, model_params, seed, save_dir)


if __name__ == '__main__':
    model_name = LOG_REG
    seed = 2021

    data_params = {'data_path': 'resources/preprocessed_data/cleaned_data_v1.csv',
                   'dataset_type': MOVIE_DATASET,
                   'preprocessed': True,
                   'vectorization': TDIDF_EMBEDDING,
                   'vector_params': {'ngram_range': (1, 3),
                                     'max_features': 10000},
                   'imbalance': SMOTE_IMBALANCE,
                   'imb_params': {'random_state': seed,
                                  'k_neighbors': 5},
                   'test_size': 0.3,
                   'shuffle': True,
                   'train': True}

    model_params = {'penalty': 'l2',
                    'C': 1.0,
                    'fit_intercept': True,
                    'random_state': seed,
                    'solver': 'lbfgs',
                    'max_iter': 1000,
                    'multi_class': 'auto',
                    'verbose': 1}

    main(model_name, data_params, model_params, seed, RESULT_DIR)
