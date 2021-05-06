import logging
from typing import Text, Dict

from constants.config import MOVIE_DATASET, SPACY
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
    model_name = SPACY
    seed = 2021

    data_params = {'data_path': 'resources/preprocessed_data/cleaned_data_v1.csv',
                   'dataset_type': MOVIE_DATASET,
                   'preprocessed': True,
                   'vectorization': None,
                   'vector_params': None,
                   'imbalance': None,
                   'imb_params': None,
                   'test_size': 0.3,
                   'shuffle': True,
                   'train': True}

    model_params = {}

    main(model_name, data_params, model_params, seed, RESULT_DIR)
