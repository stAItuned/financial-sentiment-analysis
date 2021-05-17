import logging
from typing import Text, Dict

from constants.config import YAHOO_DATASET, VADER
from constants.paths import RESULT_DIR
from scripts.pipelines.training_pipeline_unsupervised import model_training

from core.utils.plots_sentiment_analysis import plot_piechart, plot_most_frequent, plot_length_distributions

logging.basicConfig(level='INFO')
logger = logging.getLogger()


def main(model_name: Text,
         data_params: Dict,
         model_params: Dict,
         seed: int,
         save_dir: Text = None):

    _, labels, data = model_training(model_name, data_params, model_params, seed)

if __name__ == '__main__':
    model_name = VADER
    seed = 2021

    data_params = {'data_path': 'resources/yahoo_dataset/AAPL-News-cleaned.csv',
                   'dataset_type': YAHOO_DATASET,
                   'preprocessed': False,
                   'vectorization': None,
                   'vector_params': None,
                   'imbalance': None,
                   'imb_params': None,
                   'test_size': 0.99,
                   'shuffle': False,
                   'train': False}

    model_params = {}

    main(model_name, data_params, model_params, seed, RESULT_DIR)
