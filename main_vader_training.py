import logging
from typing import Text, Dict

from constants.config import TWITTER_DATASET, VADER
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

    # plot_piechart(labels, TWITTER_DATASET).show()

    # plot_most_frequent(data, 10, TWITTER_DATASET).show()

    # plot_length_distributions(data, labels, TWITTER_DATASET).show()

if __name__ == '__main__':
    model_name = VADER
    seed = 2021

    data_params = {'data_path': 'resources/twitter_dataset/TSLA.csv',
                   'dataset_type': TWITTER_DATASET,
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
