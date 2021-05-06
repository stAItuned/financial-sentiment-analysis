import logging
from typing import Text, Dict

from constants.config import FINANCIAL_DATASET, SPACY
from constants.paths import INFERENCE_DIR
from scripts.pipelines.inference_pipeline import inference_pipeline

logging.basicConfig(level='INFO')
logger = logging.getLogger()


def main(model_name: Text,
         model_path: Text,
         data_params: Dict,
         save_dir: Text = None):

    inference_pipeline(model_name, model_path, data_params, save_dir)


if __name__ == '__main__':
    model_name = SPACY
    seed = 2021

    data_params = {'data_path': 'resources/FinancialPhraseBank/fin_data.csv',
                   'dataset_type': FINANCIAL_DATASET,
                   'preprocessed': False,
                   'vectorization': None,
                   'imbalance': None,
                   'train': False}

    model_path = 'resources/output/spacy/1619801462_spacy'
    save_dir = INFERENCE_DIR

    main(model_name, model_path, data_params, save_dir)
