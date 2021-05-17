import logging
from typing import Text, Dict

from constants.config import FINANCIAL_DATASET, TRANSFOMER_MODEL
from constants.paths import RESULT_DIR
from scripts.pipelines.training_pipeline_unsupervised import inference_without_trained_model

logging.basicConfig(level='INFO')
logger = logging.getLogger()


def main(model_name: Text,
         data_params: Dict,
         model_params: Dict,
         save_dir: Text = None):

    inference_without_trained_model(model_name,
                                    data_params,
                                    model_params,
                                    save_dir)


if __name__ == '__main__':
    model_name = TRANSFOMER_MODEL
    seed = 2021

    data_params = {'data_path': 'resources/FinancialPhraseBank/fin_data.csv',
                   'dataset_type': FINANCIAL_DATASET,
                   'preprocessed': False,
                   'vectorization': None,
                   'vector_params': None,
                   'imbalance': None,
                   'imb_params': None,
                   'train': False}

    model_params = {}

    main(model_name, data_params, model_params, RESULT_DIR)
