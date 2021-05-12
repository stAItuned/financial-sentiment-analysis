import logging
from typing import Text, Dict

import torch
from torch import nn

from constants.config import SST_DATASET, \
    MAX_WORD_SENTENCE, TOKENIZER, CONV_MODEL
from constants.paths import RESULT_DIR
from scripts.datasets.dataset import NN_Dataset
from scripts.networks.conv_lstm_network import Conv1D_Network
from scripts.pipelines.nn_training_pipeline import model_training_nn

logging.basicConfig(level='INFO')
logger = logging.getLogger()


def main(model_name: Text,
         data_params: Dict,
         model_params: Dict,
         dataloader_params: Dict,
         save_dir: Text = None):

    model_training_nn(model_name, data_params, model_params, dataloader_params, save_dir)


if __name__ == '__main__':
    model_name = CONV_MODEL
    seed = 2021

    data_params = {'data_path': None,
                   'dataset_type': SST_DATASET,
                   'preprocessed': False,
                   'vectorization': TOKENIZER,
                   'imbalance': None,
                   'train': True}

    dataloader_params = {'split_size': 0.7,
                         'shuffle': True,
                         'batch_size': 64,
                         'random_seed': seed,
                         'dataset_class': NN_Dataset}

    network_params = {'emb_dim': MAX_WORD_SENTENCE,
                      'dataset_type': TOKENIZER,
                      'kernel_size': [5, 7, 9],
                      'out_channels': 10,
                      'stride': 1,
                      'padding': [0, 1, 2],
                      'pooling_kernel': 2,
                      'dropout': 0.2,
                      'device': torch.device('cpu:0')}

    training_params = {'epochs': 15,
                       'lr': 0.001,
                       'save_dir': 'resources/models/',
                       'patience': 2}

    loss = nn.BCELoss()
    optimizer = torch.optim.Adam

    model_params = {'network': network_params,
                    'loss': loss,
                    'optimizer': optimizer,
                    'training': training_params}

    main(model_name, data_params, dataloader_params, model_params, RESULT_DIR)
