import logging

import torch
from torch import nn

from constants.config import MAX_WORD_SENTENCE, MOVIE_DATASET, TOKENIZER, SMOTE_IMBALANCE
from core.utils.network_summary import summary
from scripts.datasets.dataloader import generate_dataloader
from scripts.datasets.movie_dataset import MovieDataset
from scripts.datasets.utils import dataset_generation
from scripts.models.nn_model import NetworkModel
from scripts.networks.conv_lstm_network import Conv1D_Network
from scripts.pipelines.preprocessing_pipeline import preprocessing_pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def test_conv_net():
    params = {'n_words': 50,
              'emb_dim': 100,
              'kernel_size': 3,
              'out_channels': 10,
              'stride': 2,
              'padding': 1,
              'pooling_kernel': 2,
              'dropout': 0.4}

    model = Conv1D_Network(params)

    summary(model, (10, MAX_WORD_SENTENCE,), device=torch.device('cpu:0'))


def test_network():
    seed = 2021

    dataset_params = {'data_path': 'resources/preprocessed_data/cleaned_data_v1.csv',
                      'dataset_type': MOVIE_DATASET,
                      'preprocessed': True,
                      'vectorization': TOKENIZER,
                      'vector_params': {'ngram_range': (1, 3),
                                        'max_features': 100},
                      'imbalance': SMOTE_IMBALANCE,
                      'imb_params': {'random_state': seed,
                                     'k_neighbors': 5},
                      'train': True}

    dataloader_params = {'split_size': 0.7,
                         'shuffle': True,
                         'batch_size': 32,
                         'random_seed': seed}

    network_params = {'n_words': 50,
                      'emb_dim': 100,
                      'kernel_size': 3,
                      'out_channels': 10,
                      'stride': 2,
                      'padding': 1,
                      'pooling_kernel': 2,
                      'dropout': 0.4}

    dataset = dataset_generation(dataset_params)
    dataloader = generate_dataloader(dataset, dataloader_params)
    network = Conv1D_Network(network_params)
    loss = nn.BCELoss()
    optimizer = torch.optim.Adam

    model = NetworkModel(network, dataloader, loss, optimizer)

    model.train()


if __name__ == '__main__':
    # test_conv_net()

    test_network()
