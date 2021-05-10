import logging

import torch
from torch import nn

from constants.config import MAX_WORD_SENTENCE, MOVIE_DATASET, TDIDF_EMBEDDING, SMOTE_IMBALANCE
from core.utils.network_summary import summary
from core.utils.range import scale_range
from scripts.datasets.dataloader import generate_dataloader
from scripts.datasets.movie_dataset import MovieDataset
from scripts.datasets.utils import dataset_generation
from scripts.models.nn_model import NetworkModel
from scripts.networks.conv_lstm_network import Conv1D_Network
from scripts.pipelines.preprocessing_pipeline import preprocessing_pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
                      'vectorization': TDIDF_EMBEDDING,
                      'vector_params': {'ngram_range': (1, 3),
                                        'max_features': 100},
                      'imbalance': None,
                      'imb_params': {'random_state': seed,
                                     'k_neighbors': 1},
                      'train': True}

    dataloader_params = {'split_size': 0.7,
                         'shuffle': True,
                         'batch_size': 32,
                         'random_seed': seed}

    network_params = {'n_words': 50,
                      'emb_dim': 100,
                      'dataset_type': TDIDF_EMBEDDING,
                      'kernel_size': 3,
                      'out_channels': 10,
                      'batch_size': 32,
                      'stride': 2,
                      'padding': 1,
                      'pooling_kernel': 2,
                      'dropout': 0.4}

    training_params = {'epochs': 10,
                       'lr': 0.01}

    x, y, _ = preprocessing_pipeline(dataset_params)
    y = scale_range(y, 0, 1)
    dataloader = generate_dataloader(x, y, dataloader_params)
    network = Conv1D_Network(network_params)
    loss = nn.BCELoss()
    optimizer = torch.optim.Adam

    model = NetworkModel(network, dataloader, loss, optimizer)
    model._init_optimizer(training_params['lr'])

    model.train(training_params['epochs'])


if __name__ == '__main__':
    # test_conv_net()

    test_network()
