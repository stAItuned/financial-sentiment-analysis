import unittest
import logging
import torch
from torch import nn

from constants.config import MOVIE_DATASET, TOKENIZER, MAX_WORD_SENTENCE, SST_DATASET
from scripts.datasets.dataloader import generate_dataloader
from scripts.datasets.sst_dataset import Bert_NN_Dataset
from scripts.models.conv_model import ConvModel
from scripts.models.nn_model import NetworkModel
from scripts.models.pretrained_model import Pretrained_Bert_Model
from scripts.networks.conv_lstm_network import Conv1D_Network
from scripts.networks.pretrained_bert import Pretrained_Bert_Network
from scripts.pipelines.preprocessing_pipeline import preprocessing_pipeline
from transformers import AdamW


class NetworkTest(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='\n%(asctime)s %(module)s %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    def test_conv_net(self):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        seed = 2021

        emb_dim = 100

        dataset_params = {'data_path': 'resources/preprocessed_data/cleaned_data_v1.csv',
                          'dataset_type': MOVIE_DATASET,
                          'preprocessed': True,
                          # 'target_scaling': (0, 1),
                          'vectorization': TOKENIZER,
                          # 'vector_params': {'ngram_range': (1, 3),
                          #                   'max_features': emb_dim},
                          'imbalance': None,
                          # 'imb_params': {'random_state': seed,
                          #                'k_neighbors': 3},
                          'train': True}

        dataloader_params = {'split_size': 0.7,
                             'shuffle': True,
                             'batch_size': 32,
                             'random_seed': seed}

        network_params = {'emb_dim': emb_dim,
                          'dataset_type': TOKENIZER,
                          'kernel_size': [3, 5, 7],
                          'out_channels': 10,
                          'stride': 1,
                          'padding': [0, 1, 2],
                          'pooling_kernel': 2,
                          'dropout': 0.4}

        training_params = {'epochs': 1,
                           'lr': 0.001}

        x, y, dataset, vectorizer = preprocessing_pipeline(dataset_params)
        network_params['n_words'] = vectorizer.get_n_words()

        dataloader = generate_dataloader(x, y, dataloader_params)
        network = Conv1D_Network(network_params)
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam

        model = NetworkModel(network, dataloader, loss, optimizer)
        model._init_optimizer(training_params['lr'])

        model.train(training_params['epochs'])

    def test_train_model(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        seed = 2021

        emb_dim = 100

        dataset_params = {'data_path': 'resources/preprocessed_data/cleaned_data_v1.csv',
                          'dataset_type': MOVIE_DATASET,
                          'preprocessed': True,
                          # 'target_scaling': (0, 1),
                          'vectorization': TOKENIZER,
                          # 'vector_params': {'ngram_range': (1, 3),
                          #                   'max_features': emb_dim},
                          'imbalance': None,
                          # 'imb_params': {'random_state': seed,
                          #                'k_neighbors': 3},
                          'train': True}

        dataloader_params = {'split_size': 0.7,
                             'shuffle': True,
                             'batch_size': 64,
                             'random_seed': seed}

        network_params = {'emb_dim': MAX_WORD_SENTENCE,
                          'dataset_type': TOKENIZER,
                          'kernel_size': [5, 7, 9],
                          'out_channels': 10,
                          'stride': 1,
                          'padding': [0, 1, 2],
                          'pooling_kernel': 2,
                          'dropout': 0.2}

        training_params = {'epochs': 10,
                           'lr': 0.001,
                           'save_dir': 'resources/models/',
                           'patience': 5}

        x, y, dataset, vectorizer = preprocessing_pipeline(dataset_params)
        network_params['n_words'] = vectorizer.get_n_words()

        dataloader = generate_dataloader(x, y, dataloader_params)
        network = Conv1D_Network(network_params)
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam

        model = ConvModel(network, dataloader, loss, optimizer, training_params['save_dir'])
        model._init_optimizer(training_params['lr'])

        model.train(training_params['epochs'], patience=training_params['patience'])


    def test_bert_model(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        seed = 2021

        dataset_params = {'data_path': None,
                          'dataset_type': SST_DATASET,
                          'preprocessed': False,
                          'vectorization': None,
                          'imbalance': None,
                          'train': True}

        dataloader_params = {'split_size': 0.7,
                             'shuffle': True,
                             'batch_size': 64,
                             'random_seed': seed}

        network_params = {'dropout': 0.2}

        training_params = {'epochs': 10,
                           'lr': 0.001,
                           'save_dir': 'resources/models/',
                           'patience': 5}

        x, y, dataset, vectorizer = preprocessing_pipeline(dataset_params)

        dataloader = generate_dataloader(x, y, dataloader_params, Bert_NN_Dataset)
        network = Pretrained_Bert_Network(network_params)
        loss = nn.BCELoss()
        optimizer = AdamW

        model = Pretrained_Bert_Model(network, dataloader, loss, optimizer, training_params['save_dir'])
        model._init_optimizer(training_params['lr'])

        model.train(training_params['epochs'], patience=training_params['patience'])


if __name__ == '__main__':
    unittest.main()
