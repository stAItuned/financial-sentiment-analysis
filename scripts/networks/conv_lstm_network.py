import logging

import torch
from torch import nn
from torch.nn import functional as F

from constants.config import MAX_WORD_SENTENCE
from scripts.networks.network_class import MyNetwork

logger = logging.getLogger(__name__)


class Conv1D_Network(MyNetwork):
    """
    Taking inspiration from https://arxiv.org/pdf/1408.5882.pdf

    """

    def __init__(self, params):
        super().__init__(params)

        self.embedding = nn.Embedding(num_embeddings=params['n_words'],
                                      embedding_dim=params['emb_dim'],
                                      padding_idx=0)

        self.conv_layers = generate_conv_1D(params)
        self.max_pooling = nn.MaxPool1d(kernel_size=params['pooling_kernel'])

        self.dropout = nn.Dropout(params['dropout'])

        self.fc = nn.Linear(in_features=compute_fc_in_features(MAX_WORD_SENTENCE, params),
                            out_features=1)

    def forward(self, x):
        # x = x.long()[0]
        logger.debug(f' Forwarding {self.__class__.__name__}')
        logger.debug(f' \t> Input shape: {x.shape}')

        emb = self.embedding(x)
        emb = emb.permute(0, 2, 1)
        logger.debug(f' \t> Emb shape: {emb.shape}')

        dropout_emb = self.dropout(emb)
        convs = [F.relu(conv(dropout_emb)) for conv in self.conv_layers]
        logger.debug(f' \t> Conv shape: {convs[0].shape}')

        max_pools = [self.max_pooling(conv).squeeze() for conv in convs]
        logger.debug(f' \t> Pool shape: {max_pools[0].shape}')

        concat = torch.cat(max_pools, dim=1)
        logger.debug(f' \t> Concat shape: {concat.shape}')

        flatten = torch.reshape(concat, (concat.shape[0], concat.shape[1]*concat.shape[2]))
        logger.debug(f' \t> Flatten shape: {flatten.shape}')

        out = torch.sigmoid(self.fc(flatten))
        logger.debug(f' \t> Out shape: {out.shape}')

        return out


def generate_conv_1D(params):
    conv_layers = []

    kernels = params['kernel_size']
    # in_channels = params['in_channels']
    out_channels = params['out_channels']
    stride = params['stride']
    padding = params['padding']

    if isinstance(kernels, int):
        kernel = kernels
        conv_layers.append(nn.Conv1d(in_channels=params['emb_dim'],
                                     out_channels=out_channels,
                                     kernel_size=kernel,
                                     stride=stride,
                                     padding=padding))

    elif isinstance(kernels, list):
        for kernel in kernels:
            conv_layers.append(nn.Conv1d(in_channels=params['emb_dim'],
                                         out_channels=out_channels,
                                         kernel_size=kernel),
                                         stride=stride,
                                         padding=padding)

    else:
        raise TypeError(f'No valid input for kernels: {kernels}')

    return conv_layers


def compute_fc_in_features(input_length, params):

    out_channels = params['out_channels']
    kernel_size = params['kernel_size']
    pooling_kernel = params['pooling_kernel']
    stride = params['stride']

    in_features = ((input_length // stride) // pooling_kernel) * out_channels

    return in_features