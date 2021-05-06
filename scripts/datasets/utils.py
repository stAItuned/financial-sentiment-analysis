from typing import Text

from constants.config import MOVIE_DATASET, FINANCIAL_DATASET, TWITTER_DATASET
from scripts.datasets.financial_dataset import FinancialPhraseBankDataset
from scripts.datasets.movie_dataset import MovieDataset
from scripts.datasets.twitter_dataset import TwitterDataset


def init_dataset(dataset_type: Text):

    if dataset_type == MOVIE_DATASET:
        return MovieDataset
    elif dataset_type == FINANCIAL_DATASET:
        return FinancialPhraseBankDataset
    elif dataset_type == TWITTER_DATASET:
        return TwitterDataset
    else:
        raise AttributeError(f'Dataset type not found: {dataset_type}')