from typing import Text

from constants.config import MOVIE_DATASET, FINANCIAL_DATASET
from scripts.datasets.financial_dataset import FinancialPhraseBankDataset
from scripts.datasets.movie_dataset import MovieDataset


def init_dataset(dataset_type: Text):

    if dataset_type == MOVIE_DATASET:
        return MovieDataset
    elif dataset_type == FINANCIAL_DATASET:
        return FinancialPhraseBankDataset
    else:
        raise AttributeError(f'Dataset type not found: {dataset_type}')