from typing import Text, Dict

from constants.config import MOVIE_DATASET, FINANCIAL_DATASET, TWITTER_DATASET, SST_DATASET
from core.decorators.time_decorator import timing
from scripts.datasets.financial_dataset import FinancialPhraseBankDataset
from scripts.datasets.movie_dataset import MovieDataset
from scripts.datasets.sst_dataset import SSTDataset
from scripts.datasets.twitter_dataset import TwitterDataset


def init_dataset(dataset_type: Text):

    if dataset_type == MOVIE_DATASET:
        return MovieDataset
    elif dataset_type == FINANCIAL_DATASET:
        return FinancialPhraseBankDataset
    elif dataset_type == TWITTER_DATASET:
        return TwitterDataset
    elif dataset_type == SST_DATASET:
        return SSTDataset
    else:
        raise AttributeError(f'Dataset type not found: {dataset_type}')


@timing
def dataset_generation(params: Dict):
    data_path = params['data_path']
    dataset_type = params['dataset_type']

    dataset_class = init_dataset(dataset_type)
    dataset = dataset_class(data_path)

    return dataset
