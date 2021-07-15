from typing import Text, Dict

from constants.config import POLYGLON_DATASET, TWITTER_DATASET, SST_DATASET, YAHOO_DATASET, NN_DATASET
from core.decorators.time_decorator import timing
from scripts.datasets.dataset import NN_Dataset
from scripts.datasets.sst_dataset import SSTDataset
from scripts.datasets.twitter_dataset import TwitterDataset
from scripts.datasets.yahoo_dataset import YahooDataset
from scripts.datasets.polyglon_dataset import PolyglonDataset


def init_dataset(dataset_type: Text):

    if dataset_type == TWITTER_DATASET:
        return TwitterDataset
    elif dataset_type == SST_DATASET:
        return SSTDataset
    elif dataset_type == YAHOO_DATASET:
        return YahooDataset
    elif dataset_type == NN_DATASET:
        return NN_Dataset
    elif dataset_type == POLYGLON_DATASET:
        return PolyglonDataset
    else:
        raise AttributeError(f'Dataset type not found: {dataset_type}')


@timing
def dataset_generation(params: Dict):
    data_path = params['data_path']
    dataset_type = params['dataset_type']
    ticker = params['ticker']

    dataset_class = init_dataset(dataset_type)

    if dataset_type == POLYGLON_DATASET:
        dataset = dataset_class(data_path, ticker)
    else:
        dataset = dataset_class(data_path)

    return dataset
