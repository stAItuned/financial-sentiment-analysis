from typing import Dict

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from scripts.datasets.utils import init_dataset


def generate_dataloader(x,y,
                        params: Dict):

    batch_size = params['batch_size']
    shuffle = params['shuffle']
    split_size = params['split_size']
    seed = params['random_seed']
    dataset_type = params['dataset_type']

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        train_size=split_size,
                                                        shuffle=shuffle,
                                                        random_state=seed)

    dataset_class = init_dataset(dataset_type)

    train_dataset, valid_dataset = dataset_class(x_train, y_train), dataset_class(x_test, y_test)
    datasets = {'train': train_dataset,
                'valid': valid_dataset}

    dataloader = {x: DataLoader(dataset=datasets[x],
                                batch_size=batch_size,
                                drop_last=True,
                                shuffle=shuffle
                                ) for x in ['train', 'valid']}

    return dataloader
