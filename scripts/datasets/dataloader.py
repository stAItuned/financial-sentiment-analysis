from typing import Dict

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from scripts.datasets.dataset import NN_Dataset, MyDataset


def generate_dataloader(dataset: MyDataset,
                        params: Dict):

    batch_size = params['batch_size']
    shuffle = params['shuffle']
    split_size = params['split_size']
    seed = params['random_seed']

    x, y = dataset.get_x(), dataset.get_y()

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        train_size=split_size,
                                                        shuffle=shuffle,
                                                        random_state=seed)

    train_dataset, valid_dataset = NN_Dataset(x_train, y_train), NN_Dataset(x_test, y_test)
    datasets = {'train': train_dataset,
                'valid': valid_dataset}

    dataloader = {x: DataLoader(dataset=datasets[x],
                                batch_size=batch_size,
                                drop_last=True,
                                shuffle=shuffle
                                ) for x in ['train', 'valid']}

    return dataloader
