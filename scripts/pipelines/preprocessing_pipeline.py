from typing import Dict
import numpy as np
from constants.config import TDIDF_EMBEDDING, SMOTE_IMBALANCE
from core.decorators.time_decorator import timing
from core.preprocessing.imbalance import smote_oversampling
from scripts.data.extraction import extract_dataset
from scripts.data.imbalance import fix_imbalance
from scripts.data.preprocessing import tdidf_preprocessing, data_preprocessing, x_to_vector
from scripts.datasets.utils import dataset_generation


@timing
def preprocessing_pipeline(params: Dict):

    preprocessed = params['preprocessed']
    vectorization_type = params.get('vectorization')
    imbalance_type = params.get('imbalance')
    train = params['train']

    dataset = dataset_generation(params)

    if train:
        prep_data = dataset.training_preprocessing() if not preprocessed else dataset.data
    else:
        prep_data = dataset.test_preprocessing() if not preprocessed else dataset.data

    x, y = dataset.get_x(prep_data), dataset.get_y(prep_data)

    x_vector = x_to_vector(x, params['vector_params'], vectorization_type) if vectorization_type is not None else x

    x, y = fix_imbalance(x_vector, y, params['imb_params'], imbalance_type) if imbalance_type is not None else (x_vector, np.array(y))

    return x, y, dataset


@timing
def preprocessing_oversampling_tdidf(params):

    data_path = params.get('data_path')
    preprocessed = params.get('preprocessed')
    embedding_type = params.get('embedding')
    imbalance = params.get('imbalance')

    data = extract_dataset(data_path)

    if not preprocessed:
        data = data_preprocessing(data)

    x, y = data['Phrase'], data['Sentiment']

    # Data to Embedding
    if embedding_type == TDIDF_EMBEDDING:
        x_emb, tdidf = tdidf_preprocessing(x,
                                           n_gram_range=params['emb_params']['ngram_range'],
                                           max_features=params['emb_params']['max_features'])
        x_emb = x_emb.toarray()
    else:
        x_emb = None

    # Imbalance Data
    if imbalance == SMOTE_IMBALANCE:
        x_smote, y_smote = smote_oversampling(x_emb,
                                              y,
                                              random_state=params['imb_params']['random_state'],
                                              k_neighbors=params['imb_params']['k_neighbors'])

        x_data, y_data = x_smote, y_smote

    else:
        x_data, y_data = x_emb, y

    return x_data, y_data
