from typing import Text, Tuple, Dict
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from constants.config import TDIDF_EMBEDDING, TOKENIZER
from core.decorators.time_decorator import timing
from core.preprocessing.text_preprocessing import remove_punctuations, stem_sentence, lemmatize_sentence, \
    remove_stopwords, init_nltk
import logging

from core.preprocessing.tokenizers import MyTokenizer
from core.utils.time_utils import timestamp

logger = logging.getLogger(__name__)


def data_preprocessing(data: pd.DataFrame,
                       feature: Text,
                       punctuations: bool = True,
                       lowering: bool = True,
                       stemming: bool = False,
                       lemmatization: bool = True,
                       stop_words: bool = True,
                       save_dir: Text = None):

    prep_data = data.copy(deep=True)
    logger.info('> Data Preprocessing')

    init_nltk()

    if punctuations:
        logger.info('\t> Removing Punctuations')
        prep_data[feature] = prep_data[feature].apply(remove_punctuations)

    if lowering:
        logger.info('\t> Lowering')
        prep_data[feature] = prep_data[feature].apply(lambda x: str(x).lower())

    if stop_words:
        logger.info('\t> Removing Stop Words')
        prep_data[feature] = prep_data[feature].apply(remove_stopwords)

    if stemming:
        logger.info('\t> Stemming')
        prep_data[feature] = prep_data[feature].apply(stem_sentence)

    if lemmatization:
        logger.info('\t> Lemmatization')
        prep_data[feature] = prep_data[feature].apply(lemmatize_sentence)

    prep_data = prep_data.dropna()

    if save_dir:
        filename = f'prep_data_{timestamp()}'
        filepath = f'{save_dir}{filename}.csv'
        prep_data.to_csv(filepath)
        logger.info(f'Preprocessed Data saved at {filepath}')

    return prep_data


def x_to_vector(x,
                params: Dict,
                vectorization_type: Text):

    if vectorization_type == TDIDF_EMBEDDING:
        x_tdidf, tdidf = tdidf_preprocessing(x, params)
        return x_tdidf.toarray()
    elif vectorization_type == TOKENIZER:
        # MyTokenizer
        pass
    else:
        raise AttributeError(f'No valid vectorization type found: {vectorization_type}')


@timing
def tdidf_preprocessing(x,
                        params: Dict):

    n_gram_range = params['n_gram_range'] if 'n_gram_range' in params.keys() else (1, 3)
    max_features = params['max_features'] if 'max_features' in params.keys() else 10000

    tdidf = TfidfVectorizer(ngram_range=n_gram_range,
                            max_features=max_features)

    tdidf.fit(x)

    x_tdidf = tdidf.transform(x)

    return x_tdidf, tdidf





