from typing import Text, Tuple, Dict
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from constants.config import TDIDF_EMBEDDING, TOKENIZER
from core.decorators.time_decorator import timing
from core.preprocessing.text_preprocessing import remove_punctuations, stem_sentence, \
    lemmatize_sentence, remove_stopwords, \
    init_nltk, remove_html, remove_links, \
    normalize_punctuation, normalize_whitespaces, \
    normalize_contractions, normalize_char_sequences, \
    clean_twitter

import logging

from core.utils.time_utils import timestamp
from scripts.data.tokenize import tokenize_dataset

logger = logging.getLogger(__name__)


def data_preprocessing(data: pd.DataFrame,
                       feature: Text,
                       punctuations: bool = True,
                       lowering: bool = True,
                       stemming: bool = False,
                       lemmatization: bool = True,
                       stop_words: bool = True,
                       html: bool = False,
                       links: bool = False,
                       norm_punctuation: bool = False,
                       norm_whitespaces: bool = False,
                       norm_contractions: bool = True,
                       norm_charsequences: bool = True,
                       twitter: bool = False,
                       save_dir: Text = None):
    prep_data = data.copy(deep=True)
    logger.info('> Data Preprocessing')

    init_nltk()

    if norm_contractions:
        logger.info('\t> Normalizing contractions')
        prep_data[feature] = prep_data[feature].apply(normalize_contractions)

    if twitter:
        logger.info('\t> Cleaning twitter patterns')
        prep_data['Phrase'] = prep_data['Phrase'].apply(clean_twitter)

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

    if html:
        logger.info('\t> Removing HTML tags')
        prep_data[feature] = prep_data[feature].apply(remove_html)

    if links:
        logger.info('\t> Removing HTML tags')
        prep_data[feature] = prep_data[feature].apply(remove_links)

    if norm_punctuation:
        logger.info('\t> Normalizing punctuation')
        prep_data[feature] = prep_data[feature].apply(normalize_punctuation)

    if norm_whitespaces:
        logger.info('\t> Normalizing whitespaces')
        prep_data[feature] = prep_data[feature].apply(normalize_whitespaces)

    if norm_charsequences:
        logger.info('\t> Normalizing charsequences')
        prep_data[feature] = prep_data[feature].apply(normalize_char_sequences)

    if twitter:
        logger.info('\t> Cleaning twitter patterns')
        prep_data[feature] = prep_data[feature].apply(clean_twitter)

    prep_data = prep_data.dropna()

    if save_dir:
        filename = f'prep_data_{timestamp()}'
        filepath = f'{save_dir}{filename}.csv'
        prep_data.to_csv(filepath)
        logger.info(f'Preprocessed Data saved at {filepath}')

    return prep_data


def dataset_to_vector(x, y,
                      params: Dict,
                      vectorization_type: Text):

    if vectorization_type == TDIDF_EMBEDDING:
        x_tdidf, tdidf = tdidf_preprocessing(x, params)
        return x_tdidf.toarray(), y, tdidf

    elif vectorization_type == TOKENIZER:
        x, y, tokenizer = tokenize_dataset(x, y)

        return x, y, tokenizer

    elif vectorization_type is None:
        return x
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
