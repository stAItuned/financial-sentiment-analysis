import json
from functools import lru_cache

import streamlit as st
import requests

from constants.config import TWITTER_DATASET, YAHOO_DATASET, VADER, TRANSFOMER_MODEL, TSLA_SENTIMENT, AAPL_SENTIMENT, TSLA_PRELOAD, AAPL_PRELOAD
from core.file_manager.loadings import pickle_load
from core.file_manager.os_utils import exists
from core.file_manager.savings import pickle_save
from scripts.pipelines.preprocessing_pipeline import preprocessing_pipeline
from core.utils.plots_sentiment_analysis import plot_piechart, plot_most_frequent, \
    plot_length_distributionsV2, plot_informative_table

container_1 = st.beta_container()
container_2 = st.beta_container()

ENDPOINT = 'http://localhost:8080/predict/sentiment'


def generate_twitter_data():
    data_params_twitter = {'data_path': 'resources/twitter_dataset/TSLA.csv',
                           'dataset_type': TWITTER_DATASET,
                           'preprocessed': False,
                           'vectorization': None,
                           'vector_params': None,
                           'imbalance': None,
                           'imb_params': None,
                           'shuffle': False,
                           'train': False}

    x_twitter, _, _, _ = preprocessing_pipeline(data_params_twitter)

    return x_twitter


def generate_yahoo_data():
    data_params_yahoo = {'data_path': 'resources/yahoo_dataset/AAPL-News-cleaned.csv',
                         'dataset_type': YAHOO_DATASET,
                         'preprocessed': False,
                         'vectorization': None,
                         'vector_params': None,
                         'imbalance': None,
                         'imb_params': None,
                         'test_size': 0.99,
                         'shuffle': False,
                         'train': False}

    x_yahoo, _, _, _ = preprocessing_pipeline(data_params_yahoo)

    return x_yahoo


def app():

    if exists(TSLA_SENTIMENT) and exists(AAPL_SENTIMENT)\
            and exists(TSLA_PRELOAD) and exists(AAPL_PRELOAD):
        x_twitter = pickle_load(TSLA_PRELOAD)
        x_yahoo = pickle_load(AAPL_PRELOAD)
        sentiment_twitter = pickle_load(TSLA_SENTIMENT)
        sentiment_yahoo = pickle_load(AAPL_SENTIMENT)

    else:

        x_twitter = generate_twitter_data()
        x_yahoo = generate_yahoo_data()

        twitter_data = {'sentence': x_twitter.to_list(),
                        'model': TRANSFOMER_MODEL}

        yahoo_data = {'sentence': x_yahoo.to_list(),
                      'model': TRANSFOMER_MODEL}

        response_twitter = requests.post(ENDPOINT, json=twitter_data)
        response_yahoo = requests.post(ENDPOINT, json=yahoo_data)

        sentiment_twitter = json.loads(response_twitter.content)['sentiment']
        sentiment_yahoo = json.loads(response_yahoo.content)['sentiment']

        pickle_save(x_twitter, TSLA_PRELOAD)
        pickle_save(x_yahoo, AAPL_PRELOAD)
        pickle_save(sentiment_twitter, TSLA_SENTIMENT)
        pickle_save(sentiment_yahoo, AAPL_SENTIMENT)

    # ticker
    st.title('TESLA - TSLA')

    # informative table
    st.dataframe(plot_informative_table(x_twitter, x_yahoo))

    # pie charts
    st.markdown("<h3> SENTIMENT ANALYSIS </h3>", unsafe_allow_html=True)
    st.plotly_chart(plot_piechart(sentiment_twitter, sentiment_yahoo))

    # top k elements
    # st.markdown("<h3> MOST FREQUENT WORDS </h3>", unsafe_allow_html=True)
    # slider_ph = st.empty()
    # [min, max, default, step]
    # value = slider_ph.slider("Select the number of words to show", 5, 15, 10, 1)
    # st.plotly_chart(plot_most_frequent(data, data, value))

    # length distribution
    st.markdown("<h3> LENGTH DISTRIBUTION </h3>", unsafe_allow_html=True)
    st.plotly_chart(plot_length_distributionsV2(x_twitter, sentiment_twitter, x_yahoo, sentiment_yahoo))
