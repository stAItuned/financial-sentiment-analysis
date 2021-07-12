import json
from functools import lru_cache

import streamlit as st
import requests

from constants.config import TWITTER_DATASET, YAHOO_DATASET, VADER, TRANSFOMER_MODEL, TSLA_SENTIMENT, AAPL_SENTIMENT, \
    TSLA_PRELOAD, AAPL_PRELOAD, POLYGLON_DATASET
from core.file_manager.loadings import pickle_load
from core.file_manager.os_utils import exists
from core.file_manager.savings import pickle_save
from scripts.pipelines.preprocessing_pipeline import preprocessing_pipeline
from core.utils.plots_sentiment_analysis import plot_piechart, plot_informative_table, plot_sentiment_trend

container_1 = st.beta_container()
container_2 = st.beta_container()

ENDPOINT = 'http://localhost:8080/predict/sentiment'


def generate_polyglon_data():
    data_params_polyglon = {'data_path': 'news/',
                            'dataset_type': POLYGLON_DATASET,
                            'preprocessed': True,
                            'shuffle': False}

    x_polyglon, _, _, _ = preprocessing_pipeline(data_params_polyglon)

    return x_polyglon


def app():
    # if exists(TSLA_SENTIMENT) and exists(AAPL_SENTIMENT)\
    #         and exists(TSLA_PRELOAD) and exists(AAPL_PRELOAD):
    #     x_twitter = pickle_load(TSLA_PRELOAD)
    #     x_yahoo = pickle_load(AAPL_PRELOAD)
    #     sentiment_twitter = pickle_load(TSLA_SENTIMENT)
    #     sentiment_yahoo = pickle_load(AAPL_SENTIMENT)
    #
    # else:

    x_data = generate_polyglon_data()

    data = {'sentence': x_data.to_list(),
            'model': TRANSFOMER_MODEL}

    response = requests.post(ENDPOINT, json=data)

    sentiment = json.loads(response.content)['sentiment']

    # pickle_save(x_data, TSLA_PRELOAD)
    # pickle_save(x_yahoo, AAPL_PRELOAD)
    # pickle_save(sentiment_twitter, TSLA_SENTIMENT)
    # pickle_save(sentiment_yahoo, AAPL_SENTIMENT)

    # ticker
    st.title(' ** TICKER ** ')

    # informative table
    st.dataframe(plot_informative_table(x_data))

    # pie charts
    st.markdown("<h3> SENTIMENT ANALYSIS </h3>", unsafe_allow_html=True)
    st.plotly_chart(plot_piechart(sentiment))

    # sentiment trend
    st.markdown("<h3> SENTIMENT TREND - DATA </h3>", unsafe_allow_html=True)
    st.plotly_chart(plot_sentiment_trend(x_data, sentiment, 'AAPL', 'TWITTER'))

    # length distribution
    # st.markdown("<h3> LENGTH DISTRIBUTION </h3>", unsafe_allow_html=True)
    # st.plotly_chart(plot_length_distributionsV2(x_twitter, sentiment_twitter, x_yahoo, sentiment_yahoo))
