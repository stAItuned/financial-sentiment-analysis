import json
from functools import lru_cache

import streamlit as st
import requests

import random

from constants.config import TWITTER_DATASET, YAHOO_DATASET, VADER, TRANSFOMER_MODEL, TSLA_SENTIMENT, AAPL_SENTIMENT, \
    TSLA_PRELOAD, AAPL_PRELOAD, POLYGLON_DATASET
from core.file_manager.loadings import pickle_load
from core.file_manager.os_utils import exists
from core.file_manager.savings import pickle_save
from scripts.pipelines.preprocessing_pipeline import preprocessing_pipeline
from core.utils.plots_sentiment_analysis import plot_piechart, plot_informative_table, plot_sentiment_trend

from core.utils.financial_data_utils import *

container_1 = st.beta_container()
container_2 = st.beta_container()

ENDPOINT = 'http://localhost:8080/predict/sentiment'


def generate_polyglon_data():
    data_params_polyglon = {'data_path': 'news/',
                            'train' : True,
                            'dataset_type': POLYGLON_DATASET,
                            'preprocessed': True,
                            'ticker' : 'AAPL',
                            'shuffle': False}

    x_polyglon, _, _, _ = preprocessing_pipeline(data_params_polyglon)

    return x_polyglon


st.title('HOME')

############################################
# SELECT TICKER
############################################
slider_obj = st.empty()
other_companies = get_ticker_list()
random_integer = random.randint(0, len(other_companies)-1)
input_company = st.selectbox('Select ticker',
                             other_companies['Security'], 0)

input_ticker = other_companies[other_companies['Security'] == input_company].index[0]

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

# informative table
#st.dataframe(plot_informative_table(x_data))

# pie charts
st.markdown(f"<h3> SENTIMENT ANALYSIS - {input_ticker}</h3>", unsafe_allow_html=True)
st.plotly_chart(plot_piechart(sentiment))

# sentiment trend
st.markdown("<h3> SENTIMENT TREND - DATA </h3>", unsafe_allow_html=True)
st.plotly_chart(plot_sentiment_trend(x_data, sentiment, input_ticker, 'TWITTER'))
