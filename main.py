import json
import streamlit as st
import requests
import os
import random
from datetime import datetime

from constants.config import TRANSFOMER_MODEL, POLYGLON_DATASET
from constants.paths import SCRAPED_NEWS_DIR, PREDICTION_NEWS_DIR
from core.file_manager.os_utils import ensure_folder

from scripts.pipelines.preprocessing_pipeline import preprocessing_pipeline
from core.utils.plots_sentiment_analysis import plot_piechart, plot_informative_table, plot_sentiment_trend
from core.utils.financial_data_utils import *

# setup page
container_info = st.beta_container()
container_ticker = st.beta_container()
container_prediction = st.beta_container()
container_charts = st.beta_container()
container_contancts = st.beta_container()

# endpoint for making the sentiment analysis
ENDPOINT = 'http://localhost:8080/predict/sentiment'


def generate_polyglon_data():
    data_params_polyglon = {'data_path': SCRAPED_NEWS_DIR,
                            'train' : True,
                            'dataset_type': POLYGLON_DATASET,
                            'preprocessed': True,
                            'ticker': input_ticker,
                            'shuffle': False}

    x_polyglon, _, _, _ = preprocessing_pipeline(data_params_polyglon)

    return x_polyglon


# INFO
with container_info:
    st.title('Financial Sentiment Analysis')

    # st.markdown("The goal of the project is to perform a Sentiment Analysis on near-real-time financial news and to compare them with the historical stock prices.", unsafe_allow_html=True)

    st.markdown("**Choose** the *ticker* from this menu items: you can select one of the ticker from "
                "**SP500**. \n"
                "Then, you can look at the **sentiment analysis** for the news related to that ticker, "
                "in the period detailed in the *short summary*. "
                "The first computing of a ticker processing requires few seconds. "
                "Notice that the first time you select a new ticker you should wait few seconds "
                "for the extraction and the "
                "prediction, otherwise you will see it because it is saved in the cache. ", unsafe_allow_html=True)


# SELECT TICKER
with container_ticker:

    slider_obj = st.empty()
    other_companies = get_ticker_list()
    random_integer = random.randint(0, len(other_companies)-1)
    input_company = st.selectbox('Select ticker',other_companies['Security'], 0)

    input_ticker = other_companies[other_companies['Security'] == input_company].index[0]


# NEWS AND PREDICTION
with container_prediction:

    ensure_folder(PREDICTION_NEWS_DIR)
    ensure_folder(SCRAPED_NEWS_DIR)
    PATH_DATA = f"{SCRAPED_NEWS_DIR}{input_ticker}-{str(datetime.today()).split(' ')[0]}_{14}.csv"
    PATH_PREDICTION = f"{PREDICTION_NEWS_DIR}{input_ticker}-{str(datetime.today()).split(' ')[0]}_{14}.csv"

    if os.path.exists(PATH_PREDICTION):
        x_data = pd.read_csv(PATH_PREDICTION).set_index('date')
        sentiment = x_data['sentiment'].to_list()

    else:
        x_news = generate_polyglon_data()

        data = {'sentence': x_news.to_list(),
                'model': TRANSFOMER_MODEL}

        response = requests.post(ENDPOINT, json=data)
        sentiment = json.loads(response.content)['sentiment']

        x_data = pd.DataFrame({'data': x_news,
                               'sentiment': sentiment})
        x_data.to_csv(PATH_PREDICTION)


with container_charts:

    # informative table
    st.markdown(f"<h3> SHORT SUMMARY - {input_ticker}</h3>", unsafe_allow_html=True)
    st.dataframe(plot_informative_table(x_data))

    # pie chart
    st.markdown(f"<h3> SENTIMENT ANALYSIS - {input_ticker}</h3>", unsafe_allow_html=True)
    st.plotly_chart(plot_piechart(sentiment))

    # sentiment trend
    st.markdown(f"<h3> SENTIMENT TREND AND STOCK PRICE - {input_ticker}</h3>", unsafe_allow_html=True)
    st.markdown("Here you can compare the sentiment of the financial news with the correspondent daily stock price. Therefore, it is "
                "possible to see if there is some sort of correlation among them.")
    st.plotly_chart(plot_sentiment_trend(x_data, input_ticker))