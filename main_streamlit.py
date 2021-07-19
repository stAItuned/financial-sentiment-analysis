import json
import streamlit as st
import requests
import os
import random
from datetime import datetime

from constants.config import TRANSFOMER_MODEL, POLYGLON_DATASET

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
    data_params_polyglon = {'data_path': 'news/scraped/',
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

    st.markdown("The goal of the project is to perform a Sentiment Analysis on near-real-time financial news and to compare them with the historical stock prices.", unsafe_allow_html=True)


# SELECT TICKER
with container_ticker:

    slider_obj = st.empty()
    other_companies = get_ticker_list()
    random_integer = random.randint(0, len(other_companies)-1)
    input_company = st.selectbox('Select ticker',other_companies['Security'], 0)

    input_ticker = other_companies[other_companies['Security'] == input_company].index[0]


# NEWS AND PREDICTION
with container_prediction:

    PATH_DATA = f"news/scraped/{input_ticker}-{str(datetime.today()).split(' ')[0]}_{14}.csv"
    PATH_PREDICTION = f"news/prediction/{input_ticker}-{str(datetime.today()).split(' ')[0]}_{14}.csv"

    if os.path.exists(PATH_PREDICTION):
        x_data = pd.read_csv(PATH_PREDICTION).set_index('date')
        sentiment = x_data['sentiment'].to_list()

    else:
        x_news = generate_polyglon_data()

        data = {'sentence': x_news.to_list(),
                'model': TRANSFOMER_MODEL}

        response = requests.post(ENDPOINT, json=data)
        sentiment = json.loads(response.content)['sentiment']

        x_data = pd.DataFrame({'data':x_news, 'sentiment':sentiment})
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