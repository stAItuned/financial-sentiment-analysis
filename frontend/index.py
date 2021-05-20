import streamlit as st
import datetime

from core.utils.financial_data_utils import *

import plotly.express as px
import plotly.graph_objects as go

import random

def app():
    st.title('HOME')

    slider_obj = st.empty()

    ############################################
    # SELECT TICKER
    ############################################
    other_companies = get_ticker_list()
    random_integer = random.randint(0, len(other_companies)-1)
    input_company = st.selectbox('Select ticker',
                                 other_companies['Security'], random_integer)

    input_ticker = other_companies[other_companies['Security'] == input_company].index[0]
    value = slider_obj.slider("Select the number of months: ", 4, 60, 24, 1)

    ############################################
    #  CANDLESTICK PLOT
    ############################################
    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=value * 30)
    history = get_data(input_ticker, start_date)

    fig_candlestick = go.Figure(data=[go.Candlestick(x=history.index, open=history.Open,
                                                     high=history.High, low=history.Low,
                                                     close=history.Close)])

    # responsive layout
    st.plotly_chart(fig_candlestick, use_container_width=True)

    ############################################
    # SUMMARY
    ############################################
    st.markdown("<h2> SUMMARY </h2>", unsafe_allow_html=True)
    st.dataframe(create_summary(history))

    st.markdown("<br /><br />", unsafe_allow_html=True)

    ############################################
    # BENCHMARK COMPARISON
    ############################################
    st.markdown("<h2> BENCHMARK COMPARISON </h2>", unsafe_allow_html=True)
    selected_companies = st.multiselect('Compare it with other companies on the same sector on the S&P500',
                                        other_companies['Security'])

    st.plotly_chart(plot_sector_companies(input_ticker, start_date,
                                          selected_companies, other_companies),
                    use_container_width=True)
