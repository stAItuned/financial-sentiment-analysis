import streamlit as st
import datetime

from core.utils.financial_data_utils import *

import plotly.express as px
import plotly.graph_objects as go


def app():

        st.title('HOME')

        label_obj = st.empty()
        slider_obj = st.empty()

        ticker_input = label_obj.text_input("Select ticker", 'AAPL')
        value = slider_obj.slider("Select the number of months: ", 4, 60, 24, 1)

        today = datetime.date.today()
        start_date = today - datetime.timedelta(days=value * 30)
        history = get_data(ticker_input, start_date)
        history_close = history['Adj Close']

        fig_candlestick = go.Figure(data=[go.Candlestick(x=history.index, open=history.Open,
                                             high=history.High, low=history.Low,
                                             close=history.Close)])

        # responsive layout
        st.plotly_chart(fig_candlestick, use_container_width=True)

        # summary
        st.markdown("<h3> SUMMARY </h3>", unsafe_allow_html=True)
        st.dataframe(create_summary(history_close))

        # comparison
        st.markdown("<h3> BENCHMARK COMPARISON </h3>", unsafe_allow_html=True)
        sector, similar_companies = get_sector(ticker_input)
        selected_companies = st.multiselect('Compare it with other companies on the same sector on the S&P500',
                                             similar_companies)

        st.plotly_chart(plot_sector_companies(ticker_input, start_date, selected_companies), use_container_width=True)
