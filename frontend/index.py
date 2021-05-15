import streamlit as st
import datetime

from core.utils.financial_data_utils import get_data,create_summary

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

        fig = go.Figure(data=[go.Candlestick(x=history.index, open=history.Open,
                                             high=history.High, low=history.Low,
                                             close=history.Close)])

        # responsive layout
        st.plotly_chart(fig, use_container_width=True)

        # summary
        st.dataframe(create_summary(history_close))

