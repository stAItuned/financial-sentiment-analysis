import pandas as pd
import numpy as np
import yfinance as yf

import pandas as pd

import plotly.graph_objects as go


def get_data(ticker, start):
    """
    :param ticker: ticker that we want to track
    :param start: start date of the analysis (1 year ago, by default)
    :return: historical data
    """
    return yf.download(ticker, start=start)


def create_summary(history):
    """
    :param history_close: (DataFrame)  historical prices and volume
    :return:  (DataFrame)        Summary on the changes over time
    """

    initial_price = round(history['Adj Close'][0], 2)
    final_price = round(history['Adj Close'][-1], 2)
    gain_price = round(100 * (final_price - initial_price) / initial_price, 2)

    initial_volume = round(history['Volume'][0], 2)
    final_volume = round(history['Volume'][-1], 2)
    gain_volume = round(100 * (final_volume - initial_volume) / initial_volume, 2)

    df = pd.DataFrame({'_': ['Price', 'Volume'],
                       f'    {str(history.index[0]).split(" ")[0]} ': [initial_price, initial_volume],
                       f'    {str(history.index[-1]).split(" ")[0]} ': [final_price, final_volume],
                       '    Gain / Loss - % ': [gain_price, gain_volume]
                       })

    df = df.set_index('_')

    return df


def get_ticker_list():
    """
    :param ticker: (String) ticker that we want to track
    :return: (String) sector of the company
    # :return: (list) ticker of the companies of the same sector
    # :return: (list) name of the companies of the same sector
    """

    # because of streamlit we're on the main director
    df = pd.read_csv('resources/dictionaries/S&P-Sectors.csv').set_index('Symbol')

    return df


def plot_sector_companies(ticker, date, other_companies, mapping):
    """
    :param ticker: (String) ticker that we want to track
    :param date: (date) initial date of the comparison
    :param other_companies: (list) other companies on the same sector as 'ticker'
    :return: (Figure) linechart with the comparison among the selected companies
    """
    # selected company
    main_history = get_data(ticker, date)
    company_name = mapping.loc[ticker]['Security']

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=main_history.index, y=main_history['Adj Close'], name=f"{ticker} - {company_name}"))

    for company in other_companies:
        ticker = mapping[mapping['Security'] == company].index[0]
        curr_history = get_data(ticker, date)
        fig.add_trace(go.Scatter(x=curr_history.index, y=curr_history['Adj Close'], name=f"{ticker} - {company}"))
    fig.update_layout(showlegend=True)

    return fig
