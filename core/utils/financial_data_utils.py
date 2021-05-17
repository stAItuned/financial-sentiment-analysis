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


def create_summary(history_close):
    """
    :param history_close:   historical close prices for the selected ticker
    :return:                dataframe with a short summary of the price and
                            volumes over the whole period
    """

    initial_price = round(history_close[0], 2)
    final_price = round(history_close[len(history_close) - 1], 2)
    gain_loss = round(100 * (final_price - initial_price) / initial_price, 2)

    df = pd.DataFrame({'_': ['price'],
                       'start - $': [initial_price],
                       'close - $': [final_price],
                       'gain  /  loss - %': [gain_loss],
                       'other 1 ': [' / '],
                       'other  2': [' / ']})
    df = df.set_index('_')

    return df


def get_sector(ticker):
    """
    :param ticker: (String) ticker that we want to track
    :return: (String) sector of the company
    :return: (list) companies of the same sector
    """

    # because of streamlit we're on the main director
    df = pd.read_csv('resources/dictionaries/SP500-Sectors.csv').set_index('Symbol')

    sector = df.loc[ticker]['GICS Sector']
    similar_companies = list(df[df['GICS Sector'] == sector].index)

    return sector, similar_companies


def plot_sector_companies(ticker, date, other_companies):
    """
    :param ticker: (String) ticker that we want to track
    :param date: (date) initial date of the comparison
    :param other_companies: (list) other companies on the same sector as 'ticker'
    :return: (Figure) linechart with the comparison among the selected companies
    """
    # selected company
    main_history = get_data(ticker, date)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=main_history.index, y=main_history['Adj Close'], name=ticker))

    for company in other_companies:
        curr_history = get_data(company, date)
        fig.add_trace(go.Scatter(x=curr_history.index, y=curr_history['Adj Close'], name=company))

    return fig