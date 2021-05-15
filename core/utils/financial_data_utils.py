import pandas as pd
import numpy as np
import yfinance as yf

def get_data(ticker,start):
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

    df = pd.DataFrame({'_' : ['price'],
                       'start - $' : [initial_price],
                       'close - $' : [final_price],
                       'gain  /  loss - %': [gain_loss],
                       'other 1 ' : [' / '],
                       'other  2' : [' / ']})
    df = df.set_index('_')

    return df