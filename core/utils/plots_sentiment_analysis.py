import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from collections import Counter

from nltk import word_tokenize
from nltk.corpus import stopwords

import numpy as np
import pandas as pd

import datetime as dt

import streamlit as st

import yfinance as yf


def getOrderedDictionary(data):
    """
    :param data: (Series)       --> sentences

    :return: (string, integer)  --> list with the words and
                                    list with their frequencies
    """
    dict_sentences = {}

    for sentence in data:
        tokens = word_tokenize(sentence)
        for token in tokens:
            if token in dict_sentences and token not in stopwords.words('english'):
                dict_sentences[token] += 1
            else:
                dict_sentences[token] = 1

    ordered_dict = {k: v for k, v in sorted(dict_sentences.items(), key=lambda item: item[1], reverse=True)}

    return list(ordered_dict.keys()), list(ordered_dict.values())


def get_dic_sentiment(labels):
    """
    :param labels: (Series) --> predictions of the sentiment analysis

    :return: (dic)          --> dictionary with the number of positive, negative and neutral values
    """
    dic_sentiment = dict(Counter(labels))

    if -1 in dic_sentiment:
        dic_sentiment["negative"] = dic_sentiment.pop(-1)
    else:
        dic_sentiment["negative"] = 0

    if 0 in dic_sentiment:
        dic_sentiment["neutral"] = dic_sentiment.pop(0)
    else:
        dic_sentiment["neutral"] = 0

    if 1 in dic_sentiment:
        dic_sentiment["positive"] = dic_sentiment.pop(1)
    else:
        dic_sentiment["positive"] = 0

    return dic_sentiment


def getCounters(data):
    """
    :param data: (dataframe)     --> dataframe containing the phrases and
                                     their sentiment

    :return: (lists)             --> list with the length values for pos/neg/neutral values
    """
    x_neg = data[data['sentiment'] == -1]["length"]
    x_neutr = data[data['sentiment'] == 0]["length"]
    x_pos = data[data['sentiment'] == 1]["length"]

    x_neg_dic = pd.Series(x_neg.value_counts())
    x_neutr_dic = pd.Series(x_neutr.value_counts())
    x_pos = pd.Series(x_pos.value_counts())

    return x_neg_dic, x_neutr_dic, x_pos


def unpackSeries(x):
    """
    :param x: (Series) --> index : length of the titles
                       --> values : counter
    :return:
    """
    return x.index, x.values

def plot_piechart(labels_twitter, labels_yahoo):
    """
    :param labels_twitter: (series) --> twitter sentiment
    :param labels_yahoo:   (series) --> yahoo sentiment

    :return: (plt)
    """
    dic_sentiment_twitter = get_dic_sentiment(labels_twitter)
    dic_sentiment_yahoo = get_dic_sentiment(labels_yahoo)

    #colors_twitter = ['#03045e', '#00b4d8', '#caf0f8']
    #colors_yahoo = ['#480ca8', '#7209b7', '#f72585']

    fig = make_subplots(rows=1, cols=2,
                        specs=[[{'type':'domain'}, {'type':'domain'}]],
                        subplot_titles=("Twitter", "Yahoo"))

    fig.add_trace(go.Pie(labels=list(dic_sentiment_twitter.keys()),
                         values=list(dic_sentiment_twitter.values()),
                         name="TWITTER",
                         #marker_colors=colors_twitter,
                         ),
                  1, 1)

    fig.add_trace(go.Pie(labels=list(dic_sentiment_yahoo.keys()),
                         values=list(dic_sentiment_yahoo.values()),
                         name="YAHOO",
                         #marker_colors=colors_yahoo
                         ),

                  1, 2)

    fig.update_traces(hoverinfo='label+percent+name')
    fig.update(#layout_title_text='SENTIMENT ANALYSIS',
               layout_showlegend=True)

    return fig


def plot_informative_table(data_twitter, data_yahoo):

    # get minimum and maximum date
    dates_twitter = [dt.datetime.strptime(date.split(".")[0], '%Y-%m-%dT%H:%M:%S').strftime("%Y-%m-%d %H:%M:%S") for date in data_twitter.index]
    dates_yahoo = [dt.datetime.strptime(date.split("Z")[0], '%Y-%m-%d %H:%M:%S').strftime("%Y-%m-%d %H:%M:%S") for date in data_yahoo.index]

    n_samples = [len(data_twitter), len(data_yahoo)]
    min_dates =  [min(dates_twitter), min(dates_yahoo)]
    max_dates = [max(dates_twitter), max(dates_yahoo)]

    df = pd.DataFrame({ 'Number of samples' : n_samples, 'From' : min_dates, 'To' : max_dates})
    df = df.set_index([pd.Index(['Twitter','Yahoo'])])

    return df


def plot_most_frequent(data_twitter, data_yahoo, k=10):
    """
    :param data: (Series) --> text of the tweets/headlines
    :param k: (int)       --> number of top words to show
    :param dataset: (str) --> type of the dataset (twitter_data/yahoo)

    :return: plot of the top k elements (most frequents)
    """

    palette_twitter = px.colors.sequential.Teal_r
    palette_yahoo = px.colors.sequential.BuPu_r

    keys_twitter, values_twitter = getOrderedDictionary(data_twitter)
    keys_yahoo, values_yahoo = getOrderedDictionary(data_yahoo)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Twitter", "Yahoo"))

    fig.add_trace(go.Bar(x=values_twitter[:k][::-1], y=keys_twitter[:k][::-1], orientation='h'
                         #marker=dict(color=[4, 5, 6], coloraxis="coloraxis")
                         ),
                  1, 1)

    fig.add_trace(go.Bar(x=values_yahoo[:k][::-1], y=keys_yahoo[:k][::-1],orientation='h'
                         # marker=dict(color=[4, 5, 6], coloraxis="coloraxis")
                         ),
                  1, 2)

    #fig.update_layout(title="MOST FREQUENT WORDS")

    return fig


def plot_length_distributions(data_t, labels_t, data_y, labels_y):
    """
    :param data_t: (dataframe) --> twitter dataframe
    :param labels_t: (series)  --> twitter sentiment
    :param data_y: (dataframe) --> yahoo dataframe
    :param labels_y: (series)  --> yahoo sentiment

    :return: (graph)          --> length distribution
    """
    fig = make_subplots(subplot_titles=('Twitter', 'Yahoo'),
                        cols=1, rows=2,
                        vertical_spacing=0.1,
                        horizontal_spacing=0.1
                        )

    # twitter
    data_t = pd.DataFrame(data_t)
    data_t['length'] = data_t.text.apply(lambda text: len(text))
    data_t['sentiment'] = labels_t

    twitter_neg, twitter_neutr, twitter_pos = getCounters(data_t)
    twitter_hist = [twitter_neg, twitter_neutr, twitter_pos]

    # yahoo
    data_y = pd.DataFrame(data_y)
    data_y['length'] = data_y.title.apply(lambda text: len(text))
    data_y['sentiment'] = labels_y

    yahoo_neg, yahoo_neutr, yahoo_pos = getCounters(data_y)
    yahoo_hist = [yahoo_neg, yahoo_neutr, yahoo_pos]

    group_labels = ["negative", "neutral", "positive"]

    colors_twitter = ['#03045e', '#00b4d8', '#caf0f8']
    colors_yahoo = ['#480ca8', '#7209b7', '#f72585']

    # plots
    fig_twitter = ff.create_distplot(twitter_hist, group_labels, colors=colors_twitter, curve_type='kde')
    fig_yahoo = ff.create_distplot(yahoo_hist, group_labels, group_labels, colors=colors_yahoo, curve_type='kde')

    distplot_left = fig_twitter['data']
    distplot_right = fig_yahoo['data']

    for i in range(6):
        if i <= 2:
            fig.append_trace(distplot_left[i], 1, 1)
            fig.append_trace(distplot_right[i], 2, 1)

        else:
            fig.append_trace(distplot_left[i], 1, 1)
            fig.append_trace(distplot_right[i], 2, 1)

    fig.update_layout(#title_text=f'LENGTH DISTRIBUTION',
                      autosize=False, height=700)

    return fig

def plot_length_distributionsV2(data_t, labels_t, data_y, labels_y):
    """
    :param data_t: (dataframe) --> twitter dataframe
    :param labels_t: (series)  --> twitter sentiment
    :param data_y: (dataframe) --> yahoo dataframe
    :param labels_y: (series)  --> yahoo sentiment

    :return: (graph)          --> length distribution
    """

    # twitter
    data_t = pd.DataFrame(data_t)
    data_t['length'] = data_t.text.apply(lambda text: len(text))
    data_t['sentiment'] = labels_t
    data_t['source'] = data_t.text.apply(lambda text: 'Twitter')

    # yahoo
    data_y = pd.DataFrame(data_y)
    data_y['length'] = data_y.text.apply(lambda text: len(text))
    data_y['sentiment'] = labels_y
    data_y['source'] = data_y.text.apply(lambda text: 'Yahoo')

    dic_sentiment = {-1 : "negative", 0 : "neutral", 1:"positive"}
    df_concat = data_t.append(data_y)
    df_concat['sentiment'] = df_concat['sentiment'].apply(lambda x : dic_sentiment[x])

    log_scale = st.checkbox("Logarithmic Scale")
    fig = px.histogram(df_concat, x="length",
                       color="sentiment",
                       opacity=0.8,
                       facet_row="source",
                       log_y=log_scale,  # represent bars with log scale
                       )

    return fig

def plot_sentiment_trend(data, labels, ticker, type_data):
    """
    :param df: ( ) news and respective dates
    :param labels: (list) predicted sentiment
    :param ticker: (str) ticker of the stock
    :return: (Figure) comparison among sentiment and price trends
    """

    fig = make_subplots(2, 1, row_heights=[0.7, 0.3])

    df = pd.DataFrame(data)

    # add the labels on a new 'sentiment' column
    df['sentiment'] = labels

    # filter just positive and negative news

    # since we're showing the daily trend, we group by day
    # we count the number of positive and negative news for each day
    # to do so, we use three new columns
    # |-- day -> feature on which we have to group
    # |-- positive -> 1 if the news is positive, 0 otherwise
    # |-- negative -> -1 if the news is negative, 0 otherwise
    df['day'] = df.index
    df['positive'] = df.sentiment.apply(lambda s: 1 if s == 1 else 0)
    df['negative'] = df.sentiment.apply(lambda s: -1 if s == -1 else 0)

    if type_data == 'TWITTER':
        df['day'] = df['day'].apply(lambda d: d.split("T")[0])
    else:
        df['day'] = df['day'].apply(lambda d: d.split(" ")[0])

    aggregated = df.groupby('day').sum()[['positive', 'negative']]

    data_ticker = yf.download(ticker, start=aggregated.index[0], end=aggregated.index[-1])
    data_ticker_close = data_ticker['Adj Close']

    fig.add_trace(go.Scatter(x=data_ticker_close.index, y=data_ticker_close.values, name=ticker), 1, 1)

    fig.add_trace(go.Bar(x=aggregated.index, y=aggregated.positive, marker_color='green', name="positive"), 2, 1)
    fig.add_trace(go.Bar(x=aggregated.index, y=aggregated.negative, marker_color='red', name="negative"), 2, 1)
    fig.update_layout(barmode='relative', title_text=ticker)

    return fig

