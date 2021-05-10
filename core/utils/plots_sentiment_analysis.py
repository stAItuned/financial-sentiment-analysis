import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from collections import Counter

from nltk import word_tokenize
from nltk.corpus import stopwords

import numpy as np
import pandas as pd


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

    dic_sentiment["negative"] = dic_sentiment.pop(-1)
    dic_sentiment["neutral"] = dic_sentiment.pop(0)
    dic_sentiment["positive"] = dic_sentiment.pop(1)

    return dic_sentiment


def filterQuantiles(data, confidence):
    """
    :param data: (dataframe)     --> dataframe containing the phrases and
                                     their sentiment
    :param confidence: (double)  --> confidence of the length of distribution
                                     in order to remove outliers

    :return: (lists)             --> list with the length values for pos/neg/neutral values
    """
    x_neg = data[data['sentiment'] == -1]["length"]
    x_neutr = data[data['sentiment'] == 0]["length"]
    x_pos = data[data['sentiment'] == 1]["length"]

    # remove the values outiside the 2.5th and 97.5th percentile
    # avoid long tail
    lower_quantile = (100 - confidence) / 2
    upper_quantile = (100 - lower_quantile)

    x_neg_cleaned = [x for x in x_neg if x >= np.percentile(x_neg, lower_quantile) and \
                     x <= np.percentile(x_neg, upper_quantile)]

    x_neutr_cleaned = [x for x in x_neutr if x >= np.percentile(x_neutr, lower_quantile) and \
                       x <= np.percentile(x_neutr, upper_quantile)]

    x_pos_cleaned = [x for x in x_pos if x >= np.percentile(x_pos, lower_quantile) and \
                     x <= np.percentile(x_pos, 97.5)]

    return x_neg_cleaned, x_neutr_cleaned, x_pos_cleaned


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
    fig.update(layout_title_text='SENTIMENT ANALYSIS',
               layout_showlegend=True)

    return fig


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

    fig.update_layout(title="MOST FREQUENT WORDS")

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
                        cols=2, rows=2,
                        vertical_spacing=0.1,
                        horizontal_spacing=0.1
                        )

    # twitter
    data_t = pd.DataFrame(data_t)
    data_t['length'] = data_t.Phrase.apply(lambda text: len(text))
    data_t['sentiment'] = labels_t

    twitter_neg, twitter_neutr, twitter_pos = filterQuantiles(data_t, 95)
    twitter_hist = [twitter_neg, twitter_neutr, twitter_pos]

    # yahoo
    data_y = pd.DataFrame(data_y)
    data_y['length'] = data_y.Phrase.apply(lambda text: len(text))
    data_y['sentiment'] = labels_y

    yahoo_neg, yahoo_neutr, yahoo_pos = filterQuantiles(data_y, 95)
    yahoo_hist = [yahoo_neg, yahoo_neutr, yahoo_pos]

    group_labels = ["negative", "neutral", "positive"]

    colors_twitter = ['#03045e', '#00b4d8', '#caf0f8']
    colors_yahoo = ['#480ca8', '#7209b7', '#f72585']

    # plots
    fig_twitter = ff.create_distplot(twitter_hist, group_labels, colors=colors_twitter)
    fig_yahoo = ff.create_distplot(yahoo_hist, group_labels, group_labels, colors=colors_yahoo)

    distplot_left = fig_twitter['data']
    distplot_right = fig_yahoo['data']

    for i in range(6):
        if i <= 2:
            fig.append_trace(distplot_left[i], 1, 1)
            fig.append_trace(distplot_right[i], 1, 2)

        else:
            fig.append_trace(distplot_left[i], 2, 1)
            fig.append_trace(distplot_right[i], 2, 2)

    fig.update_layout(title_text=f'LENGTH DISTRIBUTION', autosize=False, height=700)

    return fig
