import plotly.express as px
import plotly.figure_factory as ff

from collections import Counter

from nltk import word_tokenize
from nltk.corpus import stopwords

import numpy as np
import pandas as pd


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


def plot_piechart(labels, dataset):
    """
    :param labels: (series)     --> Series with the predictions
    :param dataset: (str)       --> type of the dataset (twitter_data/yahoo)
    """
    dic_sentiment = get_dic_sentiment(labels)

    if dataset == "twitter_data":
        color_palette = px.colors.sequential.Teal_r
        label_title = "TWITTER"
    elif dataset == "YAHOO":
        color_palette = px.colors.sequential.BuPu_r
        label_title = "YAHOO"

    fig = px.pie(values=dic_sentiment.values(), names=dic_sentiment.keys(),
                 title=f'SENTIMENT ANALYSIS - {label_title}',
                 color_discrete_sequence=color_palette)

    return fig


def plot_most_frequent(data, k=10, dataset="TWITTER"):
    """
    :param data: (Series) --> text of the tweets/headlines
    :param k: (int)       --> number of top words to show
    :param dataset: (str) --> type of the dataset (twitter_data/yahoo)

    :return: plot of the top k elements (most frequents)
    """

    if dataset == "twitter_data":
        color_palette = px.colors.sequential.Teal_r
        label_title = "TWITTER"
    elif dataset == "YAHOO":
        color_palette = px.colors.sequential.BuPu_r
        label_title = "YAHOO"

    dict_words = {}

    for sentence in data:
        tokens = word_tokenize(sentence)
        # already removed the stopwords
        for token in tokens:
            if token in dict_words and token not in stopwords.words('english'):
                dict_words[token] += 1
            else:
                dict_words[token] = 1

    ordered_dict = {k: v for k, v in sorted(dict_words.items(), key=lambda item: item[1], reverse=True)}

    keys = list(ordered_dict.keys())[:k]
    values = list(ordered_dict.values())[:k]

    fig = px.bar(x=values, y=keys,

                 category_orders={  # replaces default order by column name
                     "y": [y for x, y in sorted(zip(values, keys), reverse=True)]
                 },

                 title=f"MOST FREQUENT WORDS - {label_title}",
                 color_discrete_sequence=color_palette,

                 labels={
                     "x": "COUNTER",
                     "y": "WORD"
                 })

    return fig


def plot_length_distributions(data, labels, dataset):
    """
    :param data:    (df)  --> complete dataframe with text and prediction
                              # see how can be merged, otherwise merge here
    :param dataset: (str) --> type of input data (twitter/yahoo)
    """
    data = pd.DataFrame(data)
    data['length'] = data.Phrase.apply(lambda text: len(text))
    data['sentiment'] = labels

    x_neg = data[data['sentiment'] == -1]["length"]
    x_neutr = data[data['sentiment'] == 0]["length"]
    x_pos = data[data['sentiment'] == 1]["length"]

    # remove the values outiside the 2.5th and 97.5th percentile
    # avoid long tail
    x_neg_cleaned = [x for x in x_neg if x >= np.percentile(x_neg, 2.5) and \
                     x <= np.percentile(x_neg, 97.5)]

    x_neutr_cleaned = [x for x in x_neutr if x >= np.percentile(x_neutr, 2.5) and \
                       x <= np.percentile(x_neutr, 97.5)]

    x_pos_cleaned = [x for x in x_pos if x >= np.percentile(x_pos, 2.5) and \
                     x <= np.percentile(x_pos, 97.5)]

    hist_data = [x_neg_cleaned, x_neutr_cleaned, x_pos_cleaned]
    group_labels = ["negative", "neutral", "positive"]

    if dataset == "twitter_data":
        colors = ['#03045e', '#00b4d8', '#caf0f8']
        label_title = "TWITTER"
    elif dataset == "YAHOO":
        colors = ['#480ca8', '#7209b7', '#f72585']
        label_title = "YAHOO"

    fig = ff.create_distplot(hist_data, group_labels, colors=colors, bin_size=10,
                             show_curve=False)

    fig.update_layout(title_text=f'LENGTH DISTRIBUTION - {label_title}')

    return fig