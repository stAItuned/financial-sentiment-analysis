import streamlit as st

from constants.config import TWITTER_DATASET, VADER
from scripts.pipelines.training_pipeline_unsupervised import model_training

from core.utils.plots_sentiment_analysis import plot_piechart, plot_most_frequent, plot_length_distributions

container_1 = st.beta_container()
container_2 = st.beta_container()

def app():

    model_name = VADER
    seed = 2021

    data_params = {'data_path': 'resources/twitter_dataset/TSLA.csv',
                   'dataset_type': TWITTER_DATASET,
                   'preprocessed': False,
                   'vectorization': None,
                   'vector_params': None,
                   'imbalance': None,
                   'imb_params': None,
                   'test_size': 0.99,
                   'shuffle': False,
                   'train': False}

    model_params = {}

    _, labels, data = model_training(model_name, data_params, model_params, seed)

    # ticker
    st.title('TESLA - TSLA')

    # pie charts
    st.plotly_chart(plot_piechart(labels, labels))

    # top k elements
    slider_ph = st.empty()
    # [min, max, default, step]
    value = slider_ph.slider("slider", 5, 15, 10, 1)
    st.plotly_chart(plot_most_frequent(data, data, value))

    # length distribution
    st.plotly_chart(plot_length_distributions(data, labels, data, labels))

