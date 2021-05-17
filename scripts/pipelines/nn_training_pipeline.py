import logging
from typing import Text, Dict
import numpy as np
from scripts.data.metrics import report
from scripts.datasets.dataloader import generate_dataloader
from scripts.models.utils import init_model
from scripts.pipelines.preprocessing_pipeline import preprocessing_pipeline
from scripts.savings import save_model_data

logger = logging.getLogger()


def model_training_nn(model_name: Text,
                      data_params: Dict,
                      dataloader_params: Dict,
                      model_params: Dict,
                      save_dir=None):
    x, y, dataset, vectorizer = preprocessing_pipeline(data_params)

    dataloader = generate_dataloader(x, y, dataloader_params)

    model_params['dataloader'] = dataloader
    model_params['network']['n_words'] = vectorizer.get_n_words() if vectorizer is not None else None

    model_func = init_model(model_name)
    model = model_func(model_name, model_params)

    model.fit()
    y_pred = np.ravel([model.predict(x) for x in dataloader['valid']])
    y_pred_post = dataset.postprocessing(y_pred, model_name)
    y_true = np.ravel([x[1].numpy() for x in dataloader['valid']])

    report_df = report(y_true, y_pred_post)

    logger.info(f' > Model: {model.name}')
    logger.info(f' > Test result: \n {report_df}')

    # if save_dir:
    #     save_model_data(model,
    #                     data_params,
    #                     model_params,
    #                     report_df,
    #                     save_dir=save_dir)

    return model
