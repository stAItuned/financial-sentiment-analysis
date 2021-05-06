from typing import Text, Dict

from scripts.data.metrics import report
from scripts.models.utils import init_model
from scripts.pipelines.preprocessing_pipeline import preprocessing_pipeline

import logging

from scripts.savings import save_test

logger = logging.getLogger()


def inference_pipeline(model_name: Text,
                       model_path: Text,
                       data_params: Dict,
                       save_dir=None):

    model = init_model(model_name)
    model = model.load(model_path)

    x_test, y_test, dataset = preprocessing_pipeline(data_params)

    pred = model.predict(x_test)
    post_pred = dataset.postprocessing(pred, model_name)

    report_df = report(y_test, post_pred)

    logger.info(f' > Model: {model.name}')
    logger.info(f' > Dataset: {dataset.__class__.__name__}')
    logger.info(f' > Test result: \n {report_df}')

    if save_dir:
        save_test(model_name,
                  data_params,
                  report_df,
                  save_dir)





