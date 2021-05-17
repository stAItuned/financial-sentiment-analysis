import logging
from typing import Text, Dict

from scripts.data.metrics import report
from scripts.models.utils import init_model
from scripts.pipelines.preprocessing_pipeline import preprocessing_pipeline
from scripts.savings import save_model_data

logger = logging.getLogger()


def inference_without_trained_model(model_name: Text,
                                    data_params: Dict,
                                    model_params: Dict,
                                    save_dir=None):

    x, y, dataset, _ = preprocessing_pipeline(data_params)

    model_func = init_model(model_name)
    model = model_func(model_name, model_params)

    y_pred = model.predict(x)
    y_pred = model.postprocessing(y_pred)

    report_df = report(y, y_pred)

    logger.info(f' > Model: {model.name}')
    logger.info(f' > Dataset: {dataset.__class__.__name__}')
    logger.info(f' > Test result: \n {report_df}')

    if save_dir:
        save_model_data(None,
                        data_params,
                        model_params,
                        report_df,
                        save_dir=save_dir)

    return model
