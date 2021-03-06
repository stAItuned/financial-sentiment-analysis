import logging
from typing import Text, Dict

from sklearn.model_selection import train_test_split

from scripts.data.metrics import report
from scripts.models.utils import init_model
from scripts.pipelines.preprocessing_pipeline import preprocessing_pipeline
from scripts.savings import save_model_data

logger = logging.getLogger()


def model_training(model_name: Text,
                   data_params: Dict,
                   model_params: Dict,
                   seed: int,
                   save_dir=None):

    x, y, dataset, vectorizer = preprocessing_pipeline(data_params)

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=data_params['test_size'],
                                                        shuffle=True,
                                                        random_state=seed)

    model_func = init_model(model_name)
    model = model_func(model_name, model_params)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred = dataset.postprocessing(y_pred, model_name)

    report_df = report(y_test, y_pred)

    logger.info(f' > Model: {model.name}')
    logger.info(f' > Test result: \n {report_df}')

    if save_dir:
        save_model_data(model,
                        data_params,
                        model_params,
                        report_df,
                        save_dir=save_dir)

    return model
