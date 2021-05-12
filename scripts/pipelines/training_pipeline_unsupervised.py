import logging
from typing import Text, Dict

from sklearn.model_selection import train_test_split

from scripts.models.utils import init_model
from scripts.pipelines.preprocessing_pipeline import preprocessing_pipeline

logger = logging.getLogger()


def model_training(model_name: Text,
                   data_params: Dict,
                   model_params: Dict,
                   seed: int):

    x, y, dataset, _ = preprocessing_pipeline(data_params)

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=data_params['test_size'],
                                                        shuffle=True,
                                                        random_state=seed)

    model_func = init_model(model_name)
    model = model_func(model_name, model_params)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred = dataset.postprocessing(y_pred, model_name)

    return model, y_pred, x_test
