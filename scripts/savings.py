import logging
from typing import Dict, Text
import pandas as pd

from core.file_manager.os_utils import exists, ensure_folder
from core.file_manager.savings import save_json, pickle_save
from core.utils.time_utils import timestamp
from scripts.models.model_class import Model

logger = logging.getLogger(__name__)


def save_model_data(model: Model,
                    data_params: Dict,
                    model_params: Dict,
                    report_df: pd.DataFrame,
                    save_dir: Text):

    stamp = timestamp()
    output_dir = f'{save_dir}{model.name}/{model.name}_{stamp}/'
    ensure_folder(output_dir)

    pickle_save(model, f'{output_dir}{stamp}_{model.name}')
    save_json(data_params, f'{output_dir}{stamp}_data_params.json')
    save_json(model_params, f'{output_dir}{stamp}_model_params.json')
    report_df.to_csv(f'{output_dir}{stamp}_report.csv')

    logger.info(f'> Saving results at {output_dir}')


def save_test(model_name: Text,
              data_params: Dict,
              report_df: pd.DataFrame,
              save_dir: Text):

    stamp = timestamp()
    output_dir = f'{save_dir}{model_name}/{model_name}_{stamp}/'
    ensure_folder(output_dir)

    save_json(data_params, f'{output_dir}{stamp}_data_params.json')
    report_df.to_csv(f'{output_dir}{stamp}_report.csv')

    logger.info(f' > Saving test result at {output_dir}')


