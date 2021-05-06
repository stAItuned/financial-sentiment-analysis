from typing import Text

import pandas as pd


def extract_dataset(path: Text) -> pd.DataFrame:

    if 'tsv' in path:
        data = pd.read_csv(path, sep='\t')
    elif 'csv' in path:
        data = pd.read_csv(path, index_col=0)
    elif 'txt' in path:
        data = parse_txt_data(path)
    else:
        raise AttributeError(f'Error: path is nor tsv and csv: {path}')

    return data


def parse_txt_data(path: Text):
    encoding = 'utf-8'

    with open(path, 'rb') as f:
        raw_data = f.read()
        f.close()

    raw_data = str(raw_data, encoding, errors='ignore')

    data_dict = {'text': [],
                 'sentiment': []}

    raw_data = raw_data.replace('\r', '')

    for row in raw_data.split('\n'):
        if row != '':
            data_dict['text'].append(row.split('@')[0])
            data_dict['sentiment'].append(row.split('@')[1])

    data = pd.DataFrame(data_dict)

    filepath = f'{path.split(".txt")[0]}.csv'
    data.to_csv(filepath)

    return data
