from typing import Text

import requests
import pandas as pd
from datetime import datetime, timedelta

from constants.paths import POLYGLON_KEY_PATH, SCRAPED_NEWS_DIR


def get_news(ticker: Text,
             window: int = 14):
    """
    :param ticker: (str) ticker of the stock
    :param window: (int) window for the news (7 days, 14 days ..)
    """
    today = datetime.today()
    start_date = today - timedelta(days=window)
    start_date_str = str(start_date).split(" ")[0]

    with open(POLYGLON_KEY_PATH) as f:
        api_key = f.read()

    limit = "500"
    api_url = f"https://api.polygon.io/v2/reference/news?limit={limit}&order=descending&sort=published_utc&ticker={ticker}&published_utc.gte={start_date_str}&apiKey={api_key}"

    data = requests.get(api_url).json()
    headlines, dates, = [], []

    for i in range(len(data['results'])):
        headlines.append(data['results'][i]['title'])
        dates.append(data['results'][i]['published_utc'])

    output_path = f"{SCRAPED_NEWS_DIR}{ticker}-{str(today).split(' ')[0]}_{window}.csv"

    pd.DataFrame({'text': headlines,
                  'date': dates}).set_index('date').to_csv(output_path)
