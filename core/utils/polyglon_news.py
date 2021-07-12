import requests
import pandas as pd
from datetime import datetime, timedelta


def get_news(ticker, window):
    """
    :param ticker: (str) ticker of the stock
    :param window: (int) window for the news (7 days, 14 days ..)
    """

    start_date = str( datetime.today() - timedelta(days=window) ).split(" ")[0]

    # with open('../../files/api_key_polyglon.txt') as f:
    with open('files/api_key_polyglon.txt') as f:
        api_key = f.read()

    limit = "500"
    api_url = f"https://api.polygon.io/v2/reference/news?limit={limit}&order=descending&sort=published_utc&ticker={ticker}&published_utc.gte={start_date}&apiKey={api_key}"

    data = requests.get(api_url).json()
    headlines, dates, = [], []

    for i in range(len(data['results'])):
        headlines.append(data['results'][i]['title'])
        dates.append(data['results'][i]['published_utc'])

    # preprocessing here (?)

    first_date = dates[-1].split("T")[0]
    last_date = dates[0].split("T")[0]

    # output_path = f"../../news/{ticker}-{first_date}_{last_date}.csv"
    # output_path = f"../../news/{ticker}-{last_date}_{window}.csv"
    output_path = f"news/{ticker}-{last_date}_{window}.csv"

    pd.DataFrame({'text':headlines, 'date':dates}).to_csv(output_path, index=False)