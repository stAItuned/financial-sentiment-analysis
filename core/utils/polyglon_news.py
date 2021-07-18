import requests
import pandas as pd
from datetime import datetime, timedelta


def get_news(ticker, window=14):
    """
    :param ticker: (str) ticker of the stock
    :param window: (int) window for the news (7 days, 14 days ..)
    """
    today = datetime.today()
    start_date = today - timedelta(days=14)
    start_date_str = str(start_date).split(" ")[0]


    with open('files/api_key_polyglon.txt') as f:
        api_key = f.read()

    limit = "500"
    api_url = f"https://api.polygon.io/v2/reference/news?limit={limit}&order=descending&sort=published_utc&ticker={ticker}&published_utc.gte={start_date_str}&apiKey={api_key}"

    data = requests.get(api_url).json()
    headlines, dates, = [], []

    for i in range(len(data['results'])):
        headlines.append(data['results'][i]['title'])
        dates.append(data['results'][i]['published_utc'])

    # preprocessing here (?)

    output_path = f"news/scraped/{ticker}-{str(today).split(' ')[0]}_{window}.csv"

    pd.DataFrame({'text':headlines, 'date':dates}).set_index('date').to_csv(output_path)
