import json

import pandas as pd
import re

from datetime import datetime, timedelta
import requests
import pandas as pd

# TWITTER APIs SETUP
f = open("../resources/barrer-token.txt", "r")
BEARER_TOKEN = f.read()

endpoint = 'https://api.twitter.com/2/tweets/search/recent'
headers = {'authorization': f'Bearer {BEARER_TOKEN}'}    


def get_data(tweet):
    """
    :param tweet: scraped tweet gained by APIs
    
    :return data: return date and text from the input object
    """
    data = {
        'created_at': tweet['created_at'],
        'text': tweet['text']
    }
    return data


def time_travel(now, mins):
    """
    :param now: initial date
    :param mins: minutes that we want to decrease from the starting time

    :return back_in_time: updated date
    """
    now = datetime.strptime(now, dtformat)
    back_in_time = now - timedelta(minutes=mins)
    return back_in_time.strftime(dtformat)


def scrape_tweets(start_date, end_date, query):
    """
    :param start_date: initial date for scraping tweets
    :param end_date: last date for scraping tweets
    :param query: text that we want to intercept from the tweets

    :return df: dataframe containing the scraped tweets
    """

    params = {
        'query': query,
        'max_results': '100',
        'tweet.fields': 'created_at,lang'
    }

    df = pd.DataFrame()              # initialize dataframe to store tweets

    while True:
        
        if datetime.strptime(start_date, dtformat) < end_date:
            # if we have reached the "liimit"
            break
        
        pre60 = time_travel(start_date, 60)             # scrape 100 tweets every hour 
        
        params['start_time'] = pre60
        params['end_time'] = start_date
        
        response = requests.get(endpoint,                # request tweets within the new date range
                                params=params,
                                headers=headers)  
        
        for tweet in response.json()['data']:

            row = get_data(tweet)  
            df = df.append(row, ignore_index=True)

        
        print(f"{start_date} - {pre60} \t Done.")

        start_date = pre60                                       # shift

    return df

    


dtformat = '%Y-%m-%dT%H:%M:%SZ'  # the date format string required by twitter
now = datetime.now()             # current date
end = now - timedelta(days=1)    # last date -> 2 days scraping data
now = now.strftime(dtformat)     # convert the current datetime to the format used for API

now = time_travel(now, 180)      # Twitter API doesn't allow to scrape real time data
                                 # so, I go 3hrs backwards


query = '(tesla OR tsla OR elon musk) (lang:en)'

df_scraped = scrape_tweets(now,end,query)

df_scraped.to_csv('../data/TSLA.csv',index=False)       # save the CSV file