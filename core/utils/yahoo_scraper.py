import re
import csv
from time import sleep
from bs4 import BeautifulSoup
import requests
import pandas as pd

request = 'https://news.search.yahoo.com/search?p={}'

headers = {
    'accept': '*/*',
    'accept-encoding': 'gzip, deflate, br',
    'accept-language': 'en-US,en;q=0.9',
    'referer': 'https://www.google.com',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36 Edg/85.0.564.44'
}


def scrape_news(query):
    """
    :param query: string to search on "yahoo.news"

    :return df: dataframe containing all the results

        Scraping over all the results within each page.
        If the button "next" (referred to the page) is available,
        it continues to iterate over all the pages
    """

    url = request.format(query)

    titles = []
    sources = []
    posted = []
    descriptions = []

    # iterate over all the available pages
    while True:

        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        # there is a "NewsArticle" class for each news headline
        # returns an array with all the articles
        articles = soup.find_all('div','NewsArticle')


        # each element has a specific tag and class
        # in order to extract the text from the html tag, use ".text"
        for article in articles:

            titles.append(article.find('h4','s-title').text.strip())
            sources.append(article.find('span','s-source').text.strip())
            posted.append(article.find('span','s-time').text.replace('.', '').strip())
            descriptions.append(article.find('p','s-desc').text.strip())

        try:
            # look for the "next" button
            # and get the link
            url = soup.find('a','next').get('href')
            time.sleep(1)
        except AttributeError:
            # otherwise exit
            break

    df = pd.DataFrame({'title' : titles, 'source' : sources, 'posted' : posted, 'description' : descriptions})

    # it might have duplicates over all the pages
    return df.drop_duplicates()



df_scraped = scrape_news('Apple')

df_scraped.to_csv('../data/AAPL-News.csv',index=False)