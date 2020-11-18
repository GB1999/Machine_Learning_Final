import praw
import pandas as pd
import datetime
import requests
import re
import json
import string
import time as tm
from dateutil import rrule
from textblob import TextBlob
from datetime import datetime, timedelta, date, time

class RedditScraper:
    def __init__(self, subreddits, years):
        self.regex = re.compile('[^a-zA-Z]')
        self.subreddits = subreddits
        self.years = years
        #subreddit = "wallstreetbets"
        self.reddit_df = pd.DataFrame()
        self.start_time = tm.time()

    # Create a function to clean the tittles
    def cleanTxt(self, text):
        return self.regex.sub(' ', text)

    def loadStocks(self, stock_dir):
        self.stocks = pd.read_csv(stock_dir)
        self.dates = self.stocks.columns.tolist()[4:]
        print(self.dates)
        self.dates = [datetime.strptime(date.split()[-1], '%m/%d/%Y') for date in self.dates]
        self.stocks["Date Added Datetime"] = pd.to_datetime(self.stocks['Date Added'])
        self.stocks["Date Removed Datetime"] = pd.to_datetime(self.stocks['Date Removed'])


    def scrape(self):
        self.total_dict = {}
        # for every day between the start date and end date
        for dt in self.dates:
            print(f'Scraping week {dt}')
            week_end_dt = dt + timedelta(days = 6, hours=23, minutes=59, seconds=59)
            # set start and end of day as timestamp
            week_start_ts = int(datetime.timestamp(dt))
            week_end_ts = int(datetime.timestamp(week_end_dt))
            # get stocks that were on SandP 500 at this date
            mask = (self.stocks['Date Added Datetime'] <= dt) & (self.stocks['Date Removed Datetime'] >= week_end_dt)
            cur_stocks = self.stocks[mask]
            # sort stocks by market cap on current day
            cur_stocks = cur_stocks.sort_values(by=['Market Cap: ' + dt.strftime("%m/%d/%Y")])
            #ensure that there are only 500 stocks for current day
            if(len(cur_stocks) > 500):
                cur_stocks = cur_stocks.drop(cur_stocks.index[500:-1])

            print(f'There were {len(cur_stocks)} on the S & P 500 this week')

            weekly_dict = {}

            # for each stock on the S and P 500 that day
            for i in range(len(cur_stocks)):
                print(f'Making request for stock {i} / {len(cur_stocks)}')
                try:
                    # encapsulate with try catch in case invalid fetch request
                    q_subreddits = ",".join(self.subreddits)
                    q_stock = cur_stocks.loc[i, 'Company']

                    weekly_dict[i] = 0
                    posts = []
                    # format request url with start and end date
                    request_url = f'https://api.pushshift.io/reddit/submission/search/?after={week_start_ts}&before={week_end_ts}&sort_type=score&sort=desc&subreddit={q_subreddits}&q={q_stock}'
                    response = requests.get(request_url).json()['data']
                    posts = posts + response

                    for post in posts:
                        clean_title = self.cleanTxt(post['title'])
                        score = int(post['score'])
                        sentiment_score = int(TextBlob(clean_title).sentiment.polarity * score)
                        weekly_dict[i] += (sentiment_score)

                    self.status(q_stock, request_url, response)

                    self.total_dict[dt.strftime("%m/%d/%Y")] = weekly_dict

                except Exception as e:  # This is the correct syntax
                        print(e)

            self.total_pd = pd.DataFrame(self.total_dict)
            print('///////////////////////////////////////////////////////////////////////////////')
            print(f'it took {tm.time()-self.start_time} to collect sentiment data for week {dt}')
            print('///////////////////////////////////////////////////////////////////////////////')
            self.save()

    def status(self, stock, url, posts):
        print(f'Making request from {url}')
        print(f'There were {len(posts)} posts this week for stock {stock}')
        print(posts)

    def save(self):
        self.total_pd.to_csv('./data/stock_sentiment.csv')

subreddits = ['wallstreetbets','RobinHood','DayTrading','Investing','StockMarket', 'SecurityAnalysis', 'stockaday']

rs = RedditScraper(subreddits,5)
rs.loadStocks(r"D:\Documents\UTRGV\Fall Semester 2020\Machine Learning\Group Project\data\Collected\FINAL.csv")
rs.scrape()
rs.save()