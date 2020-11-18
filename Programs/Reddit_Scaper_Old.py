import praw
import requests
import pandas as pd
from datetime import datetime
import numpy as np
from bs4 import BeautifulSoup

class DataCollection:
    pd.set_option('display.max_columns', None)
    def get_tickers(self):
        # grab HTML Text
        yahoo_source = requests.get("https://finance.yahoo.com/trending-tickers").text
        soup = BeautifulSoup(yahoo_source, features="html.parser")
        data = [[]]
        for marker in soup.find_all("tr"):
            temp = []
            for child in marker.children:
                if 'data-col' in child.get("class")[0]:
                    # print(dir(child))
                    for grandchild in child.descendants:
                        if "<" not in str(grandchild):
                            temp.append(str(grandchild))
            data.append(temp)

        self.df = pd.DataFrame(data = data, columns=["Ticker", "Name", "Last Price", "Market Time", "Change", "Percent Change", "Volume", "Average Vol", "Market Cap"],)


    def get_reddit_posts(self, username, password, sub_red):
        reddit = praw.Reddit(client_id = "EBrucUwHH1wSlw",
                             client_secret = "q21oq4aCpUSXE0zgAcsGsAuriDs",
                             username = username,
                             password = password,
                             user_agent = "reddit_scaper",)

        subreddit = reddit.subreddit(sub_red)
        hot_python = subreddit.hot(limit = 100)

        self.reddit_df = pd.DataFrame()
        for submission in hot_python:
            for ticker in self.df["Ticker"]:
                if len(str(ticker)) > 1 and str(ticker) in str(submission.title):
                    print(ticker)
                    data = {"Title": submission.title,
                            "Date Published": datetime.utcfromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                            "ID": submission.id,
                            "Popularity": submission.score,
                            "Number Comments": submission.num_comments,
                            }
                    self.reddit_df = self.reddit_df.append(pd.DataFrame(data, index=[0]), ignore_index=True)

    def print_data(self):
        print(self.df)
        print(self.reddit_df)
    def save_data(self):
        self.df.to_csv('./data/ticker.csv')
        self.reddit_df.to_csv('./data/reddit.csv')

data_col = DataCollection()
data_col.get_tickers()
data_col.get_reddit_posts("BlunderJach", "Cerberus123!", "wallstreetbets")
data_col.print_data()
data_col.save_data()