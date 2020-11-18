import praw
import pandas as pd
import datetime
import requests
import re
import json
import time as tm
from dateutil import rrule
from datetime import datetime, timedelta, date, time

API_KEY = "AFXIRVO0L45TIAMV"
df_reddit = pd.read_csv(r"/Group Project/data/Downloaded/market_cap_weekly.csv")
start_date = df_reddit.columns.tolist()[-1]
df_reddit["Date Removed Datetime"] = pd.to_datetime(df_reddit['Date Removed'])
start_date = datetime.strptime(start_date.split()[-1], '%m/%d/%Y')

mask = (df_reddit['Date Removed Datetime'] >= start_date)
df_reddit = df_reddit[mask]

null_dates = df_reddit[df_reddit['Market Cap: 11/13/2020'].isnull()]

weeks = (52 * 5)
dates = df_reddit.columns[5:]
print(len(dates))

print(null_dates)
# for each stock (up to 500 per day for free API)
for i in null_dates.index:
    ticker = null_dates.loc[i, "Ticker"]
    try:
        request_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY_ADJUSTED&symbol={ticker}&outputsize=full&apikey={API_KEY}"
        data = requests.get(request_url).json()['Weekly Adjusted Time Series']
        print(data)
        market_cap = [
                         float(stock_dict["4. close"]) * float(stock_dict["6. volume"])
                         for stock_dict in list(data.values())
                     ][:weeks]
        print(len(market_cap))

        df_reddit.loc[df_reddit['Ticker'] == ticker, dates[:len(market_cap)]] = market_cap



    except Exception as e:
        print(e)

    # sleep to avoid api limitations

    tm.sleep(12)
df_reddit.to_csv('./data/filled.csv')
print(df_reddit.head())
print(df_reddit.tail())