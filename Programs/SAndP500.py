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
class SAndP500:
    def __init__(self, cur_csv_dir, chng_csv_dir):
        # read list of stocks currently on S & P 500
        self.stocks = pd.read_csv(cur_csv_dir)
        # read list of changes made to S & P 500
        self.changes = pd.read_csv(chng_csv_dir)


    def prepare(self):

        self.stocks.drop('GICS Sector', axis=1, inplace=True)
        self.stocks.drop('GICS Sub-Industry', axis=1, inplace=True)
        self.stocks.drop('SEC filings', axis=1, inplace=True)
        self.stocks.drop('Headquarters Location', axis=1, inplace=True)
        self.stocks.drop('CIK', axis=1, inplace=True)
        self.stocks.drop('Founded', axis=1, inplace=True)

        # fill empty cells
        self.stocks.loc[self.stocks["Date Added"].isnull(), 'Date Added'] = "12/31/2014"
        self.stocks['Date Removed'] = datetime.now().strftime("%m/%d/%Y")
        # reformat changes date format to match stocks
        self.changes['Date'] = self.changes['Date'].apply(lambda x: datetime.strptime(x, '%d-%b-%y').strftime('%m/%d/%Y'))



    def apply_changes(self):
        # for each row in S & P 500 changes
        max_date = datetime.now().strftime("%m/%d/%Y")
        min_date = "12/31/2014"
        for i in reversed(range(len(self.changes))):
            # add stock
            change_date = self.changes.loc[i, "Date"]
            new_ticker = self.changes.loc[i, "New Ticker"]
            old_ticker = self.changes.loc[i, "Old Ticker"]
            new_company = self.changes.loc[i, "New Company"]
            old_company = self.changes.loc[i, "Old Company"]
            self.add_stock(new_company, new_ticker, change_date, max_date)
            self.add_stock(old_company, old_ticker, min_date, change_date)

        # drop S & P 500 companies outside of 5 year period
        self.stocks["Date Removed Datetime"] = pd.to_datetime(self.stocks['Date Removed'])
        self.stocks.drop(self.stocks[self.stocks['Date Removed Datetime'] < datetime.strptime(min_date, '%m/%d/%Y')].index, inplace = True)

    def add_stock(self, company, ticker, date_added, date_replaced):
        # if ticker is already in list, update add and removal dates
        if ticker in self.stocks['Ticker'].tolist():
            self.stocks.loc[self.stocks['Ticker'] == ticker, "Date Added"] = date_added
            self.stocks.loc[self.stocks['Ticker'] == ticker, "Date Removed"] = date_replaced
        else:
            self.stocks = self.stocks.append({
                'Ticker': ticker,
                'Company': company,
                'Date Added': date_added,
                'Date Removed': date_replaced
            },
            ignore_index=True)

    def format(self):
        try:
            self.stocks['Date Added'] = self.stocks['Date Added'].apply(
                lambda x: datetime.strptime(x, '%m/%d/%Y').strftime('%d/%m/%Y'))
            self.stocks['Date Removed'] = self.stocks['Date Removed'].apply(
                lambda x: datetime.strptime(x, '%m/%d/%Y').strftime('%d/%m/%Y'))
        except Exception as e:
            print(e)

    def cal_market_cap(self):
        end_date = datetime.combine(date.today(), time())
        # number of weeks stop market is open a year
        weeks = (52 * 5)

        # get dates that stock market was open
        request_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY_ADJUSTED&symbol=IBM&outputsize=full&apikey={API_KEY}"
        data = requests.get(request_url).json()['Weekly Adjusted Time Series']
        dates = ['Market Cap: ' + datetime.strptime(date, '%Y-%m-%d').strftime('%m/%d/%Y') for date in
                 list(data.keys())][:weeks]
        self.stocks[dates] = ""

        # for each stock (up to 500 per day for free API)
        for i in range(len(self.stocks)):
            ticker = self.stocks.loc[i, "Ticker"]
            try:
                request_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY_ADJUSTED&symbol={ticker}&outputsize=full&apikey={API_KEY}"
                data = requests.get(request_url).json()['Weekly Adjusted Time Series']
                market_cap = [
                            float(stock_dict["4. close"]) * float(stock_dict["6. volume"])
                            for stock_dict in list(data.values())
                              ][:weeks]

                # not all stocks will go to last date
                self.stocks.loc[i, dates[:len(market_cap)]] = market_cap



            except Exception as e:
                print(e)

            # sleep to avoid api limitations

            tm.sleep(12)
            sp.save('./data/Downloaded/market_cap_weekly2.csv')
            print(self.stocks.head())
            print(self.stocks[dates[0]])



    def save(self, save_dir):
        self.stocks.to_csv(save_dir)

sp = SAndP500(
    r'/Group Project/data/Downloaded/S&P 500 component stocks.csv',
    r'D:\Documents\UTRGV\Fall Semester 2020\Machine Learning\Group Project\data\Downloaded\S&P 500 components changes.csv'
)

sp.prepare()
sp.apply_changes()
sp.cal_market_cap()
sp.save('./data/Downloaded/market_cap_weekly2.csv')

