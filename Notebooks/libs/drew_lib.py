import os
import requests
import pandas as pd
from dotenv import load_dotenv
import datetime as dt
import alpaca_trade_api as tradeapi
import json
from alpaca_trade_api.rest import REST, TimeFrame
load_dotenv()



# get ohlcv data for individual stock with alpaca api call and make into a dataframe

alpaca_key = os.getenv("ALPACA_API_KEY")
alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY")

tradeapi = REST(alpaca_key, alpaca_secret_key)

def get_daily_trade_data(ticker, start_date_str, end_date_str, tradeapi): 
    ticker_df = tradeapi.get_bars(
        ticker,
        TimeFrame.Day,
        start_date_str,
        end_date_str
    ).df
    ticker_df['symbol'] = ticker

    return ticker_df



# create a dataframe of closing prices and ticker symbols from a list of ticker symbols

ticker_list = pd.read_csv('./Data/Cleaned_Data/Ticker_library.csv')['Ticker'].to_list()

def make_tickers_df(ticker_list, start_date_str, end_date_str, tradeapi):
    ticker_dfs_list = [get_daily_trade_data(ticker, start_date_str, end_date_str, tradeapi) for ticker in ticker_list]
    tickers_df = pd.concat(ticker_dfs_list, axis=0, join='outer')
    tickers_df.index = tickers_df.index.date
    tickers_df = tickers_df[['ticker','close','volume']]
    return tickers_df



# use list of tickers to make dictionary with tickers as keys and ohlcv data as values (could be useful for alternative data manipulation)

def make_tickers_dict(ticker_list, start_date_str, end_date_str, tradeapi):
    return {ticker: get_daily_trade_data(ticker, start_date_str, end_date_str, tradeapi) for ticker in ticker_list}