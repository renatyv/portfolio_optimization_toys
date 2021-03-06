import warnings
from typing import Optional

import pandas as pd
import yfinance as yf
import os
from yahoo_fin import stock_info as yfsi
import tqdm
import numpy as np

from pypoanal.assets import SharesHistory

SHARES_INFO_FILEPATH = 'info/tickers_info.csv'
DATA_DIR = 'priceVolData'


def _price_vol_path(ticker: str) -> str:
    return os.path.join(DATA_DIR, ticker + '.csv')


def get_quote_type(ticker: str) -> Optional[str]:
    """
    downloads qoute type ('EQUITY' or 'ETF' or 'MUTUAL FUND' etc.)
    for a ticker from Yahoo Finance.
    :rtype: str
    :param ticker: 'MSFT' or 'GOOG', etc...
    :return: None if there is no data or 'ETF' or 'EQUITY' or 'MUTUAL FUND',
    etc.
    """
    try:
        quote_type = yfsi.get_quote_data(ticker).get('quoteType', None)
        return quote_type
    except IndexError:
        return None


def _download_price_volume_history(ticker: str, show_errors=True) -> pd.DataFrame:
    """ Downloads ticker Adj Price and Volume history from Yahoo Finance
    Returns the corresponding pandas DataFrame object
    :param ticker: 'MSFT' or 'GOOG', etc...
    :return: dataframe with Date as index and two columns:
    adjusted price, volume
    """
    price_volume_history = yf.download([ticker], period="max", progress=False, show_errors=show_errors, threads=False)
    return price_volume_history.drop(["Close", "Open", "Low", "High"], axis=1).dropna(how="all")


def download_and_save_price_history(tickers_to_download: list[str]) -> list[str]:
    """ downloads prices for list of tickers to
    corresponing csv file in priceVolData directory
    :returns list of tickers for which download failed
    """
    tickers_failed_to_download = []
    for ticker in tqdm.tqdm(tickers_to_download):
        try:
            price_and_volume_df = _download_price_volume_history(ticker, show_errors=False)
            if len(price_and_volume_df) > 0:
                price_and_volume_df.to_csv(_price_vol_path(ticker))
            else:
                tickers_failed_to_download.append(ticker)
        except Exception as e:
            tickers_failed_to_download.append(ticker)
            print(e)
    return tickers_failed_to_download


def _load_price_volume_history(ticker: str) -> pd.DataFrame:
    """
    Has side effect of downloading the ticker if it can not be read locally
    :param ticker:
    :return: table indexed with date, two columns: 'Adj Close', 'Volume'
    """
    filepath = _price_vol_path(ticker)
    if os.path.exists(filepath):
        price_volume_history = pd.read_csv(filepath,
                                           parse_dates=['Date'],
                                           index_col='Date',
                                           dtype={'Adj Close': np.float64,
                                                  'Volume': np.float64,
                                                  'Close': np.float64,
                                                  'Open': np.float64,
                                                  'High': np.float64,
                                                  'Low': np.float64})
        return price_volume_history
    else:
        price_volume_history = _download_price_volume_history(ticker)
        price_volume_history.to_csv(filepath)
        return price_volume_history


def load_price_and_volume_histories(tickers: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ load all prices to a single table for analysis
    :return prices table indexed with dates, column = tickers, i.e. 'GOOG', 'AMZN',...
    :return volumes table indexed with dates, column = tickers, i.e. 'GOOG', 'AMZN',...
    """
    prices_history = pd.DataFrame()
    volume_history = pd.DataFrame()
    for ticker in tqdm.tqdm(tickers, desc='Loading price and volume'):
        ticker_prices_vols = _load_price_volume_history(ticker)
        ticker_prices_df = ticker_prices_vols['Adj Close'].rename(ticker)
        ticker_vols_df = ticker_prices_vols['Volume'].rename(ticker)
        prices_history = prices_history.join(ticker_prices_df, how='outer')
        volume_history = volume_history.join(ticker_vols_df, how='outer')
    return prices_history, volume_history


def load_random_saved_tickers(sample_size: int = 300) -> list[str]:
    """Loads list of tickers from SharesOutstanding csv file"""
    return load_all_saved_tickers_info().sample(n=sample_size).index.to_list()


def load_all_saved_tickers_info() -> pd.DataFrame:
    """Load from file"""
    tickers_info_df = pd.read_csv(SHARES_INFO_FILEPATH). \
        dropna(subset=['sharesOutstanding']).\
        drop_duplicates(subset=['ticker'], keep='first').\
        set_index('ticker')
    # tickers_info_df = pd.read_csv(SHARES_OUTSTANDING_FILEPATH, index_col='ticker').\
    #     dropna(subset=['sharesOutstanding'])
    # remove_duplicate_indexes_inplace(tickers_info_df)
    return tickers_info_df


def load_shares_outstanding(tickers: set[str]) -> pd.Series:
    """
    number of shares for each ticker
    :param tickers: list of tickers to load
    :return: pd.Series({'AMZN':10000.0, 'AAPL':12323000.0})
    """
    tickers_df = load_all_saved_tickers_info()
    selected_tickers = tickers_df.loc[tickers_df.index.isin(tickers), 'sharesOutstanding']
    # warn that some ticker are not loaded
    not_laoded_tickers = [ticker for ticker in tickers if not (ticker in tickers_df.index)]
    if not_laoded_tickers:
        warnings.warn(f'ignored tickers: {not_laoded_tickers}')
    return selected_tickers


def load_shares_history(tickers: set[str]) -> SharesHistory:
    """
    :param tickers:
    :returns: price history, volumes history, outstanding shares
    """
    shares_outstanding: pd.Series = load_shares_outstanding(tickers)
    prices_history_df, volumes_history_df = load_price_and_volume_histories(set(shares_outstanding.index.tolist()))
    return SharesHistory(prices_history_df, volumes_history_df, shares_outstanding)


def _download_info(tickers: list[str]) -> pd.DataFrame:
    """ downloads info for each ticker
    :param tickers:
    :return: DataFrame with columns 'ticker', 'sharesOutstanding', etc...
    """
    info_list = ['quoteType',
                 'marketCap',
                 'sharesOutstanding',
                 'exchange',
                 'bookValue',
                 'longName',
                 'shortName',
                 'trailingPE',
                 'trailingAnnualDividendYield']
    tickers_info = dict()
    tickers_info['ticker'] = []
    for info_label in info_list:
        tickers_info[info_label] = []
    for ticker in tqdm.tqdm(tickers):
        try:
            ticker_info = yfsi.get_quote_data(ticker)
        except (IndexError, AssertionError):
            print(f'{ticker} data is not downloaded')
        else:
            tickers_info['ticker'].append(ticker)
            for info_label in info_list:
                tickers_info[info_label].append(ticker_info.get(info_label, None))
    resulting_df = pd.DataFrame(tickers_info)
    resulting_df.index.names = ['index']
    return resulting_df
