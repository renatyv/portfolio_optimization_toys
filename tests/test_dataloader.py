import os
import pandas as pd
from pypoanal import dataloader


def test_download_price_vol():
    apple_price_vol = dataloader._download_price_volume_history('AAPL')
    assert 'Adj Close' in apple_price_vol
    assert 'Volume' in apple_price_vol


def test_download_and_save_prices():
    tickers = ['AAPL', 'GOOG']
    dataloader.download_and_save_price_history(tickers)
    # check if downloaded files exist
    for ticker in tickers:
        path = dataloader._price_vol_path(ticker)
        assert os.path.exists(path)
    prev_mod_time = os.path.getmtime(dataloader._price_vol_path('AAPL'))
    # re-download files and check that files are updated
    dataloader.download_and_save_price_history(tickers)
    for ticker in tickers:
        path = dataloader._price_vol_path(ticker)
        assert prev_mod_time < os.path.getmtime(path)
    # check file sizes
    assert os.path.getsize(dataloader._price_vol_path('AAPL')) > 400 * 1024
    assert os.path.getsize(dataloader._price_vol_path('GOOG')) > 150 * 1024
    # check columns
    apple_price_vol = pd.read_csv(dataloader._price_vol_path('AAPL'))
    assert 'Adj Close' in apple_price_vol
    assert 'Volume' in apple_price_vol


def test_get_price_vol_data():
    AAPL_price_vol_df = dataloader._load_price_volume_history('AAPL')
    assert 'Adj Close' in AAPL_price_vol_df
    assert 'Volume' in AAPL_price_vol_df


def test_download_info_1():
    df = dataloader._download_info(['AAPL'])
    assert len(df[df['ticker'] == 'AAPL']) > 0


def test_download_info_2():
    df = dataloader._download_info(['AAPL'])
    assert df.loc[df['ticker'] == 'AAPL'].loc[:, 'sharesOutstanding'].values[0] > 100


def test_load_shares_outstanding():
    shares_ser = dataloader.load_shares_outstanding(['AAPL', 'GOOG', 'F'])
    assert len(shares_ser) == 3
