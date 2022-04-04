# python notebooks and small library for basic analysis of stocks

## Goals:
* Compare methods for portfolio allocation:
  * Markovitz optimal portfolio (max Sharpie portfolio)
  * Using Ledoit Wolf caovariance shrinkage
  * Minimal volatility portfolio
  * Market capitalization-weighted portfolio
  * Hierarchical Risk Parity

## Downloading data
Package **yahoo_fin** is used to download list of tickers listed for S&P500, NYSE, and NASDAQ, see

```download_ticker_data.ipynb```

This jupyter notebook is used to download and save tickers along with some additional info (outstanding shares, full name, etc...) in .csv files in folder

```info/```




