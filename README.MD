# python notebooks and small library for basic analysis of stocks

## Goals:
* Compare methods for portfolio allocation:
  * Markovitz optimal portfolio (max Sharpie portfolio)
  * Using Ledoit Wolf caovariance shrinkage
  * Minimal volatility portfolio
  * Market capitalization-weighted portfolio
  * Hierarchical Risk Parity

## Downloading tikcers along with their info
Package **yahoo_fin** is used to download list of tickers listed for S&P500, NYSE, and NASDAQ, see

```download_ticker_info.ipynb```

Download ticker price and volume history

```download_price_history.ipynb```

data is saved into ```info/```

## Downloading price and volume historical data

```analyze_portfolios.ipynb```

Historical data for each ticker is saved into ```priceVolData/``` directory 



