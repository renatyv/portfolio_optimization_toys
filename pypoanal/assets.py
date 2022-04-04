from collections import namedtuple

import numpy as np
import pandas as pd

#  must be positive, short selling is not supported
SharesWeights = pd.Series  # pd.Series([0.2,0.3,0.5], index=['AMZN','GOOG','AAPL'])
#  must be positive, short selling is not supported
SharesNumber = SharesWeights  # pd.Series([1,12,3], index=['AMZN','GOOG','AAPL']), number of shares


# should probably use dataclass instead
Portfolio = namedtuple('Portfolio', ['cash', 'shares'])


def shares_value(shares: SharesNumber, shares_prices: pd.Series) -> np.float64:
    """
    :param shares_prices: pd.Series({'AMZN':200.0, 'GOOG':100.0})
    :param shares: pd.Series({'AMZN':1.0, 'GOOG':2.0})
    :return: 400.0
    """
    return (shares_prices * shares).sum()


def portfolio_value(portfolio: Portfolio, shares_prices: pd.Series) -> np.float64:
    """
    :param portfolio: (100.0, pd.Series({'AMZN':1.0, 'GOOG':2.0}))
    :param shares_prices: pd.Series({'AMZN':200.0, 'GOOG':100.0})
    :return: 500.0
    """
    return portfolio.cash + shares_value(portfolio.shares, shares_prices)
