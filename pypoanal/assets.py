from dataclasses import dataclass

import numpy as np
import pandas as pd

#  must be positive, short selling is not supported
SharesWeights = pd.Series  # pd.Series([0.2,0.3,0.5], index=['AMZN','GOOG','AAPL'])
#  must be positive, short selling is not supported
SharesNumber = pd.Series  # pd.Series([1,12,3], index=['AMZN','GOOG','AAPL']), number of shares


def shares_value(shares: SharesNumber, shares_prices: pd.Series) -> np.float64:
    """
    returns dot product of shares number and shares prices
    :param shares_prices: pd.Series({'AMZN':200.0, 'GOOG':100.0})
    :param shares: pd.Series({'AMZN':1.0, 'GOOG':2.0})
    :return: 400.0
    """
    return (shares_prices * shares).sum()


@dataclass
class Portfolio:
    cash: np.float64 = np.float64(0.0)
    shares: SharesNumber = pd.Series()

    def value(self, shares_prices: pd.Series) -> np.float64:
        """
        :param shares_prices: pd.Series({'AMZN':200.0, 'GOOG':100.0})
        :return: 500.0
        """
        return self.cash + shares_value(self.shares, shares_prices)
