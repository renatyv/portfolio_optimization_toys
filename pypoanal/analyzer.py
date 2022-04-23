import math
import warnings
import datetime
import pandas as pd
import numpy as np
from cvxpy import SolverError
from pypfopt import risk_models
from pypfopt.exceptions import OptimizationError
from scipy.sparse.linalg import ArpackNoConvergence

from pypoanal import assets
import pypoanal.portfolio_calculators as pcalc
from pypoanal import dataloader
import tqdm


def choose_liquid_tickers(volumes_history: pd.DataFrame,
                          start_date: datetime.date,
                          end_date: datetime.date,
                          min_volume=50,
                          liquid_days_percent=90) -> list[str]:
    """
    | returns list of tickers, for whom:
     more than min_volume shares were traded for liquid_days_percent % of days in the specified period
     :param liquid_days_percent: minimal percent of days when ticker was trading
     :param min_volume: USD volume
     :param volumes_history: trading volumes. DataFrame indexed with trading dates. Each column corresponds to a ticker
     """
    trading_days_in_period = len(volumes_history[start_date:end_date])
    required_liquid_trading_days = trading_days_in_period * liquid_days_percent / 100.0
    volumes_sample = volumes_history.loc[start_date:end_date, :]
    # for each ticker compute number of days for which volume was higher that MIN_VOLUME
    liquid_trading_days = volumes_sample.apply(lambda x: len(x.loc[x > min_volume]), axis=0)
    # choose tickers for which number of liquid days is more than required
    liquid_tickers = liquid_trading_days[liquid_trading_days > required_liquid_trading_days].index.tolist()
    return liquid_tickers


def shares_weights_performance(weights: assets.SharesWeights,
                               price_history: pd.DataFrame,
                               frequency=252) -> tuple[np.float64, np.float64]:
    """
    Sigma volatility is computed using Covariance matrix using LW shrinkage.
    return rate is the return rate for
    1) buying the portfolio on the first date of prices DataFrame
    2) selling on the last day of prices DataFrame
    :param frequency: trading days per period (default 252 days, per year)
    :param weights: portfolio weights
    :param price_history: table indexed with dates, column of prices for tickets
    :returns sigma:, computed Covariance matrix using LW shrinkage
    :returns return rate: return rate.
    """
    price_history_reduced = price_history[weights.index]
    weights_array: np.ndarray = weights.values
    cov_matrix: np.ndarray = risk_models.CovarianceShrinkage(price_history_reduced).ledoit_wolf()
    sigma = np.sqrt(weights_array.transpose() @ cov_matrix @ weights_array * frequency)
    initial_prices: np.ndarray = price_history_reduced.fillna(method='bfill').iloc[0]
    final_prices: np.ndarray = price_history_reduced.fillna(method='ffill').iloc[-1]
    initial_shares_value = initial_prices @ weights_array
    final_shares_value = final_prices @ weights_array
    return_rate = (final_shares_value -
                   initial_shares_value) / initial_shares_value
    return sigma, return_rate
