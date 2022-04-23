import warnings
from typing import Callable

import numpy as np
import pandas as pd
from pypfopt import risk_models, EfficientFrontier, expected_returns, HRPOpt

from pypoanal.assets import SharesWeights

SharesOutstanding = pd.Series
PriceHistory = pd.DataFrame
# function shares_outstanding, price_history -> SharesWeights
PortfolioWeightsCalculator = Callable[[SharesOutstanding, PriceHistory], SharesWeights]


def sharpie_weights(shares_outstanding: pd.Series,
                    price_history: pd.DataFrame) -> SharesWeights:
    """maximaize beta=return/volatility using LedoitWolf covariance shrinkage"""
    cov_matrix = risk_models.CovarianceShrinkage(price_history, frequency=252).ledoit_wolf()
    mu = expected_returns.capm_return(price_history)
    optimizer = EfficientFrontier(mu, cov_matrix)
    # compute efficient frontier
    optimizer.max_sharpe(risk_free_rate=0.02)
    portfolio_weights = pd.Series(dict(optimizer.clean_weights()))
    return portfolio_weights


def equal_weights(shares_outstanding: pd.Series,
                  price_history: pd.DataFrame) -> SharesWeights:
    """Equal weight 1/N for each share, where N = number of columns in price_history"""
    n = len(price_history.columns)
    return pd.Series(np.ones(n) / n, index=price_history.columns)


def mcap_weights(shares_outstanding: pd.Series,
                 price_history: pd.DataFrame) -> SharesWeights:
    """Allocate according to the market capitalization at the end of the period"""
    if shares_outstanding.empty > 0:
        warnings.warn(f'ticker_nshares is empty')
    latest_prices: pd.Series = price_history.fillna(method='ffill').iloc[-1]
    total_mcap = (latest_prices * shares_outstanding).sum()
    if total_mcap == 0:
        return pd.Series()
    weights = (latest_prices * shares_outstanding).dropna() / total_mcap
    return weights[weights > 0]


def ledoitw_weights(shares_outstanding: pd.Series,
                    price_history: pd.DataFrame) -> SharesWeights:
    """Minimal volatility using Ledoit-Wolf covariance matrix shrinkage"""
    cov_matrix = risk_models.CovarianceShrinkage(price_history, frequency=252).ledoit_wolf()
    optimizer = EfficientFrontier(None, cov_matrix)
    # compute efficient frontier
    optimizer.min_volatility()
    portfolio_weights = pd.Series(dict(optimizer.clean_weights()))
    return portfolio_weights


def expcov_weights(shares_outstanding: pd.Series,
                   price_history: pd.DataFrame) -> SharesWeights:
    """Weights giving minimal volatility using exponential covariance matrix"""
    cov_matrix = risk_models.exp_cov(price_history, span=179)
    optimizer = EfficientFrontier(None, cov_matrix)
    # compute efficient frontier
    optimizer.min_volatility()
    portfolio_weights = pd.Series(dict(optimizer.clean_weights()))
    return portfolio_weights


def hrp_weights(shares_outstanding: pd.Series,
                price_history: pd.DataFrame) -> SharesWeights:
    """Hierarchical risk parity to minimize volatility"""
    returns = expected_returns.returns_from_prices(price_history)
    optimizer = HRPOpt(returns)
    optimizer.optimize()
    portfolio_weights = pd.Series(dict(optimizer.clean_weights()))
    return portfolio_weights


CALCULATORS: dict[str, PortfolioWeightsCalculator] = {
    'max_sharpe': sharpie_weights,
    'HRP': hrp_weights,
    'exp_cov': expcov_weights,
    'equal': equal_weights,
    'MCAP': mcap_weights,
    'ledoitw_cov': ledoitw_weights
}
