from typing import Callable

import numpy as np
import pandas as pd
from pypfopt import risk_models, EfficientFrontier, expected_returns, HRPOpt

from pypoanal.assets import SharesWeights

SharesOutstanding = pd.Series
PriceHistory = pd.DataFrame
# function shares_outstanding, price_history -> SharesWeights
PortfolioWeightsCalculator = Callable[[SharesOutstanding, PriceHistory], SharesWeights]


def compute_sharpie_weights(shares_outstanding: pd.Series,
                            price_history: pd.DataFrame) -> SharesWeights:
    """maximaize beta=return/volatility using LedoitWolf covariance shrinkage"""
    cov_matrix = risk_models.CovarianceShrinkage(price_history, frequency=252).ledoit_wolf()
    mu = expected_returns.capm_return(price_history)
    optimizer = EfficientFrontier(mu, cov_matrix)
    # compute efficient frontier
    optimizer.max_sharpe(risk_free_rate=0.02)
    portfolio_weights = pd.Series(dict(optimizer.clean_weights()))
    return portfolio_weights


def compute_equal_weights(shares_outstanding: pd.Series,
                          price_history: pd.DataFrame) -> SharesWeights:
    """Equal weight 1/N for each share, where N = number of columns in price_history"""
    n = len(price_history.columns)
    return pd.Series(np.ones(n) / n, index=price_history.columns)


def compute_mcap_weights(shares_outstanding: pd.Series,
                         price_history: pd.DataFrame) -> SharesWeights:
    """Allocate according to the market capitalization at the end of the period"""
    latest_prices: pd.Series = price_history.fillna(method='ffill').iloc[-1]
    total_mcap = (latest_prices * shares_outstanding).sum()
    if total_mcap == 0:
        return pd.Series(dtype=np.float64)
    weights = (latest_prices * shares_outstanding).dropna() / total_mcap
    return weights[weights > 0]


def compute_ledoitw_weights(shares_outstanding: pd.Series,
                            price_history: pd.DataFrame) -> SharesWeights:
    """Minimal volatility using Ledoit-Wolf covariance matrix shrinkage"""
    cov_matrix = risk_models.CovarianceShrinkage(price_history, frequency=252).ledoit_wolf()
    optimizer = EfficientFrontier(None, cov_matrix)
    # compute efficient frontier
    optimizer.min_volatility()
    portfolio_weights = pd.Series(dict(optimizer.clean_weights()))
    return portfolio_weights


def compute_expcov_weights(shares_outstanding: pd.Series,
                           price_history: pd.DataFrame) -> SharesWeights:
    """Weights giving minimal volatility using exponential covariance matrix"""
    cov_matrix = risk_models.exp_cov(price_history, span=179)
    optimizer = EfficientFrontier(None, cov_matrix)
    # compute efficient frontier
    optimizer.min_volatility()
    portfolio_weights = pd.Series(dict(optimizer.clean_weights()))
    return portfolio_weights


def compute_hrp_weights(shares_outstanding: pd.Series,
                        price_history: pd.DataFrame) -> SharesWeights:
    """Hierarchical risk parity to minimize volatility"""
    returns = expected_returns.returns_from_prices(price_history)
    optimizer = HRPOpt(returns)
    optimizer.optimize()
    portfolio_weights = pd.Series(dict(optimizer.clean_weights()))
    return portfolio_weights


CALCULATORS: dict[str, PortfolioWeightsCalculator] = {
    'max_sharpe': compute_sharpie_weights,
    'HRP': compute_hrp_weights,
    'exp_cov': compute_expcov_weights,
    'equal': compute_equal_weights,
    'MCAP': compute_mcap_weights,
    'ledoitw_cov': compute_ledoitw_weights
}
