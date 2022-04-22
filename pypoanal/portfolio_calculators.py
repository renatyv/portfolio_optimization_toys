import abc
import warnings
import numpy as np
import pandas as pd
from pypfopt import risk_models, EfficientFrontier, expected_returns, HRPOpt, DiscreteAllocation

from pypoanal.assets import SharesWeights



class PortfolioWeightsCalculator:
    @staticmethod
    @abc.abstractmethod
    def get_weights(shares_outstanding: pd.Series,
                    price_history: pd.DataFrame) -> SharesWeights:
        """
        returns weights for each ticker
         :param shares_outstanding: number of shares for each ticker
         :param price_history: table indexed with Date,
         columns 'AMZN', 'GOOG' and values - adjusted price
         """
        pass


class MaxSharpeCalculator(PortfolioWeightsCalculator):
    """ mazimaize beta using LedoitWolf """

    def __str__(self):
        return 'max_sharpe'

    @staticmethod
    def get_weights(shares_outstanding: pd.Series,
                    price_history: pd.DataFrame) -> SharesWeights:
        cov_matrix = risk_models.CovarianceShrinkage(price_history, frequency=252).ledoit_wolf()
        mu = expected_returns.capm_return(price_history)
        optimizer = EfficientFrontier(mu, cov_matrix)
        # compute efficient frontier
        optimizer.max_sharpe(risk_free_rate=0.02)
        portfolio_weights = pd.Series(dict(optimizer.clean_weights()))
        return portfolio_weights


class EqualWeightsCalculator(PortfolioWeightsCalculator):
    """ equal weights 1/N, where N = number of columns in price_history"""

    def __str__(self):
        return 'equal'

    @staticmethod
    def get_weights(shares_outstanding: pd.Series,
                    price_history: pd.DataFrame) -> SharesWeights:
        n = len(price_history.columns)
        return pd.Series(np.ones(n) / n, index=price_history.columns)


class MCAPWeightsCalculator(PortfolioWeightsCalculator):
    """ allocate according to the market capitalization at the end of the period"""

    def __str__(self):
        return 'MCAP'

    @staticmethod
    def get_weights(shares_outstanding: pd.Series,
                    price_history: pd.DataFrame) -> SharesWeights:
        if shares_outstanding.empty > 0:
            warnings.warn(f'ticker_nshares is empty')
        latest_prices: pd.Series = price_history.fillna(method='ffill').iloc[-1]
        total_mcap = (latest_prices * shares_outstanding).sum()
        if total_mcap == 0:
            return pd.Series()
        weights = (latest_prices * shares_outstanding).dropna() / total_mcap
        return weights[weights > 0]


class LedoitWolfWeightsCalculator(PortfolioWeightsCalculator):
    """ minimal volatility using Ledoit-Wolf covariance matrix shrinkage"""

    def __str__(self):
        return 'ledoitw_cov'

    @staticmethod
    def get_weights(shares_outstanding: pd.Series,
                    price_history: pd.DataFrame) -> SharesWeights:
        cov_matrix = risk_models.CovarianceShrinkage(price_history, frequency=252).ledoit_wolf()
        optimizer = EfficientFrontier(None, cov_matrix)
        # compute efficient frontier
        optimizer.min_volatility()
        portfolio_weights = pd.Series(dict(optimizer.clean_weights()))
        return portfolio_weights


class ExpCovWeightsCalculator(PortfolioWeightsCalculator):
    """ minimal volatility using exponential covariance matrix"""

    def __str__(self):
        return 'exp_cov'

    @staticmethod
    def get_weights(shares_outstanding: pd.Series,
                    price_history: pd.DataFrame) -> SharesWeights:
        cov_matrix = risk_models.exp_cov(price_history, span=179)
        optimizer = EfficientFrontier(None, cov_matrix)
        # compute efficient frontier
        optimizer.min_volatility()
        portfolio_weights = pd.Series(dict(optimizer.clean_weights()))
        return portfolio_weights


class HRPWeightsCalculator(PortfolioWeightsCalculator):
    """ Hierarchical risk parity"""

    def __str__(self):
        return 'HRP'

    @staticmethod
    def get_weights(shares_outstanding: pd.Series,
                    price_history: pd.DataFrame) -> SharesWeights:
        returns = expected_returns.returns_from_prices(price_history)
        optimizer = HRPOpt(returns)
        optimizer.optimize()
        portfolio_weights = pd.Series(dict(optimizer.clean_weights()))
        return portfolio_weights
