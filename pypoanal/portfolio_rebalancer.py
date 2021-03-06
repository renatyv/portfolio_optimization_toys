import numpy as np

from pypoanal.assets import shares_value, Portfolio, SharesWeights, SharesNumber
from typing import Callable
import pandas as pd
from loguru import logger


def _clean_weights(portfolio_weights: SharesWeights, eps=0.0001) -> SharesWeights:
    return portfolio_weights[portfolio_weights > eps]


def _compute_fees_for_rebalance(old_shares: SharesNumber,
                                new_shares: SharesNumber,
                                prices: pd.Series,
                                fees_percent: float) -> np.float64:
    """
    Computes fees paid as a result of portfolio rebalance
    :param old_shares: pd.Series({'AMZN':12, 'GOOG':100})
    :param new_shares: pd.Series({'AMZN':12, 'GOOG':100})
    :param prices: pd.Series({'AMZN':12.3, 'GOOG':100.2})
    :param fees_percent: 0.04
    :return: 0.1
    """
    # pandas is aware of indexes.
    # It will compute old_shares.subtract and '*' index-by-index, ignoring the actual order
    return (old_shares.subtract(new_shares, fill_value=0.0).abs() * prices).sum() * fees_percent / 100.0


def _compute_leftover_after_rebalance(old_portfolio: Portfolio,
                                      rebalanced_shares: SharesNumber,
                                      latest_prices: pd.Series,
                                      fees_percent: np.float64) -> np.float64:
    """
    Computes leftover paid as a result of portfolio rebalance
    leftover after rebalance
    :param latest_prices: pd.Series({'AMZN':12.3, 'GOOG':100.2})
    :param rebalanced_shares: pd.Series({'AMZN':12, 'GOOG':100})
    :param old_portfolio: (102.3, pd.Series({'AMZN':12, 'GOOG':100}))
    :param fees_percent: 0.04
    :return: 120.1
    """
    new_portfolio_value = shares_value(rebalanced_shares, latest_prices)
    old_portfolio_value: np.float64 = old_portfolio.value(latest_prices)
    fees = _compute_fees_for_rebalance(old_portfolio.shares, rebalanced_shares, latest_prices, fees_percent)
    return old_portfolio_value - new_portfolio_value - fees


def _reduce_portfolio_until_leftover_positive(shares: SharesNumber,
                                              latest_prices: pd.Series,
                                              leftover_fn: Callable[[SharesNumber], np.float64],
                                              fees_percent: float) -> SharesNumber:
    """
    if fees are too big, reduce amount of shares, starting from less weighty one
    :param shares: pd.Series({'AMZN':100, 'GOOG':100})
    :param leftover_fn: returns leftover cash after rebalance
    :param latest_prices: pd.Series({'AMZN':1.2, 'GOOG':1.2})
    :return:
    """
    if leftover_fn(shares) >= 0:
        return shares
    only_positive_shares = shares[shares > 0]
    price_sorted_tickers: pd.Series = only_positive_shares.sort_values(key=lambda ser: latest_prices[ser.index],
                                                                       ascending=True)
    leftover = leftover_fn(only_positive_shares)
    for ticker in price_sorted_tickers.index:
        if leftover < 0:
            num_tickers = min(np.ceil(-leftover / latest_prices[ticker]),
                              only_positive_shares[ticker])
            only_positive_shares[ticker] -= num_tickers
            portfolio_value_change = num_tickers * latest_prices[ticker]
            # sold some shares
            leftover += portfolio_value_change
            # additional fees, worst case estimate
            leftover -= portfolio_value_change * fees_percent / 100.0
        else:
            break
    return only_positive_shares


class AllocationException(Exception):
    pass


def reallocate_portfolio(old_portfolio: Portfolio,
                         new_portfolio_weights: SharesWeights,
                         latest_prices: pd.Series,
                         fees_percent=0.04) -> tuple[Portfolio, np.float64]:
    """allocate discrete amount of shares
    :param old_portfolio: portfolio before re-allocation
    :param new_portfolio_weights: {'GOOG': 0.5, 'AMZN': 0.2, 'MSFT': 0.3}
    :param latest_prices: shares prices
    :return (allocated equity, fees)
    """
    portfolio_value = old_portfolio.value(latest_prices)
    # portfolio_estimate: PortfolioShares = np.ceil(new_portfolio_weights * total_value / latest_prices)
    portfolio_estimate: SharesNumber = np.round(new_portfolio_weights * portfolio_value / latest_prices)
    portfolio_estimate = _clean_weights(portfolio_estimate)
    leftover_fn = lambda portfolio: _compute_leftover_after_rebalance(old_portfolio,
                                                                      portfolio,
                                                                      latest_prices,
                                                                      fees_percent)
    reduced_portfolio = _reduce_portfolio_until_leftover_positive(portfolio_estimate,
                                                                  latest_prices,
                                                                  leftover_fn,
                                                                  fees_percent)
    leftover = leftover_fn(reduced_portfolio)
    fees = _compute_fees_for_rebalance(old_portfolio.shares,
                                       reduced_portfolio,
                                       latest_prices,
                                       fees_percent)
    if fees < 0 or leftover < 0:
        logger.error('fees are negative')
        logger.debug(f'fees: {fees} leftover: {leftover}')
        logger.debug(f'Old portolio: {old_portfolio.shares.to_list()}')
        logger.debug(f'New portolio: {new_portfolio_weights.to_list()}')
        logger.debug(f'Prices: {latest_prices.to_list()}')
        raise AllocationException('Smth gone wrong')
    return Portfolio(cash=leftover, shares=_clean_weights(reduced_portfolio)), fees


def allocate_discrete(portfolio_weights: SharesWeights,
                      latest_prices: pd.Series,
                      cash: np.float64,
                      fees_percent=0.04) -> tuple[Portfolio, np.float64]:
    """
    allocates portfolio with integer values
    :return: Portfolio, fees
    """
    return reallocate_portfolio(old_portfolio=Portfolio(cash, pd.Series(dtype=np.float64)),
                                new_portfolio_weights=portfolio_weights,
                                latest_prices=latest_prices,
                                fees_percent=fees_percent)
