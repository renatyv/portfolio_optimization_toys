import numpy as np

# from pypoanal import portfolio_calculators as pcalc
from pypoanal import assets
from pypoanal.assets import shares_value, Portfolio
from pypoanal.assets import SharesWeights
from pypoanal.assets import SharesNumber
from typing import Callable
import pandas as pd


def clean_weights(portfolio_weights: SharesWeights, eps=0.0001) -> SharesWeights:
    return portfolio_weights[portfolio_weights > eps]


def compute_rebalance_fees(old_shares: SharesNumber,
                           new_shares: SharesNumber,
                           prices: pd.Series,
                           fees_percent: float) -> np.float64:
    """
    :param old_shares: pd.Series({'AMZN':12, 'GOOG':100})
    :param new_shares: pd.Series({'AMZN':12, 'GOOG':100})
    :param prices: pd.Series({'AMZN':12.3, 'GOOG':100.2})
    :param fees_percent: 0.04
    :return: 0.1
    """
    return (old_shares.subtract(new_shares, fill_value=0.0).abs() * prices).sum() * fees_percent / 100.0


def compute_rebalance_leftover(old_portfolio: Portfolio,
                               rebalanced_shares: SharesNumber,
                               latest_prices: pd.Series,
                               fees_percent: float) -> np.float64:
    """
    leftover after rebalance
    :param latest_prices: pd.Series({'AMZN':12.3, 'GOOG':100.2})
    :param rebalanced_shares: pd.Series({'AMZN':12, 'GOOG':100})
    :param old_portfolio: (102.3, pd.Series({'AMZN':12, 'GOOG':100}))
    :param fees_percent: 0.04
    :return: 120.1
    """
    new_portfolio_value = shares_value(rebalanced_shares, latest_prices)
    old_portfolio_value = old_portfolio.value(latest_prices)
    fees = compute_rebalance_fees(old_portfolio.shares, rebalanced_shares, latest_prices, fees_percent)
    return old_portfolio_value - new_portfolio_value - fees


def reduce_portfolio_until_leftover_positive(shares: SharesNumber,
                                             latest_prices: pd.Series,
                                             leftover_fn: Callable[[SharesNumber], np.float64],
                                             fees_percent: float) -> SharesNumber:
    """
    if fees are too big, reduce amount of shares, starting from less weighty one
    :param shares: pd.Series({'AMZN':100, 'GOOG':100})
    :param target_weights: pd.Series({'AMZN':0.6, 'GOOG':0.4})
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
            num_tickers = min(np.ceil(-leftover/latest_prices[ticker]),
                              only_positive_shares[ticker])
            only_positive_shares[ticker] -= num_tickers
            portfolio_value_change = num_tickers*latest_prices[ticker]
            # sold some shares
            leftover += portfolio_value_change
            # additional fees, worst case estimate
            leftover -= portfolio_value_change*fees_percent/100.0
        else:
            break
    return only_positive_shares


def reallocate_portfolio(old_portfolio: Portfolio,
                         new_portfolio_weights: SharesWeights,
                         latest_prices: pd.Series,
                         fees_percent=0.04) -> tuple[Portfolio, np.float64]:
    """allocate discrete amount of shares
    :param old_portfolio:
    :param new_portfolio_weights: {'GOOG': 0.5, 'AMZN': 0.2, 'MSFT': 0.3}
    :param latest_prices: shares prices
    :param initial_cash: money available, not taking into account value of the old_portfolio
    :return (allocated equity, fees)
    """
    portfolio_value = old_portfolio.value(latest_prices)
    # portfolio_estimate: PortfolioShares = np.ceil(new_portfolio_weights * total_value / latest_prices)
    portfolio_estimate: SharesNumber = np.round(new_portfolio_weights * portfolio_value / latest_prices)
    portfolio_estimate = clean_weights(portfolio_estimate)
    leftover_fn = lambda portfolio: compute_rebalance_leftover(old_portfolio, portfolio, latest_prices, fees_percent)
    reduced_portfolio = reduce_portfolio_until_leftover_positive(portfolio_estimate,
                                                                 latest_prices,
                                                                 leftover_fn,
                                                                 fees_percent)
    leftover = leftover_fn(reduced_portfolio)
    fees = compute_rebalance_fees(old_portfolio.shares, reduced_portfolio, latest_prices, fees_percent)
    assert fees >= 0
    assert leftover >= 0
    return Portfolio(cash=leftover, shares=clean_weights(reduced_portfolio)), fees


def allocate_discrete(portfolio_weights: SharesWeights,
                      latest_prices: pd.Series,
                      cash: np.float64,
                      fees_percent=0.04) -> tuple[assets.Portfolio, np.float64]:
    """
    allocates portfolio with integer values
    :param potfolio_weights:
    :param latest_prices:
    :param cash:
    :param fees_percent:
    :return: Portfolio, fees
    """
    return reallocate_portfolio(old_portfolio=assets.Portfolio(cash, pd.Series()),
                                new_portfolio_weights=portfolio_weights,
                                latest_prices=latest_prices,
                                fees_percent=fees_percent)
