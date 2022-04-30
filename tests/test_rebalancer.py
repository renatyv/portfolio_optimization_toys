import numpy as np
import pandas as pd

from pypoanal import assets
from pypoanal import portfolio_rebalancer


def test_portfolio_value_empty():
    prices = pd.Series({'GOOG': 2000.0, 'AMZN': 3000.0})
    portfolio = pd.Series(dtype=np.float64)
    assert assets.shares_value(portfolio, prices) == 0


def test_portfolio_value_1():
    prices = pd.Series({'GOOG': 2000.0, 'AMZN': 3000.0})
    portfolio = pd.Series({'GOOG': 2.0, 'AMZN': 2.0})
    assert assets.shares_value(portfolio, prices) == 2 * 2000 + 2 * 3000
    prices = pd.Series({'GOOG': 2000.0, 'AMZN': 3000.0})
    portfolio = pd.Series({'GOOG': 2.0})
    assert assets.shares_value(portfolio, prices) == 2 * 2000


def test_portfolio_value_2():
    prices = pd.Series({'GOOG': 2000.0, 'AMZN': 3000.0})
    portfolio = pd.Series({'GOOG': 2.0})
    assert assets.shares_value(portfolio, prices) == 2 * 2000


def test_reallocate_from_empty_portfolio_1():
    old_portfolio = assets.Portfolio(4000.0, pd.Series(dtype=np.float64))
    portfolio_weights = pd.Series({'GOOG': 0.5, 'AMZN': 0.5})
    latest_prices = pd.Series({'GOOG': 2000.0, 'AMZN': 2000.0})
    new_portfolio, accumulated_fees = portfolio_rebalancer.reallocate_portfolio(old_portfolio,
                                                                                portfolio_weights,
                                                                                latest_prices,
                                                                                fees_percent=0.0)
    assert accumulated_fees == 0
    assert new_portfolio.cash == 0
    assert new_portfolio.shares.to_dict() == {'GOOG': 1.0, 'AMZN': 1.0}


def test_reallocate_from_empty_portfolio_2():
    old_portfolio = assets.Portfolio(5000.0, pd.Series(dtype=np.float64))
    portfolio_weights = pd.Series({'GOOG': 0.5, 'AMZN': 0.5})
    latest_prices = pd.Series({'GOOG': 2000.0, 'AMZN': 2000.0})
    fees_p = 1.0
    new_portfolio, accumulated_fees = portfolio_rebalancer.reallocate_portfolio(old_portfolio,
                                                                                portfolio_weights,
                                                                                latest_prices,
                                                                                fees_percent=fees_p)
    assert accumulated_fees == (2000 + 2000) / 100.0
    assert new_portfolio.cash == 1000 - accumulated_fees
    assert new_portfolio.shares.to_dict() == {'GOOG': 1.0, 'AMZN': 1.0}


def test_reallocate_portfolio_the_same():
    portfolio_shares = pd.Series({'GOOG': 1.0, 'AMZN': 1.0})
    old_portfolio = assets.Portfolio(100.0, portfolio_shares)
    portfolio_weights = pd.Series({'GOOG': 0.4, 'AMZN': 0.6})
    latest_prices = pd.Series({'GOOG': 1000.0, 'AMZN': 2000.0})
    new_portfolio, fees = portfolio_rebalancer.reallocate_portfolio(old_portfolio,
                                                                    portfolio_weights,
                                                                    latest_prices,
                                                                    fees_percent=1)
    assert new_portfolio.shares.to_dict() == {'GOOG': 1.0, 'AMZN': 1.0}
    assert fees == 0.0
    assert new_portfolio.cash == 100.0 - fees


def test_reallocate_portfolio_1():
    cash = 100.0
    old_portfolio = assets.Portfolio(cash, pd.Series({'GOOG': 1.0, 'AMZN': 1.0}))
    portfolio_weights = pd.Series({'GOOG': 0.4, 'AMZN': 0.6})
    latest_prices = pd.Series({'GOOG': 1000.0, 'AMZN': 2000.0})
    new_portfolio, fees = portfolio_rebalancer.reallocate_portfolio(old_portfolio,
                                                                    portfolio_weights,
                                                                    latest_prices,
                                                                    fees_percent=1)
    assert new_portfolio.shares.to_dict() == {'GOOG': 1.0, 'AMZN': 1.0}
    assert fees == 0.0
    assert new_portfolio.cash == cash - fees


def test_reallocate_portfolio_2():
    cash = 100.0
    old_portfolio = assets.Portfolio(cash, pd.Series({'GOOG': 1.0, 'AMZN': 2.0}))
    portfolio_weights = pd.Series({'GOOG': 2 / 3, 'AMZN': 1 / 3})
    latest_prices = pd.Series({'GOOG': 1000.0, 'AMZN': 1000.0})
    new_portfolio, fees = portfolio_rebalancer.reallocate_portfolio(old_portfolio,
                                                                    portfolio_weights,
                                                                    latest_prices,
                                                                    fees_percent=1)
    assert new_portfolio.shares.to_dict() == {'GOOG': 2.0, 'AMZN': 1.0}
    assert fees == 20.0
    assert new_portfolio.cash == cash - fees


def test_compute_rebalance_fees_equal():
    shares = pd.Series({'GOOG': 1.0, 'AMZN': 1.0})
    prices = pd.Series({'GOOG': 100.0, 'AMZN': 200.0})
    fees = portfolio_rebalancer._compute_fees_for_rebalance(shares, shares, prices, fees_percent=1.0)
    assert fees == 0


def test_compute_rebalance_fees_1():
    old_shares = pd.Series({'GOOG': 1.0, 'AMZN': 1.0})
    new_shares = pd.Series({'GOOG': 0.0, 'AMZN': 0.0})
    prices = pd.Series({'GOOG': 100.0, 'AMZN': 100.0})
    return np.float64(2.0) == portfolio_rebalancer._compute_fees_for_rebalance(old_shares, new_shares, prices,
                                                                               fees_percent=1.0)


def test_compute_rebalance_fees_2():
    old_shares = pd.Series({'GOOG': 1.0, 'AMZN': 1.0})
    new_shares = pd.Series({'AAPL': 1.0, 'GOOG': 0.0})
    prices = pd.Series({'AAPL': 100.0, 'GOOG': 100.0, 'AMZN': 100.0})
    return np.float64(2.0) == portfolio_rebalancer._compute_fees_for_rebalance(old_shares, new_shares, prices,
                                                                               fees_percent=1.0)


def test_compute_leftover_portfolios_equal():
    cash = 100.0
    old_portfolio = assets.Portfolio(cash, pd.Series({'GOOG': 1.0, 'AMZN': 1.0}))
    prices = pd.Series({'GOOG': 100.0, 'AMZN': 100.0, 'AAPL': 100.0, 'NOTATICKER': 4000})
    leftover = portfolio_rebalancer._compute_leftover_after_rebalance(old_portfolio,
                                                                      old_portfolio.shares,
                                                                      prices,
                                                                      fees_percent=0.02)
    assert leftover == cash


def test_compute_leftover_portfolios_1():
    old_portfolio = assets.Portfolio(100.0,
                                     pd.Series({'GOOG': 1.0, 'AMZN': 1.0}))
    new_shares = pd.Series({'GOOG': 1.0, 'AAPL': 1.0})
    prices = pd.Series({'GOOG': 100.0, 'AMZN': 100.0, 'AAPL': 100.0})
    leftover = portfolio_rebalancer._compute_leftover_after_rebalance(old_portfolio,
                                                                      new_shares,
                                                                      prices,
                                                                      fees_percent=1.0)
    # sell AMZN and buy AAPL, fees = 2.0
    assert leftover == 100.0 - 2.0


def test_compute_leftover_portfolios_2():
    old_portfolio = assets.Portfolio(0.0,
                                     pd.Series({'GOOG': 1.0, 'AMZN': 1.0}))
    new_shares = pd.Series({'MSFT': 1.0, 'AAPL': 1.0})
    latest_prices = pd.Series({'GOOG': 100.0, 'AMZN': 100.0, 'AAPL': 200.0, 'MSFT': 200.0})
    leftover = portfolio_rebalancer._compute_leftover_after_rebalance(old_portfolio,
                                                                      new_shares,
                                                                      latest_prices,
                                                                      fees_percent=1.0)
    fees = 2.0 + 4.0  # sell old and buy new portfolio
    assert leftover == -200.0 - fees


def test_allocate_discrete_1():
    p_weights = pd.Series({'GOOG': 0.5, 'AMZN': 0.5})
    latest_prices = pd.Series({'GOOG': 100.0, 'AMZN': 100.0})
    fees_percent = 0.0
    cash = 1000.0
    portfolio, fees = portfolio_rebalancer.allocate_discrete(p_weights, latest_prices, cash, fees_percent)
    assert portfolio.shares.to_dict() == {'GOOG': 5.0, 'AMZN': 5.0}
    assert portfolio.cash == 0.0
    assert fees == 0.0


def test_allocate_discrete_leftover():
    p_weights = pd.Series({'GOOG': 0.5, 'AMZN': 0.5})
    latest_prices = pd.Series({'GOOG': 100.0, 'AMZN': 100.0})
    fees_percent = 0.0
    cash = 205.0
    portfolio, fees = portfolio_rebalancer.allocate_discrete(p_weights, latest_prices, cash, fees_percent)
    assert portfolio.shares.to_dict() == {'GOOG': 1.0, 'AMZN': 1.0}
    assert portfolio.cash == 5.0
    assert fees == 0.0


def test_allocate_discrete_fees():
    p_weights = pd.Series({'GOOG': 0.5, 'AMZN': 0.5})
    latest_prices = pd.Series({'GOOG': 100.0, 'AMZN': 100.0})
    fees_percent = 1.0
    cash = 205
    portfolio, fees = portfolio_rebalancer.allocate_discrete(p_weights, latest_prices, cash, fees_percent)
    assert portfolio.shares.to_dict() == {'GOOG': 1.0, 'AMZN': 1.0}
    assert fees == 2.0
    assert portfolio.cash == 3.0


def test_allocate_discrete_2():
    # enough cash to buy  both shares, but fees do not allow to do it.
    p_weights = pd.Series({'GOOG': 0.49, 'AMZN': 0.51})
    latest_prices = pd.Series({'GOOG': 100.0, 'AMZN': 10.0})
    fees_percent = 1.0
    cash = 190.0
    portfolio, fees = portfolio_rebalancer.allocate_discrete(p_weights, latest_prices, cash, fees_percent)
    assert portfolio.shares.to_dict() == {'GOOG': 1.0, 'AMZN': 8.0}
    assert fees == 1.8
    assert portfolio.cash == 10.0 - 1.8
