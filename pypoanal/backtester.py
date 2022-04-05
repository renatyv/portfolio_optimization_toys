import datetime
import math
import warnings

from cvxpy import SolverError
from pypfopt.exceptions import OptimizationError
from scipy.sparse.linalg import ArpackNoConvergence
import numpy as np
import pandas as pd
import tqdm

from pypoanal import analyzer, rebalancer, dataloader, assets
import pypoanal.portfolio_calculators as pcalc
from pypoanal.assets import Portfolio
from pypoanal.dataloader import SharesHistory


def reallocate_portfolio_periodically(shares_weights_calculator: pcalc.PortfolioWeightsCalculator,
                                      rebalance_dates: list[datetime.date],
                                      initial_money: np.float64,
                                      fees_percent: np.float64,
                                      shares_history: SharesHistory,
                                      progress_bar=True) -> tuple[list[assets.Portfolio], list[np.float64]]:
    initial_portfolio = Portfolio(cash=initial_money)
    portfolio_history = [initial_portfolio]
    fees_history = [np.float64(0.0)]
    # parse price, volume, outstanding
    price_history, volume_history, shares_outstanding = shares_history
    # price history without NANs
    forward_filled_price_history = price_history.fillna(method='ffill')
    # backtest
    for sample_start_date, sample_end_date in zip(
            tqdm.tqdm(rebalance_dates[:-1], disable=not progress_bar, desc=str(shares_weights_calculator)),
            rebalance_dates[1:]):  # tqdm(zip(,)) does not draw progress bar
        # load previous values
        old_portfolio = portfolio_history[-1]
        # slice and prepare data
        liquid_tickers: list[str] = analyzer.choose_liquid_tickers(volume_history, sample_start_date,
                                                                   sample_end_date)
        prices_sample_df = price_history.loc[sample_start_date:sample_end_date, liquid_tickers]
        prices_at_sample_end = forward_filled_price_history[:sample_end_date].iloc[-1]
        #     rebalance
        try:
            allocated_shares_weights = shares_weights_calculator.get_weights(shares_outstanding, prices_sample_df)
        except (SolverError, OptimizationError, ArpackNoConvergence, ValueError) as error:
            warnings.warn(f'{error} \n using old portfolio')
            allocated_portfolio = old_portfolio
            fees = 0
        else:
            allocated_portfolio, fees = rebalancer.reallocate_portfolio(old_portfolio,
                                                                        allocated_shares_weights,
                                                                        prices_at_sample_end,
                                                                        fees_percent=fees_percent)
        # save
        portfolio_history.append(allocated_portfolio)
        fees_history.append(fees)
    return portfolio_history, fees_history


def compute_rebalance_dates(start_date: datetime.date,
                            end_date: datetime.date,
                            rebalance_period: datetime.timedelta):
    # precompute rebalance_dates
    num_periods = math.floor((end_date - start_date).days / rebalance_period.days)
    rebalance_dates = [start_date + rebalance_period * n for n in range(num_periods)]
    return rebalance_dates


def portfolios_values_history(portfolio_history: list[tuple[datetime.date, assets.Portfolio]],
                              price_history: pd.DataFrame) -> list[np.float64]:
    forward_filled_prices = price_history.fillna(method='ffill')
    get_current_prices = lambda date: forward_filled_prices[:date].iloc[-1]
    return [portfolio.value(get_current_prices(date)) for date, portfolio in portfolio_history]


def compare_calculators_for_periodic_rebalance(shares_weights_calculators: list[pcalc.PortfolioWeightsCalculator],
                                               tickers: list[str],
                                               initial_cash: np.float64,
                                               rebalance_dates: list[datetime.date],
                                               fees_percent=np.float64(0.04),
                                               shares_history: SharesHistory = None,
                                               progress_bar: bool = True) -> tuple[pd.DataFrame,
                                                                          pd.DataFrame,
                                                                          dict[str, list[assets.Portfolio]]]:
    """
    :param tickers: list of used tickers
    :param shares_weights_calculators:
    :param rebalance_dates: [2010-10-10,2011-10-10], first date --- start of the backtest
    :return: table of portfolio values in USD (shares + cash), indexed with Dates, columns = calculators = 'MCAP', 'equal', 'exp_cov'
    :return: table of rebalance fees in USD, indexed with Dates, columns = calculators = "MCAP", "equal", "exp_cov'
    :return: portfolios_history over time: {'MCAP': [(100.0,{'GOOG':1.0, 'AMZN': 1.0}), (10.12,{'GOOG':2.0, 'AMZN': 2.0})], 'equal':[]}
    """
    # load data
    if not shares_history:
        shares_history = dataloader.load_shares_history(set(tickers))
    # init dataframes
    values_history_per_calc = pd.DataFrame(index=rebalance_dates)
    fees_history_per_calc = pd.DataFrame(index=rebalance_dates)
    portfolio_history_per_calc = dict()
    for calculator in shares_weights_calculators:
        # init history
        portfolios, fees_history = reallocate_portfolio_periodically(calculator,
                                                                            rebalance_dates,
                                                                            initial_cash,
                                                                            fees_percent,
                                                                            shares_history,
                                                                            progress_bar)
        # save to dataframes
        calc_name = str(calculator)
        portfolio_history_per_calc[calc_name] = portfolios
        fees_history_per_calc[calc_name] = fees_history
        values_history_per_calc[calc_name] = portfolios_values_history(list(zip(rebalance_dates, portfolios)),
                                                                       shares_history.price_history)
    return values_history_per_calc, fees_history_per_calc, portfolio_history_per_calc
