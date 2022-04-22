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


# use backtester to compute rolling performance
def rolling_performance(calculators: list[pcalc.PortfolioWeightsCalculator],
                        tickers: list[str],
                        start_date: datetime.date = datetime.date(2012, 1, 1),
                        end_date: datetime.date = datetime.date.today(),
                        step_days: int = 180,
                        sample_period: datetime.timedelta = datetime.timedelta(days=365),
                        test_period: datetime.timedelta = datetime.timedelta(days=180),
                        shares_history: dataloader.SharesHistory = None) -> (pd.DataFrame, pd.DataFrame):
    """
    :param calculators:
    :param shares_history: (prices_history_df, volumes_history_df, shares_outstanding)
    :param start_date:
    :param end_date:
    :param step_days: more days => less frequency data
    :param sample_period: train sample length
    :param test_period: test sample length
    :return: return rates: table index=Date, columns=(MCAP, exp_cov, HRP,...) --- name of the calculator,
    values = return rate
    :return: sigmas: table index=Date, columns=(MCAP, exp_cov, HRP,...) --- name of the calculator,
    values = annualized volatility (sigma)
    """
    # prepare dates list
    full_period_days = (end_date - sample_period - test_period - start_date).days
    num_steps = math.floor(full_period_days / step_days)
    sample_start_dates = [start_date + datetime.timedelta(days=step_num * step_days) for step_num in range(num_steps)]
    # load data
    if not shares_history:
        shares_history = dataloader.load_shares_history(set(tickers))
    prices_history_df, volumes_history_df, shares_outstanding = \
        shares_history.price_history, shares_history.volume_history, shares_history.shares_outstanding
    # init returned dataframes
    return_rates_df = pd.DataFrame(index=sample_start_dates)
    sigmas_df = pd.DataFrame(index=sample_start_dates)
    for calculator in calculators:
        print(calculator)
        sigmas = []
        return_rates = []
        for sample_start in tqdm.tqdm(sample_start_dates):
            sample_end = sample_start + sample_period
            test_end = sample_end + test_period
            liquid_tickers: list[str] = choose_liquid_tickers(volumes_history_df, sample_start, test_end)
            sample_prices_df = prices_history_df.loc[sample_start:sample_end, liquid_tickers]
            test_prices_df = prices_history_df.loc[sample_end:test_end, liquid_tickers]
            try:
                new_portfolio_weights = calculator.get_weights(shares_outstanding, sample_prices_df)
            except (SolverError, OptimizationError, ArpackNoConvergence,
                    ValueError) as anc:  # ArpackNoConvergence for ledoit, ValueError for HRP
                warnings.warn(f'{str(anc)}')
                sigma = np.nan
                return_rate = np.nan
            else:
                # cleaned_weights = remove_insignificant_weights(new_portfolio_weights, weight_remove=0.05)
                sigma, return_rate = shares_weights_performance(new_portfolio_weights, test_prices_df)
            sigmas.append(sigma)
            return_rates.append(return_rate)
        # Put results into dataframe
        calc_name = str(calculator)
        return_rates_df[calc_name] = return_rates
        sigmas_df[calc_name] = sigmas
    return return_rates_df, sigmas_df
