import datetime

from pypoanal import backtester, portfolio_calculators


def test_backtest_MCAP():
    start_date = datetime.date(2009, 5, 10)
    backtest_end_date = datetime.date(2022, 1, 15)
    rebalance_period = datetime.timedelta(days=360)
    rebalance_dates = backtester.compute_rebalance_dates(start_date, backtest_end_date, rebalance_period)
    initial_cash = 10 ** 4
    tickers_list = ['ABCL', 'ABGI', 'ACIW', 'AAPL', 'GOOG']
    calculators = {'MCAP': portfolio_calculators.compute_mcap_weights}
    values_df, _, _ = backtester.compare_calculators_for_periodic_rebalance(calculators, tickers_list,
                                                                            initial_cash, rebalance_dates,
                                                                            progress_bar=True)
    assert len(values_df) > 0


def test_backtest_ECOV():
    start_date = datetime.date(2009, 5, 10)
    backtest_end_date = datetime.date(2022, 1, 15)
    rebalance_period = datetime.timedelta(days=360)
    rebalance_dates = backtester.compute_rebalance_dates(start_date, backtest_end_date, rebalance_period)
    initial_cash = 10 ** 4
    tickers_list = ['ABCL', 'ABGI', 'ACIW', 'AAPL', 'GOOG']
    calculators = {'ECOV': portfolio_calculators.compute_expcov_weights}
    values_df, _, _ = backtester.compare_calculators_for_periodic_rebalance(calculators,
                                                                            tickers_list,
                                                                            initial_cash,
                                                                            rebalance_dates,
                                                                            progress_bar=True)
    assert len(values_df) > 0
