import datetime

import pypoanal.dataloader
import pypoanal.rebalancer
from pypoanal import analyzer as analyzer
import pandas as pd


def test_get_quote_type():
    assert pypoanal.dataloader.get_quote_type('GOOG') == 'EQUITY'
    assert pypoanal.dataloader.get_quote_type('VTI') == 'ETF'
    assert pypoanal.dataloader.get_quote_type('AAV') == 'MUTUALFUND'
    assert pypoanal.dataloader.get_quote_type('KISSMYASSS') is None


def test_portfolio_performance():
    year_1 = datetime.date(2000, 1, 1)
    year_2 = datetime.date(2001, 1, 1)
    year_3 = datetime.date(2002, 1, 1)
    constant_prices = pd.DataFrame({'Date': [year_1, year_2, year_3],
                                    'AAPL': [100.0, 100.0, 100.0],
                                    'GOOG': [200.0, 200.0, 200.0]})
    constant_prices.set_index('Date', inplace=True)
    double_prices = pd.DataFrame({'Date': [year_1, year_2, year_3],
                                  'AAPL': [100.0, 150.0, 200.0],
                                  'GOOG': [100.0, 150.0, 200.0]})
    double_prices.set_index('Date', inplace=True)
    equal_portfolio_weights = pd.Series({'AAPL': 0.5, 'GOOG': 0.5})
    sigma, ret = analyzer.shares_weights_performance(equal_portfolio_weights, constant_prices)
    assert sigma == 0
    assert ret == 0
    sigma, ret = analyzer.shares_weights_performance(equal_portfolio_weights, double_prices)
    assert ret > 0.9
