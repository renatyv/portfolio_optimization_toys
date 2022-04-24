import pypoanal.portfolio_calculators as pc
import pandas as pd
import datetime
import numpy as np
from hypothesis import given, example
from hypothesis import strategies as st
from pytest import approx


def test_MCAPWeightsCalculator_nans():
    mcap_calc = pc.compute_mcap_weights
    ticker_nshares = pd.Series({'GOOG': 100.0, 'AMZN': 100.0})
    price_history = pd.DataFrame({'GOOG': [100.0, 100.0], 'AMZN': [100.0, np.nan]},
                                 index=[datetime.date(2010,1,1),
                                        datetime.date(2010,1,2)])
    weights: pd.Series = mcap_calc(ticker_nshares, price_history)
    assert weights.to_dict() == {'GOOG': 0.5, 'AMZN': 0.5}

def test_MCAPWeightsCalculator_equal_1():
    mcap_calc = pc.compute_mcap_weights
    ticker_nshares = pd.Series({'GOOG': 100.0, 'AMZN': 200.0})
    price_history = pd.DataFrame({'GOOG': [200.0], 'AMZN': [100.0]},
                                 index=[datetime.date(2010, 1, 1)])
    weights: pd.Series = mcap_calc(ticker_nshares, price_history)
    assert weights.to_dict() == {'GOOG': 0.5, 'AMZN': 0.5}


def test_MCAPWeightsCalculator_1():
    mcap_calc = pc.compute_mcap_weights
    ticker_nshares = pd.Series({'GOOG': 10.0, 'AMZN': 10.0})
    price_history = pd.DataFrame({'GOOG': [200.0], 'AMZN': [800.0]},
                                 index=[datetime.date(2010, 1, 1)])
    weights: pd.Series = mcap_calc(ticker_nshares, price_history)
    assert weights.to_dict() == {'GOOG': 0.2, 'AMZN': 0.8}


def test_MCAPWeightsCalculator_empy_shares():
    mcap_calc = pc.compute_mcap_weights
    ticker_nshares = pd.Series()
    price_history = pd.DataFrame({'GOOG': [200.0], 'AMZN': [100.0]},
                                 index=[datetime.date(2010, 1, 1)])
    weights: pd.Series = mcap_calc(ticker_nshares, price_history)
    assert weights.to_dict() == {}


def test_MCAPWeightsCalculator_nan_history():
    mcap_calc = pc.compute_mcap_weights
    ticker_nshares = pd.Series({'GOOG': 200.0, 'AMZN': 100.0})
    price_history = pd.DataFrame({'GOOG': [np.nan], 'AMZN': [np.nan]},
                                 index=[datetime.date(2010, 1, 1)])
    weights: pd.Series = mcap_calc(ticker_nshares, price_history)
    assert weights.to_dict() == {}

@given(goog_c=st.integers(min_value=0), amzn_c=st.integers(min_value=0), goog_p=st.floats(min_value=0.0), amzn_p=st.floats(min_value=0.0))
def test_MCAPWeightsCalculator_hypothesis(goog_c, amzn_c, goog_p, amzn_p):
    mcap_calc = pc.compute_mcap_weights
    ticker_nshares = pd.Series({'GOOG': goog_c, 'AMZN': amzn_c})
    price_history = pd.DataFrame({'GOOG': [goog_p], 'AMZN': [amzn_p]},
                                 index=[datetime.date(2010, 1, 1)])
    weights: pd.Series = mcap_calc(ticker_nshares, price_history)
    assert weights.empty or weights.sum() == approx(1.0)


