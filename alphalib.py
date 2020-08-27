import numpy as np


def sharpe(rets, rf, periods):
    """
    Returns Sharpe Ratio, assuming that risk-free rate stayed constant.
    
    rf - annualized risk-free rate
    periods - periods in a year
    """
    raise NotImplementedError()


def vol(rets):
    return rets.std()


def annualized_vol(rets, periods):
    return vol(rets) * np.sqrt(periods)


def cum_returns(rets):
    return (rets + 1).cumprod() - 1


def cagr(rets, periods):
    """
    Cumulative Annual Growth Rate (CAGR)
    """
    cum_period_return = (1 + rets).prod() ** (1 / len(rets)) - 1
    return (1 + cum_period_return) ** periods - 1
