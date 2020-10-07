import numpy as np


class Portfolio:

    def __init__(self):
        pass

    @staticmethod
    def get_returns(weights, returns):
        """
        Calculates weighted return for each portfolio. If both 'weights' and 'returns' are 2d array,
        it is considered that each row represents a different portfolio.

        Arguments:
            weights: np.ndarray
                Weights for each asset in each portfolio.

            returns: np.ndarray
                Returns for each asset in each portfolio.

        Returns:
            np.ndarray
                An array with weighted returns for each portfolio.
        """
        return weights @ returns.T

    @staticmethod
    def get_volatility(weights, cov_mat):
        """
        TODO: create docstring for Portfolio.get_volatility()
        """

        if weights.ndim == 1:
            return np.sqrt(weights @ cov_mat @ weights.T)
        else:
            return np.sqrt(np.multiply(weights @ cov_mat, weights).sum(axis=1))

    @staticmethod
    def get_sharpe(weights, returns, cov_mat, rf):
        r = Portfolio.get_returns(weights, returns)
        sigma = Portfolio.get_volatility(weights, cov_mat)

        return (r - rf) / sigma
    