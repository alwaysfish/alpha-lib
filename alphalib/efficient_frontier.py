import pandas as pd
import numpy as np
from .portfolio import Portfolio


class EfficientFrontier:
    """
    Efficient Frontier class allows to find different types of portfolios along the
    efficient frontier curve. It can be used to find weights of minimum variance and highest
    Sharpe ratio portfolios.

    Arguments:
        er: pd.Series
            Expected returns for each asset.

        cov_mat: np.ndarray
            Covariance matrix of returns.
    """
    def __init__(self, er, cov_mat):
        self.er = er
        self.cov_mat = cov_mat
        self.n_assets = len(er)
        self.asset_names = er.index.values

    def min_variance(self):
        """
        Returns weights of the portfolio with minimum variance.

        Returns:
            weights: pd.Series
        """

        raise NotImplemented()

    def max_sharpe(self, rf):
        """
        Returns weights of the portfolio with the highest Sharpe ratio.

        Arguments:
            rf: float
                Risk free rate. The period of the risk free rate should correspond to the
                frequency of expected returns.

        Returns:
             weights: pd.Series
        """

        raise NotImplemented()

    def get_random_portfolios(self, n=100, return_weights=False):
        """
        Generates and returns weights for n random portfolios.

        Arguments:
            n: int
                Number of portfolios to generate.

            return_weights: bool, default=False
                True if weights for each portfolio should also be returned.

        Returns:
            DataFrame with expected returns and volatility for each portfolio.
            If 'return_weights' is True, then weights for each asset in each portfolio are also returned.
        """

        weights = self._get_rand_weights(n)

        p_r = Portfolio.get_returns(weights, self.er)
        p_sigma = Portfolio.get_volatility(weights, self.cov_mat)

        p_df = pd.DataFrame({'return': p_r, 'volatility': p_sigma})

        if return_weights:
            return p_df, weights
        else:
            return p_df

    def _get_rand_weights(self, n):
        """
        Returns a DataFrame with random weights for each portfolio. The dimensions of returned
        DataFrame is m x n, where m - number of assets, n - number of portfolios.

        Arguments:
            n: int
                Number of portfolios to generate weights for.

        Returns:
            weights: pd.DataFrame
                A DataFrame with weights for each assset in each portfolio.
        """

        w = np.random.rand(n, self.n_assets)
        return pd.DataFrame(w / np.sum(w, axis=1, keepdims=True), columns=self.asset_names)
