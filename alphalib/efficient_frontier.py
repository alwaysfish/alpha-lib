import pandas as pd
import numpy as np
from scipy.optimize import minimize
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

    def min_volatility(self, *, target_ret=None):
        """
        Returns weights of the portfolio with minimum volatility.

        Arguments:
            target_ret: float, default=None
                Target return, for which to find a portfolio with lowest volatility. If target_ret is None,
                then weights for Global Minimum-Variance portfolio are returned.

        Returns:
            weights: pd.Series
        """

        if target_ret is None:
            return self._optimize(0, gmv=True)
        else:
            return self._optimize_for_target('min_vol', target_ret)

    def max_sharpe(self, *, rf):
        """
        Returns weights of the portfolio with the highest Sharpe ratio.

        Arguments:
            rf: float
                Risk free rate. The period of the risk free rate should correspond to the
                frequency of expected returns.

        Returns:
             weights: pd.Series
        """
        return self._optimize(rf)

    def max_return(self, *, target_vol):
        """
        Returns weights of the portfolio with the highest return for a given traget volatility.

        Arguments:
            target_vol: float
                Target volatility for which to find a portfolio with the highest return.

        Returns:
            weights: pd.Series
        """
        return self._optimize_for_target('max_ret', target_vol)

    def plot(self):
        """
        Plots Efficient Frontier curve.
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

        p_r = Portfolio.get_return(weights, self.er)
        p_sigma = Portfolio.get_volatility(weights, self.cov_mat)

        p_df = pd.DataFrame({'return': p_r, 'volatility': p_sigma})

        if return_weights:
            return weights.merge(p_df, left_index=True, right_index=True)
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

    def _init_optimization(self):

        # set initial weights
        w_init = np.repeat(1 / self.n_assets, self.n_assets)

        # set bounds for weights
        w_bounds = ((0, 1),) * self.n_assets

        # constraint 1: weights add to 1
        constraint = {
            'type': 'eq',
            'fun': lambda weights: np.sum(weights) - 1
        }

        return w_init, w_bounds, constraint

    def _optimize(self, rf, gmv=False):
        """
        Finds weights of a optimal portfolio. If gmv (Global Minimum-Variance portfolio)
        is False, then a portfolio with highest Sharpe ratio is found. Otherwise, a global
        minimum-variance portfolio is found.

        Arguments:
            rf: float
                Risk free rate. The period of the risk free rate must correspond to the
                frequency of expected returns.

            gmv: bool
                True, if global minimum-variance portfolio should be found.
                False, if a portfolio with highest Sharpe ratio should be found.

        Returns:
            pd.Series
                Weights of the optimal portfolio.
        """

        w_init, w_bounds, constr1 = self._init_optimization()

        if gmv:
            er = np.repeat(0.01, self.n_assets)
        else:
            er = self.er

        def neg_sharpe(w, r, cov_mat, _rf):
            return -Portfolio.get_sharpe(w, r, cov_mat, _rf)

        results = minimize(neg_sharpe, w_init, args=(er, self.cov_mat, rf), method='SLSQP',
                           options={'disp': False}, constraints=constr1,
                           bounds=w_bounds)

        return pd.Series(results.x, index=self.asset_names)

    def _optimize_for_target(self, optimization, target):

        w_init, w_bounds, constr1 = self._init_optimization()

        if optimization == 'max_ret':
            constr2 = {
                'type': 'eq',
                'args': (self.cov_mat,),
                'fun': lambda weights, cov_mat: Portfolio.get_volatility(weights, cov_mat) - target
            }
            args = (self.er,)

            def neg_portf_return(w, er):
                return -Portfolio.get_return(w, er)

            func = neg_portf_return

        elif optimization == 'min_vol':
            constr2 = {
                'type': 'eq',
                'args': (self.er,),
                'fun': lambda weights, er: Portfolio.get_return(weights, er) - target
            }
            args = (self.cov_mat,)
            func = Portfolio.get_volatility
        else:
            raise ValueError('Only max_ret and min_vol optimizations are accepted')

        results = minimize(func, w_init,
                           args=args, method='SLSQP',
                           options={'disp': False},
                           constraints=[constr1, constr2],
                           bounds=w_bounds)

        return pd.Series(results.x, index=self.asset_names)
