import pandas as pd
import scipy.stats
import numpy as np


class Returns:
    """

    """
    def __init__(self, rets, periods):

        if rets is None:
            raise ValueError("'rets' cannot be None")

        if isinstance(rets, pd.DataFrame) or isinstance(rets, pd.Series):
            self.rets = rets
        else:
            self.rets = pd.DataFrame(rets)

        self.periods = periods
        self.num_periods = self.rets.shape[0]

    def __getitem__(self, item):
        return Returns(self.rets[item], self.periods)

    def mean(self):
        """
        Returns mean returns.
        """
        return self.rets.mean()

    def std(self, ddof=1):
        """
        Returns standard deviaton (volatility).

        Arguments:
            ddof: int, default=1
                Degrees of freedom. The standard deviaton is normalized by N - ddof. If ddof = 1, the function returns
                sample standard devation.
        """
        return self.rets.std(ddof=ddof)

    def semideviation(self):
        """
        Returns semideviation.
        """
        return self.rets[self.rets < self.rets.mean()].std()

    def skewness(self):
        """
        Returns skewness of returns.
        """
        return self.rets.skew()

    def kurtosis(self):
        """
        Returns kurtosis of returns.
        """
        return self.rets.kurtosis()

    def correlation(self):
        """
        Returns correlation of returns.
        """
        return self.rets.corr()

    def covariance(self):
        """
        Returns covariance-variance matrix.
        """
        return self.rets.cov()

    def var(self, var_type='historical', conf_level=0.95):
        """
        Returns Value-at-Risk (VaR).

        Arguments:
            var_type: {'historical', 'gaussian', 'cornish-fisher'}, default='historical'

            conf_level: int, default=95%

        Returns:
            VaR: float
                Value-at-Risk
        """
        if var_type == 'historical':
            return -self.rets.quantile(1 - conf_level)

        elif var_type == 'gaussian':
            sigma = self.std()
            avg = self.mean()
            return -(avg + sigma * scipy.stats.norm.ppf(1 - conf_level))

        elif var_type == 'cornish-fisher':
            # https://en.wikipedia.org/wiki/Cornish-Fisher_expansion
            skew = self.skewness()
            kurt = self.kurtosis()
            z = scipy.stats.norm.ppf(1 - conf_level)
            z = z + (z ** 2 - 1) / 6 * skew + (z ** 3 - 3 * z) * kurt / 24 - (2 * z ** 3 - 5 * z) / 36 * skew ** 2

            return -(self.mean() + self.std() * z)

        else:
            raise ValueError("Only 'historical', 'gaussian' and 'cornish-fisher' VaR types are accepted")

    def cvar(self, conf_level=0.95):
        """
        Returns Conditional Value-at-Risk (CVaR).

        Arguments:
            conf_level: int, default=95%

        Returns:
            float
        """
        return -(self.rets[self.rets < -self.var('historical', conf_level)].mean())

    def drawdowns(self):
        """
        Returns drawdowns for each time step.

        Returns:
            drawdowns: pandas Series or DataFrame
        """
        raise NotImplemented()

    def annualized_rets(self):
        """
        Returns annualized returns.
        """
        return (self.rets + 1).prod() ** (self.periods / self.num_periods) - 1

    def annualized_vol(self):
        """
        Returns annualized volatility.
        """
        return self.std() * np.sqrt(self.periods)

    def sharpe(self, risk_free_rate):
        """
        Returns Sharpe ratios.

        Arguments:
            risk_free_rate: float

        Returns:
            sharpe: pandas Series of DataFrame
        """
        return (self.annualized_rets() - risk_free_rate) / self.annualized_vol()

    def cagr(self):
        """
        Returns Cumulative Annual Growth Rate (CAGR)
        """
        raise NotImplemented()

    def test(self):
        pass
