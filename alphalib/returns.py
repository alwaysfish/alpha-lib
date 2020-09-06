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

        if periods == 'y':
            self.periods = 1
        elif periods == 'q':
            self.periods = 4
        elif periods == 'm':
            self.periods = 12
        elif periods == 'd':
            raise NotImplemented()

        self.num_periods = self.rets.shape[0]

    def __getitem__(self, item):
        return self.rets[item]

    def mean(self):
        return self.rets.mean()

    def std(self):
        return self.rets.std()

    def skewness(self):
        """
        Calculates skewness of returns.
        """
        return self.rets.skew()

    def kurtosis(self):
        """
        Calculates kurtosis of returns.
        """
        return self.rets.kurtosis()

    def correlation(self):
        """
        Calculates correlation of returns.
        """
        return self.rets.corr()

    def covariance(self):
        raise NotImplemented()

    def var(self, var_type='historical', conf_level=0.95):
        """
        Calculates Value-at-Risk (VaR).

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
        Calculates Conditional Value-at-Risk (VaR).

        Arguments:
            conf_level: int, default=95%

        Returns:
            float
        """
        return -(self.rets[self.rets < -self.var('historical', conf_level)].mean())

    def drawdowns(self):
        """
        Calculates drawdowns for each time step.

        Returns:
            drawdowns: pandas Series or DataFrame

        """
        raise NotImplemented()

    def annualized_rets(self):
        """
        Returns annualized returns.
        """
        rets = (self.rets + 1).prod() ** (self.periods / self.num_periods) - 1
        return rets

    def annualize_vol(self):
        raise NotImplemented()

    def sharpe(self):
        raise NotImplemented()
