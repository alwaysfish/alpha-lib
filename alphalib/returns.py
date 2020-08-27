import pandas as pd
import scipy.stats


class Returns:
    def __init__(self, rets):

        if rets is None:
            raise ValueError("'rets' cannot be None")

        self.rets = rets

    def skewness(self):
        """
        Calculates skewness of returns.
        """
        if isinstance(self.rets, pd.DataFrame) or isinstance(self.rets, pd.Series):
            return self.rets.skew()
        else:
            return scipy.stats.skew(self.rets, bias=False)

    def kurtosis(self):
        """
        Calculates kurtosis of returns.
        """
        if isinstance(self.rets, pd.DataFrame) or isinstance(self.rets, pd.Series):
            return self.rets.kurtosis()
        else:
            return scipy.stats.kurtosis(self.rets, bias=False)

    def correlation(self):
        """
        Calculates correlation of returns.
        """
        return self.rets.corr()

    def var(self, var_type='historical', conf_level=0.95):
        """
        Calculates Value-at-Risk (VaR).

        Arguments:
            var_type: {'historical', 'gaussian', 'cornish-fisher'}, default='historical'

            conf_level: int, default=95%
        """
        if var_type == 'historical':
            return -self.rets.quantile(1 - conf_level)

        elif var_type == 'gaussian':
            sigma = self.rets.std()
            avg = self.rets.mean()
            return -(avg + sigma * scipy.stats.norm.ppf(1 - conf_level))

        elif var_type == 'cornish-fisher':
            # https://en.wikipedia.org/wiki/Cornish-Fisher_expansion
            skew = self.skewness()
            kurt = self.kurtosis()
            z = scipy.stats.norm.ppf(1 - conf_level)
            z = z + (z ** 2 - 1) / 6 * skew + (z ** 3 - 3 * z) * kurt / 24 - (2 * z ** 3 - 5 * z) / 36 * skew ** 2

            return -(self.rets.mean() + self.rets.std() * z)

        else:
            raise ValueError("Only 'historical', 'gaussian' and 'cornish-fisher' VaR types are accepted")

    def cvar(self):
        """
        Calculates Conditional Value-at-Risk (VaR).
        """
        return -(self.rets[self.rets < -self.var('historical', 0.95)].mean())
