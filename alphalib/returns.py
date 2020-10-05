import pandas as pd
import scipy.stats
import numpy as np


class Returns:
    """
    """
    def __init__(self, rets, freq):

        if isinstance(rets, pd.DataFrame) or isinstance(rets, pd.Series):
            self.rets = rets
        else:
            self.rets = pd.DataFrame(rets)

        self.freq = freq
        self.length = self.rets.shape[0]

    def __getitem__(self, item):
        return Returns(self.rets[item], self.freq)

    def __add__(self, other):
        return Returns(self.rets + other, self.freq)

    def __sub__(self, other):
        return Returns(self.rets - other, self.freq)

    @property
    def shape(self):
        return self.rets.shape

    def min(self):
        return self.rets.min()

    def max(self):
        return self.rets.max()

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

        Notes:
           Cornish Fisher expansion is a way to transform a standard Guassian random variable z into a non-Gaussian
           random variable.

           References:
               https://faculty.washington.edu/ezivot/econ589/ssrn-id1997178.pdf
               https://en.wikipedia.org/wiki/Cornish-Fisher_expansion
        """
        if var_type == 'historical':
            return -self.rets.quantile(1 - conf_level)

        elif var_type == 'gaussian':
            sigma = self.std()
            avg = self.mean()
            return -(avg + sigma * scipy.stats.norm.ppf(1 - conf_level))

        elif var_type == 'cornish-fisher':
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

    def annualized_vol(self):
        """
        Returns annualized volatility.
        """
        return self.std() * np.sqrt(self.freq)

    def sharpe(self, risk_free_rate):
        """
        Returns Sharpe ratio(s).

        Arguments:
            risk_free_rate: float

        Returns:
            sharpe: pandas Series
        """
        rf_per_period = (1 + risk_free_rate) ** (1 / self.freq) - 1
        excess_returns = self - rf_per_period
        return excess_returns.annualized_rets() / self.annualized_vol()

    def mean_historical(self, compounding=True, annualised=True):
        """
        Calculates mean historical return.

        Arguments:
             compounding: bool, default=True
                If True, compounded return is returned, otherwise an arithmetical mean is returned.

            annualised: bool, default=True
                If Ture, annualised mean historical return is returned.
        """

        freq = 1
        if annualised:
            freq = self.freq

        if compounding:
            return (self.rets + 1).prod() ** (freq / self.length) - 1
        else:
            return self.rets.mean() * freq
