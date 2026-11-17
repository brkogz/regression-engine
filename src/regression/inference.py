import numpy as np
import scipy.stats as sp_stats

from regression.correlation import Correlation
from regression.simple_ols import SimpleOLS


class Inference:
    """
    Statistical inference, hypothesis testing, and prediction intervals for simple OLS.

    Parameters
    ----------
    x : np.ndarray
        1D array of independent variable observations.
    y : np.ndarray
        1D array of dependent variable observations.
    alpha : float, default=0.05
        Significance level for two-tailed hypothesis tests and prediction
        intervals, strictly within the open interval (0, 1).

    Attributes
    ----------
    alpha : float
        Significance level in (0, 1).
    ols : SimpleOLS
        Fitted ordinary least squares regression model instance.
    stats : SummaryStatistics
        Underlying summary statistics instance reused from the OLS model.
    correlation : Correlation
        Fitted correlation analysis instance.
    r_std_error : float
        Standard error of the sample correlation coefficient.
    b0_std_error : float
        Standard error of the intercept estimator.
    b1_std_error : float
        Standard error of the slope estimator.
    t_stat : float
        Two-tailed test statistic for the slope and correlation coefficients.
    f_stat : float
        Analysis of variance F-test statistic computed via the identity F = t^2.
    f_critical : float
        Right-tail critical value from the F-distribution with (1, n - 2) degrees of freedom.
    critical_value : float
        Two-tailed critical value from the Student's t-distribution (df = n - 2)
        for small samples (n < 30) or standard normal Z-distribution for large samples (n >= 30).
    is_significant : bool
        True if the absolute test statistic exceeds the critical value, rejecting
        the null hypothesis of zero linear relationship.

    Raises
    ------
    ValueError
        If `alpha` is not strictly between 0 and 1, if a near-perfect linear fit
        (`unexplained_ratio <= 1e-12`) prevents valid standard error estimation,
        or if input arrays fail dimensionality, size, finiteness, or variance checks.

    Notes
    -----
    Reuses precomputed centered sums of squares from `SummaryStatistics` and point
    estimates from `SimpleOLS` and `Correlation` to avoid redundant array allocations.
    Enforces a scale-invariant threshold (`unexplained_ratio <= 1e-12`) prior to
    computing standard errors to prevent division-by-zero instabilities.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> None:
        if not ((alpha > 0) and (alpha < 1)):
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = float(alpha)
        self.ols = SimpleOLS(x, y)
        self.stats = self.ols.stats
        self.correlation = Correlation(x, y)
        if self.correlation.unexplained_ratio <= 1e-12:
            raise ValueError("perfect linear fit is not allowed for inference")
        self.r_std_error = float(np.sqrt(self.correlation.unexplained_ratio / (self.stats.n - 2)))
        self.b0_std_error = float(self.ols.std_error * np.sqrt(
            1 / self.stats.n + (self.stats.mean_x * self.stats.mean_x / self.stats.dev_x2)))
        self.b1_std_error = float(self.ols.std_error / np.sqrt(self.stats.dev_x2))
        self.t_stat = float(self.ols.b1 / self.b1_std_error)
        self.f_stat = self.t_stat * self.t_stat
        self.f_critical = float(sp_stats.f.ppf(1 - self.alpha, 1, self.stats.n - 2))
        if self.stats.n < 30:
            self.critical_value = float(sp_stats.t.ppf(1 - (self.alpha / 2), self.stats.n - 2))
        else:
            self.critical_value = float(sp_stats.norm.ppf(1 - (self.alpha / 2)))
        self.is_significant = bool(abs(self.t_stat) > self.critical_value)

    def predict_interval(self, x_input: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute exact two-sided prediction intervals for new observations.

        Parameters
        ----------
        x_input : np.ndarray
            1D array of new independent variable values.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Two 1D arrays `(lower_bound, upper_bound)` representing the lower and
            upper prediction interval bounds for each element in `x_input`.

        Raises
        ------
        ValueError
            If `x_input` is not 1D, is empty, or contains non-finite values.

        Notes
        -----
        Incorporates residual variance, sample mean uncertainty (1 / n), and
        out-of-sample leverage penalty via element-wise vectorization without
        allocating intermediate reduction matrices.
        """
        y_pred = self.ols.predict(x_input)
        dev_x_input = x_input - self.stats.mean_x
        distance_penalty = (dev_x_input * dev_x_input) / self.stats.dev_x2
        margin_of_error = self.critical_value * self.ols.std_error * np.sqrt(1 + 1 / self.stats.n + distance_penalty)
        lower_bound = y_pred - margin_of_error
        upper_bound = y_pred + margin_of_error
        return lower_bound, upper_bound