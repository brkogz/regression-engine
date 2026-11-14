import numpy as np
from regression.summary_stats import SummaryStatistics


class Correlation:
    """
    Compute Pearson correlation and variance decomposition for simple linear regression.

    Parameters
    ----------
    x : np.ndarray
        One-dimensional array of independent variable observations.
    y : np.ndarray
        One-dimensional array of dependent variable observations.

    Attributes
    ----------
    stats : SummaryStatistics
        Validated summary statistics and centered cross-deviation sums.
    total_variation : float
        Total sum of squares of the dependent variable around its mean.
    r : float
        Sample Pearson linear correlation coefficient in the range [-1.0, 1.0].
    r_squared : float
        Coefficient of determination representing the explained proportion of variance.
    explained_variation : float
        Sum of squares of the dependent variable explained by the linear model.
    unexplained_variation : float
        Residual sum of squares not explained by the linear model.
    unexplained_ratio : float
        Coefficient of alienation representing the unexplained proportion of variance.

    Raises
    ------
    ValueError
        If inputs fail summary statistics validation or if y has zero variance.

    Notes
    -----
    The correlation coefficient is computed directly from centered deviations
    rather than raw uncentered sums or residual square roots to preserve the sign
    of the relationship and prevent floating-point catastrophic cancellation.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.stats = SummaryStatistics(x, y)
        if np.ptp(self.stats.y) == 0.0:
            raise ValueError("y must not be constant (zero variance)")
        self.total_variation = self.stats.dev_y2
        self.r = float(self.stats.dev_xy / np.sqrt(self.stats.dev_x2 * self.total_variation))
        self.r_squared = self.r * self.r
        self.explained_variation = self.r_squared * self.total_variation
        self.unexplained_variation = self.total_variation - self.explained_variation
        self.unexplained_ratio = self.unexplained_variation / self.total_variation