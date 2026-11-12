from regression.summary_stats import SummaryStatistics
import numpy as np

class SimpleOLS:
    """
    Simple linear regression fitted by ordinary least squares.
    Fits ``y_hat = (b0 + b1 * x)`` in closed form.

    Parameters
    ----------
    x : np.ndarray
        1D array of independent (explanatory) variable observations.
    y : np.ndarray
        1D array of dependent variable observations.

    Attributes
    ----------
    stats : SummaryStatistics
        Validated summary statistics instance for training data `x` and `y`.
    b1 : float
        Estimated slope coefficient (dev_xy / dev_x2).
    b0 : float
        Estimated intercept term (mean_y - b1 * mean_x).
    y_hat : np.ndarray
        Read-only fitted values for training data (b0 + b1 * x).
    residuals : np.ndarray
        Read-only residual vector (y - y_hat).
    residual_sum2 : float
        Sum of squared residuals (SSE).
    std_error : float
        Sample standard error of the estimate (n - 2 degrees of freedom).
    pop_std_error : float
        Population standard error of the estimate (N in denominator).

    Raises
    ------
    ValueError
        If inputs violate dimensionality, size (n < 3), finiteness, or variance rules.
    """
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.stats = SummaryStatistics(x, y)
        self.b1 = self.stats.dev_xy / self.stats.dev_x2
        self.b0 = self.stats.mean_y - (self.b1 * self.stats.mean_x)
        self.y_hat = self.b0 + (self.b1 * self.stats.x)
        self.y_hat.flags.writeable = False
        self.residuals = self.stats.y - self.y_hat
        self.residuals.flags.writeable = False
        self.residual_sum2 = float(self.residuals @ self.residuals)
        self.std_error = float(np.sqrt(self.residual_sum2 / (self.stats.n - 2)))
        self.pop_std_error = float(np.sqrt(self.residual_sum2 / self.stats.n))

    def predict(self, x_new: np.ndarray) -> np.ndarray:
        """
        Predict dependent variable values for new observations.

        Parameters
        ----------
        x_new : np.ndarray
            1D array of new independent variable values.

        Returns
        -------
        np.ndarray
            Predicted values vector (b0 + b1 * x_new).

        Raises
        ------
        ValueError
            If `x_new` is not 1D, is empty, or contains non-finite values.
        """
        x_new = np.asarray(x_new, dtype=np.float64)
        if x_new.ndim != 1:
            raise ValueError("x_new must have 1 dimension")
        elif x_new.size == 0:
            raise ValueError("x_new must have at least one element")
        elif not np.all(np.isfinite(x_new)):
            raise ValueError("x_new must be finite")
        prediction = self.b0 + (self.b1 * x_new)
        return prediction