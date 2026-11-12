import numpy as np

class SummaryStatistics:
    """
    Foundational summary statistics and centered cross-products for 1D arrays `x` and `y`.

    Parameters
    ----------
    x : np.ndarray
        1D array of independent (explanatory) variable observations.
    y : np.ndarray
        1D array of dependent variable observations.

    Attributes
    ----------
    n : int
        Number of paired observations.
    x, y : np.ndarray
        Read-only 1D float64 input arrays.
    sum_x, sum_y : float
        Raw sums of `x` and `y`.
    mean_x, mean_y : float
        Arithmetic means of `x` and `y`.
    sum_x2, sum_y2, sum_xy : float
        Raw sums of squares and cross-products.
    dev_x, dev_y : np.ndarray
        Read-only centered deviation vectors (x - mean_x) and (y - mean_y).
    dev_x2, dev_y2, dev_xy : float
        Sums of squared deviations and cross-deviations.

    Raises
    ------
    ValueError
        If `x` or `y` is not 1D, if sizes do not match, if `n < 3`,
        if any element is non-finite (NaN/Inf), or if `x` is constant (zero variance).

    Notes
    -----
    Two-pass centering (`x - mean_x`) is used instead of naive one-pass raw-sum
    identities to prevent IEEE 754 floating-point catastrophic cancellation, while
    BLAS dot products (`@`) compute sums of squares without allocating intermediate squared arrays.
    """
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.array(x, dtype=np.float64, copy=True)
        y = np.array(y, dtype=np.float64, copy=True)
        x.flags.writeable = False
        y.flags.writeable = False
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y must have 1 dimension")
        elif x.size != y.size:
            raise ValueError("x and y must have the same size")
        elif x.size < 3:
            raise ValueError("x and y must have at least three elements")
        elif not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise ValueError("x and y must all be finite (not nan)")
        elif np.ptp(x) == 0:
            raise ValueError("x must not be constant (zero variance)")
        self.n = int(x.size)
        self.x = x
        self.y = y
        self.sum_x = float(np.sum(x))
        self.mean_x = float(self.sum_x / self.n)
        self.dev_x = x - self.mean_x
        self.dev_x.flags.writeable = False
        self.dev_x2 = float(self.dev_x @ self.dev_x)
        self.sum_y = float(np.sum(y))
        self.mean_y = float(self.sum_y / self.n)
        self.sum_x2 = float(x @ x)
        self.sum_y2 = float(y @ y)
        self.dev_y = y - self.mean_y
        self.dev_y.flags.writeable = False
        self.sum_xy = float(x @ y)
        self.dev_xy = float(self.dev_x @ self.dev_y)
        self.dev_y2 = float(self.dev_y @ self.dev_y)