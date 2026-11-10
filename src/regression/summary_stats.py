import numpy as np

class SummaryStatistics:
    """
        Compute and cache foundational summary statistics for simple linear
        regression and correlation analysis.

        Grounded in Prof. Dr. Neyran Orhunbilge, *Applied Regression and
        Correlation Analysis* (Chapter II.1, pp. 8-12, Footnote 10).

        Parameters
        ----------
        x : np.ndarray
            One-dimensional array of independent (explanatory) variable observations.
        y : np.ndarray
            One-dimensional array of dependent variable observations.

        Attributes
        ----------
        n : int
            Number of paired observations (sample size n or population size N).
        x : np.ndarray
            Validated 1D float64 array of independent variable values.
        y : np.ndarray
            Validated 1D float64 array of dependent variable values.
        sum_x : float
            Raw sum of x values, \\sum x.
        sum_y : float
            Raw sum of y values, \\sum y.
        mean_x : float
            Arithmetic mean of x, \\bar{x} = (\\sum x) / n.
        mean_y : float
            Arithmetic mean of y, \\bar{y} = (\\sum y) / n.
        sum_x2 : float
            Raw sum of squared x values, \\sum x^2.
        sum_y2 : float
            Raw sum of squared y values, \\sum y^2.
        sum_xy : float
            Raw sum of cross-products, \\sum xy.
        dev_x : np.ndarray
            Centered deviation vector for x, (x - \\bar{x}).
        dev_y : np.ndarray
            Centered deviation vector for y, (y - \\bar{y}).
        dev_x2 : float
            Sum of squared deviations of x, \\sum(x - \\bar{x})^2.
        dev_y2 : float
            Sum of squared deviations of y, \\sum(y - \\bar{y})^2.
        dev_xy : float
            Sum of cross-deviations, \\sum(x - \\bar{x})(y - \\bar{y}).

        Raises
        ------
        ValueError
            If `x` or `y` is not 1D, if sizes do not match, if `n < 3`,
            if any element is non-finite (NaN/Inf), or if `x` has zero variance.

        Notes
        -----
        Although Footnote 10 (p. 10) derives the algebraic raw-sum identities
        \\sum(x - \\bar{x})^2 = \\sum x^2 - (\\sum x)^2 / n for manual calculation,
        this engine computes `dev_x2`, `dev_y2`, and `dev_xy` directly via BLAS
        dot products (`@`) on centered vectors to prevent IEEE 754 floating-point
        catastrophic cancellation.
        """
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y must have 1 dimension")
        elif x.size != y.size:
            raise ValueError("x and y must have the same size")
        elif x.size < 3:
            raise ValueError("x and y must have at least three elements")
        elif not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise ValueError("x and y must all be finite (not nan)")
        self.n = int(x.size)
        self.sum_x = float(np.sum(x))
        self.mean_x = float(self.sum_x / self.n)
        self.dev_x = x - self.mean_x
        self.dev_x2 = float(self.dev_x @ self.dev_x)
        if np.isclose(self.dev_x2, 0, atol=1e-8):
            raise ValueError("dev_x2 cannot be 0")
        self.x = x
        self.y = y
        self.sum_y = float(np.sum(y))
        self.mean_y = float(self.sum_y / self.n)
        self.sum_x2 = float(x @ x)
        self.sum_y2 = float(y @ y)
        self.dev_y = y - self.mean_y
        self.sum_xy = float(x @ y)
        self.dev_xy = float(self.dev_x @ self.dev_y)
        self.dev_y2 = float(self.dev_y @ self.dev_y)

