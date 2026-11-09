import numpy as np

class SummaryStatistics:
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








