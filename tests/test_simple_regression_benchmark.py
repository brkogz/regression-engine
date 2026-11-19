import numpy as np

from regression import Inference


def test_benchmark() -> None:
    """Verify end-to-end OLS estimation, ANOVA decomposition, inference, and prediction intervals against an analytical reference dataset."""
    x = np.array([2, 3, 5, 6, 7, 10, 7, 8])
    y = np.array([1, 2, 3, 5, 6, 7, 8, 8])
    alpha = 0.05
    notes_inference = Inference(x, y, alpha)
    lower, upper = notes_inference.predict_interval(np.array([9]))
    assert np.isclose(notes_inference.ols.b0, -0.625, rtol=0, atol=1e-12)
    assert np.isclose(notes_inference.ols.b1, 0.9375, rtol=0, atol=1e-12)
    assert np.isclose(notes_inference.ols.pop_std_error, 1.1075, rtol=0, atol=5e-5)
    assert np.isclose(notes_inference.ols.std_error, 1.2788, rtol=0, atol=5e-5)

    assert np.isclose(notes_inference.correlation.r, 0.9007, rtol=0, atol=5e-5)
    assert np.isclose(notes_inference.correlation.r_squared, 0.811, rtol=0, atol=5e-4)
    assert np.isclose(notes_inference.stats.dev_y2, 52, rtol=0, atol=1e-12)
    assert np.isclose(notes_inference.ols.residual_sum2, 9.8125, rtol=0, atol=1e-12)
    assert np.isclose(notes_inference.stats.dev_y2 - notes_inference.ols.residual_sum2, 42.1875, rtol=0, atol=1e-12)

    assert np.isclose(notes_inference.b0_std_error, 1.196, rtol=0, atol=5e-4)
    assert np.isclose(notes_inference.b1_std_error, 0.185, rtol=0, atol=5e-4)
    assert np.isclose(notes_inference.t_stat, 5.079, rtol=0, atol=5e-4)
    assert np.isclose(notes_inference.f_stat, 25.796, rtol=0, atol=5e-4)
    assert np.isclose(notes_inference.critical_value, 2.447, rtol=0, atol=5e-4)
    assert np.isclose(notes_inference.f_critical, 5.99, rtol=0, atol=5e-3)
    assert notes_inference.is_significant is True

    assert np.allclose(notes_inference.ols.predict(np.array([9])), 7.8125, rtol=0, atol=1e-12)
    assert np.allclose((lower[0], upper[0]), (4.23, 11.40), rtol=0, atol=5e-3)