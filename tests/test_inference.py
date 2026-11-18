import numpy as np
import scipy.stats as sp_stats
import pytest

from regression.inference import Inference


def test_inference() -> None:
    """Verify standard errors, test statistics, and prediction intervals on a hand-calculated dataset."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 9.0])
    alpha = 0.05
    x_input = np.array([4.0, 5.0, 6.0])
    inference_test = Inference(x, y, alpha)
    lower, upper = inference_test.predict_interval(x_input)
    expected_interval = sp_stats.t.interval(1 - alpha, inference_test.stats.n - 2, inference_test.ols.predict(x_input),
                                            np.sqrt([5.0, 8.75, 14.0]))
    assert np.isclose(inference_test.r_std_error, np.sqrt(3 / 52), atol=1e-12)
    assert np.isclose(inference_test.b0_std_error, np.sqrt(3.5), atol=1e-12)
    assert np.isclose(inference_test.b1_std_error, np.sqrt(0.75), atol=1e-12)
    assert np.isclose(inference_test.t_stat, 7 / np.sqrt(3), atol=1e-12)
    assert np.isclose(inference_test.correlation.r / inference_test.r_std_error, inference_test.t_stat, atol=1e-12)
    assert np.isclose(inference_test.f_stat, 49 / 3, atol=1e-12)
    assert np.isclose(inference_test.critical_value, sp_stats.t.ppf(1 - alpha / 2, 1), atol=1e-12)
    assert np.isclose(inference_test.f_critical, sp_stats.f.ppf(1 - alpha, 1, 1), atol=1e-12)
    assert inference_test.is_significant is False
    assert np.allclose((lower, upper), expected_interval, atol=1e-12)


def test_inference_matches_scipy() -> None:
    """Verify standard errors against scipy.stats.linregress and asymptotic normal critical value on a large-offset dataset."""
    rng = np.random.default_rng(0)
    x = 1e9 + rng.normal(size=1000)
    y = rng.normal(size=1000)
    alpha = 0.05
    model = Inference(x, y, alpha)
    expected = sp_stats.linregress(x, y)
    assert np.isclose(model.b1_std_error, expected.stderr, atol=1e-12)
    assert np.isclose(model.b0_std_error, expected.intercept_stderr, atol=1e-12)
    assert np.isclose(model.critical_value, sp_stats.norm.ppf(1 - alpha / 2), atol=1e-12)


@pytest.mark.parametrize(
    "x, y, alpha, msg",
    [
        (np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 9.0]), -0.5, "between 0 and 1"),
        (np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 9.0]), 1.5, "between 0 and 1"),
        (np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 9.0]), np.nan, "between 0 and 1"),
        (np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0]), 0.05, "perfect linear fit"),
        (np.array([1.0, 2.0]), np.array([2.0, 6.0]), 0.05, "at least three"),
    ],
)
def test_inference_invalid_inputs(x, y, alpha, msg) -> None:
    """Verify that Inference raises ValueError with the expected message for invalid inputs."""
    with pytest.raises(ValueError, match=msg):
        Inference(x, y, alpha)