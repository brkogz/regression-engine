from regression.simple_ols import SimpleOLS
import numpy as np
import scipy.stats as stats
import pytest


def test_simple_ols() -> None:
    """Verify OLS coefficients, fitted values, residuals, standard errors, and predictions on a hand-calculated dataset."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 9.0])
    x_new = np.array([4.0, 5.0, 6.0])
    simple_ols_test = SimpleOLS(x, y)
    assert np.isclose(simple_ols_test.b0, -2.0, atol=1e-12)
    assert np.isclose(simple_ols_test.b1, 3.5, atol=1e-12)
    assert np.allclose(simple_ols_test.y_hat, [1.5, 5.0, 8.5], atol=1e-12)
    assert not simple_ols_test.y_hat.flags.writeable
    assert np.allclose(simple_ols_test.residuals, [0.5, -1.0, 0.5], atol=1e-12)
    assert np.isclose(np.sum(simple_ols_test.residuals), 0, atol=1e-12)
    assert not simple_ols_test.residuals.flags.writeable
    assert np.isclose(simple_ols_test.residual_sum2, 1.5, atol=1e-12)
    assert np.isclose(simple_ols_test.std_error, np.sqrt(1.5), atol=1e-12)
    assert np.isclose(simple_ols_test.pop_std_error, np.sqrt(0.5), atol=1e-12)
    assert np.allclose(simple_ols_test.predict(x_new), [12.0, 15.5, 19.0], atol=1e-12)


def test_simple_ols_matches_scipy() -> None:
    """Verify OLS slope and intercept against scipy.stats.linregress on a large-offset dataset."""
    rng = np.random.default_rng(0)
    x = 1e9 + rng.normal(size=1000)
    y = rng.normal(size=1000)
    model = SimpleOLS(x, y)
    expected = stats.linregress(x, y)
    assert np.isclose(model.b0, expected.intercept, atol=1e-12)
    assert np.isclose(model.b1, expected.slope, atol=1e-12)


@pytest.mark.parametrize(
    "x_new, msg",
    [
        (np.array([[1.0, 2.0]]), "1 dimension"),
        (np.array([]), "at least one element"),
        (np.array([1.0, np.nan]), "finite"),
        (np.array([1.0, np.inf]), "finite")
    ],
)
def test_predict_invalid_inputs(x_new, msg) -> None:
    """Verify that predict raises ValueError with the expected message for invalid inputs."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 9.0])
    model = SimpleOLS(x, y)
    with pytest.raises(ValueError, match=msg):
        model.predict(x_new)