from regression.correlation import Correlation
import numpy as np
import pytest
import scipy.stats as stats


def test_correlation() -> None:
    """Verify Pearson correlation and variance decomposition metrics on a hand-calculated dataset."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 9.0])
    correlation_test = Correlation(x, y)
    assert np.isclose(correlation_test.total_variation, 26, atol=1e-12)
    assert np.isclose(correlation_test.r, 7 / np.sqrt(52), atol=1e-12)
    assert np.isclose(correlation_test.r_squared, 49 / 52, atol=1e-12)
    assert np.isclose(correlation_test.explained_variation, 24.5, atol=1e-12)
    assert np.isclose(correlation_test.unexplained_variation, 1.5, atol=1e-12)
    assert np.isclose(correlation_test.unexplained_ratio, 3.0 / 52.0, atol=1e-12)


@pytest.mark.parametrize(
    "y, msg",
    [
        (np.array([5.0, 5.0, 5.0]), "zero variance"),
    ],
)
def test_correlation_invalid_inputs(y, msg) -> None:
    """Verify that Correlation raises ValueError when the dependent variable has zero variance."""
    x = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match=msg):
        Correlation(x, y)


def test_correlation_matches_scipy() -> None:
    """Verify Pearson correlation coefficient against scipy.stats.linregress on a large-offset dataset."""
    rng = np.random.default_rng(0)
    x = 1e9 + rng.normal(size=1000)
    y = rng.normal(size=1000)
    model = Correlation(x, y)
    expected = stats.linregress(x, y)
    assert np.isclose(model.r, expected.rvalue, atol=1e-12)