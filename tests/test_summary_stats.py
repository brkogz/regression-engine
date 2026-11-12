from regression.summary_stats import SummaryStatistics
import pytest
import numpy as np

def test_summary_stats()-> None:
    """Verify summary statistics and algebraic identities on a hand-calculated dataset."""
    x = np.array([1.0,2.0,3.0])
    y = np.array([2.0,4.0,9.0])
    stats = SummaryStatistics(x,y)
    assert stats.n==3
    assert np.isclose(stats.sum_x, 6, atol=1e-6)
    assert np.isclose(stats.sum_y, 15, atol=1e-6)
    assert np.isclose(stats.sum_xy, 37, atol=1e-6)
    assert np.isclose(stats.mean_x, 2, atol=1e-6)
    assert np.isclose(stats.mean_y, 5, atol=1e-6)
    assert np.isclose(stats.sum_x2, 14, atol=1e-6)
    assert np.isclose(stats.sum_y2, 101, atol=1e-6)
    assert np.allclose(stats.dev_x, [-1,0,1], atol=1e-6)
    assert np.allclose(stats.dev_y, [-3,-1,4], atol=1e-6)
    assert np.isclose(stats.dev_x2, 2, atol=1e-6)
    assert np.isclose(stats.dev_y2, 26, atol=1e-6)
    assert np.isclose(stats.dev_xy, 7, atol=1e-6)
    assert np.isclose(stats.dev_x2, (stats.sum_x2 -   (stats.sum_x * stats.sum_x)/stats.n), atol=1e-6)
    assert np.isclose(stats.dev_xy,(stats.sum_xy - ((stats.sum_x * stats.sum_y)/stats.n)), atol=1e-6 )
    assert np.isclose(np.sum(stats.dev_x), 0.0, atol=1e-12)

@pytest.mark.parametrize(
    "x, y, msg",
    [
        ([[1.0, 2.0, 3.0]], [2.0, 4.0, 9.0], "1 dimension"),
        ([1.0, 2.0, 3.0, 5.0], [2.0, 4.0, 9.0], "same size"),
        ([1.0, 2.0], [2.0, 4.0], "at least three"),
        ([1.0, 2.0, np.nan], [2.0, 4.0, 9.0], "finite"),
        ([1.0, 2.0, np.inf], [2.0, 4.0, 9.0], "finite"),
        ([1.0, 2.0, 3.0], [2.0, np.nan, 9.0], "finite"),
        ([2.0, 2.0, 2.0], [2.0, 4.0, 9.0], "constant"),
    ],
)
def test_invalid_inputs(x, y, msg) -> None:
    """Each invalid input raises ValueError with the expected message."""
    x_arr = np.array(x)
    y_arr = np.array(y)
    with pytest.raises(ValueError, match=msg):
        SummaryStatistics(x_arr, y_arr)

def test_inputs_are_copied_and_read_only() -> None:
    """Stored arrays are independent copies and cannot be modified."""
    x = np.array([1.5,2.0,3.0])
    y = np.array([2.0,4.0,9.0])
    stats = SummaryStatistics(x,y)
    x[0] = 87
    assert stats.x[0] == 1.5
    assert not stats.x.flags.writeable
    assert not stats.y.flags.writeable
    assert not stats.dev_x.flags.writeable
    assert not stats.dev_y.flags.writeable
    with pytest.raises(ValueError):
        stats.x[0] = 8

def test_small_scale_x_is_accepted() -> None:
    """Tiny but non-constant x (e.g. returns ~1e-5) must not be rejected."""
    x = np.array([1e-5, 2e-5, 3e-5])
    y = np.array([1.0, 2.0, 4.0])
    stats = SummaryStatistics(x, y)
    assert np.isclose(stats.dev_x2, 2e-10, atol=1e-15)

def test_dev_x2_stable_with_large_offset() -> None:
    """Centered sum of squares stays accurate where the raw-sum identity fails."""
    rng = np.random.default_rng(0)
    x = 1e9 + rng.normal(size=1000)
    y = rng.normal(size=1000)
    stats = SummaryStatistics(x, y)
    expected = np.var(x) * stats.n
    naive = stats.sum_x2 - stats.sum_x**2 / stats.n
    assert np.isclose(stats.dev_x2, expected)
    assert not np.isclose(naive, expected, rtol=1e-3)