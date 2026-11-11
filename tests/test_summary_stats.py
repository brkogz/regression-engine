from regression.summary_stats import SummaryStatistics
import pytest
import numpy as np

def test_summary_stats()-> None:
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

def test_summary_stats_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        x = np.array([[1.0, 2.0, 3.0]])
        y = np.array([2.0, 4.0, 9.0])
        SummaryStatistics(x, y)
    with pytest.raises(ValueError):
        x = np.array([1.0, 2.0, 3.0,5.0])
        y = np.array([2.0, 4.0, 9.0])
        SummaryStatistics(x, y)
    with pytest.raises(ValueError):
        x = np.array([1.0, 2.0])
        y = np.array([2.0, 4.0])
        SummaryStatistics(x, y)
    with pytest.raises(ValueError):
        x = np.array([1.0, 2.0,np.nan])
        y = np.array([2.0, 4.0, 9.0])
        SummaryStatistics(x, y)
    with pytest.raises(ValueError):
        x = np.array([2.0, 2.0, 2.0])
        y = np.array([2.0, 4.0, 9.0])
        SummaryStatistics(x, y)







