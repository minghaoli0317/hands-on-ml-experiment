import numpy as np
import pandas as pd

from market_volatility.evaluate import forecast_metrics, qlike_loss


def test_perfect_forecast_has_zero_losses():
    actual = pd.Series([0.1, 0.2, 0.3])
    predictions = pd.DataFrame({"perfect": actual})
    metrics = forecast_metrics(actual, predictions)
    assert np.isclose(metrics.loc["perfect", "MAE"], 0.0)
    assert np.isclose(metrics.loc["perfect", "RMSE"], 0.0)
    assert np.isclose(metrics.loc["perfect", "QLIKE"], 0.0)
    assert np.isclose(qlike_loss(actual, actual), 0.0)
