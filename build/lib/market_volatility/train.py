"""Baselines and candidate forecasting models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from market_volatility.features import FEATURE_COLUMNS, TARGET_COLUMN


HAR_FEATURES = ["rv_5", "rv_20", "rv_60"]


@dataclass(frozen=True)
class ModelSpec:
    """An estimator together with the exact feature subset it uses."""

    name: str
    features: list[str]
    estimator: RegressorMixin


@dataclass
class FittedModel:
    """A fitted estimator that remembers its input columns."""

    name: str
    features: list[str]
    estimator: RegressorMixin

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        predictions = self.estimator.predict(frame[self.features])
        return np.clip(np.asarray(predictions, dtype=float), 1e-6, None)


def model_specs(random_state: int = 42) -> dict[str, ModelSpec]:
    """Return a small, deliberately diverse set of candidate models."""

    return {
        "har_linear": ModelSpec(
            name="har_linear",
            features=HAR_FEATURES,
            estimator=LinearRegression(),
        ),
        "ridge": ModelSpec(
            name="ridge",
            features=FEATURE_COLUMNS,
            estimator=make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        ),
        "random_forest": ModelSpec(
            name="random_forest",
            features=FEATURE_COLUMNS,
            estimator=RandomForestRegressor(
                n_estimators=400,
                max_features=0.7,
                min_samples_leaf=10,
                n_jobs=-1,
                random_state=random_state,
            ),
        ),
        "hist_gradient_boosting": ModelSpec(
            name="hist_gradient_boosting",
            features=FEATURE_COLUMNS,
            estimator=HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=300,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=random_state,
            ),
        ),
    }


def baseline_predictions(
    frame: pd.DataFrame,
    training_target_mean: float,
) -> pd.DataFrame:
    """Generate three forecasts requiring no fitted ML model."""

    return pd.DataFrame(
        {
            "baseline_train_mean": training_target_mean,
            "baseline_persistence_rv20": frame["rv_20"],
            "baseline_historical_rv60": frame["rv_60"],
        },
        index=frame.index,
    )


def fit_models(
    training_frame: pd.DataFrame,
    names: Iterable[str] | None = None,
    random_state: int = 42,
) -> dict[str, FittedModel]:
    """Fit requested candidate models using training observations only."""

    specs = model_specs(random_state=random_state)
    selected_names = list(specs) if names is None else list(names)
    unknown = set(selected_names).difference(specs)
    if unknown:
        raise KeyError(f"Unknown model names: {sorted(unknown)}")

    y_train = training_frame[TARGET_COLUMN]
    fitted: dict[str, FittedModel] = {}
    for name in selected_names:
        spec = specs[name]
        estimator = clone(spec.estimator)
        estimator.fit(training_frame[spec.features], y_train)
        fitted[name] = FittedModel(name, spec.features, estimator)
    return fitted


def predict_models(
    fitted_models: dict[str, FittedModel],
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return one aligned prediction column per fitted candidate."""

    return pd.DataFrame(
        {name: model.predict(frame) for name, model in fitted_models.items()},
        index=frame.index,
    )
