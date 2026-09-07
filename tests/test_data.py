import numpy as np
import pandas as pd

from market_volatility.data import chronological_split, final_training_set


def sample_dataset():
    index = pd.bdate_range("2015-01-01", "2022-12-31")
    return pd.DataFrame({"value": np.arange(len(index))}, index=index)


def test_chronological_split_orders_and_purges_boundaries():
    dataset = sample_dataset()
    splits = chronological_split(
        dataset,
        validation_start="2017-01-01",
        test_start="2020-01-01",
        purge_horizon=20,
    )
    train_candidates = dataset.loc[dataset.index < "2017-01-01"]
    validation_candidates = dataset.loc[
        (dataset.index >= "2017-01-01") & (dataset.index < "2020-01-01")
    ]
    assert len(splits.train) == len(train_candidates) - 20
    assert len(splits.validation) == len(validation_candidates) - 20
    assert splits.train.index.max() < splits.validation.index.min()
    assert splits.validation.index.max() < splits.test.index.min()


def test_final_training_set_is_purged_before_test():
    dataset = sample_dataset()
    result = final_training_set(dataset, test_start="2020-01-01", purge_horizon=20)
    expected = dataset.loc[dataset.index < "2020-01-01"].iloc[:-20]
    pd.testing.assert_frame_equal(result, expected)
