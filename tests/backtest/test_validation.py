import pandas as pd

from python.backtest.validation import walk_forward_split


def test_walk_forward_split():
    dates = pd.date_range("2020-01-01", periods=1000, freq="B")
    df = pd.DataFrame({"value": range(1000)}, index=dates)

    splits = list(walk_forward_split(df, n_splits=5, train_pct=0.6, embargo_days=5))
    assert len(splits) == 5

    for train_idx, test_idx in splits:
        # No overlap
        assert len(set(train_idx) & set(test_idx)) == 0
        # Train comes before test
        assert df.index[train_idx[-1]] < df.index[test_idx[0]]
        # Embargo gap exists
        gap = (df.index[test_idx[0]] - df.index[train_idx[-1]]).days
        assert gap >= 5
