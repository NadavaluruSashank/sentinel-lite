"""Pure feature and monitoring logic, kept free of database/model I/O so it can be unit-tested."""

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371


def haversine(la1, lo1, la2, lo2):
    """Great-circle distance in km between two lat/lng points (scalars or arrays)."""
    la1, lo1, la2, lo2 = map(np.radians, [la1, lo1, la2, lo2])
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a))


def add_order_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived features shared by training and the scoring API."""
    df = df.copy()
    df["same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)
    df["freight_ratio"] = df["total_freight"] / (df["total_price"] + 1)
    return df


def to_model_matrix(df: pd.DataFrame, columns, medians: pd.Series) -> pd.DataFrame:
    """One-hot encode states and align to the training columns, filling gaps with training medians."""
    X = pd.get_dummies(df, columns=["customer_state", "seller_state"])
    return X.reindex(columns=columns, fill_value=0).astype(float).fillna(medians)


def weekly_alerts(df: pd.DataFrame, window: int = 8, n_std: float = 3.0, min_spread: float = 1.0) -> pd.DataFrame:
    """Flag weeks whose late rate exceeds the previous `window`-week mean + n_std * std.

    Expects a `late_pct` column ordered by week. The current week is excluded from its own baseline.
    """
    df = df.copy()
    prev = df["late_pct"].shift(1)
    df["baseline"] = prev.rolling(window, min_periods=window // 2).mean()
    df["spread"] = prev.rolling(window, min_periods=window // 2).std()
    df["alert"] = df["late_pct"] > df["baseline"] + n_std * df["spread"].clip(lower=min_spread)
    return df
