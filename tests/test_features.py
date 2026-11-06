import numpy as np
import pandas as pd
import pytest

from src.features import add_order_features, haversine, to_model_matrix, weekly_alerts


def test_haversine_zero_for_same_point():
    assert haversine(-23.55, -46.63, -23.55, -46.63) == pytest.approx(0.0)


def test_haversine_sao_paulo_to_rio():
    # São Paulo -> Rio de Janeiro is roughly 360 km as the crow flies
    assert haversine(-23.55, -46.63, -22.91, -43.17) == pytest.approx(360, abs=10)


def test_haversine_is_vectorised():
    d = haversine(np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, 0.0]))
    assert d[0] == pytest.approx(0.0)
    assert d[1] == pytest.approx(111.2, abs=0.5)  # one degree of latitude


def test_add_order_features():
    df = pd.DataFrame({
        "customer_state": ["SP", "RJ"],
        "seller_state": ["SP", "SP"],
        "total_price": [99.0, 0.0],
        "total_freight": [10.0, 5.0],
    })
    out = add_order_features(df)
    assert out["same_state"].tolist() == [1, 0]
    assert out["freight_ratio"].tolist() == pytest.approx([0.1, 5.0])
    assert "same_state" not in df.columns  # input not mutated


def test_to_model_matrix_aligns_columns_and_fills_medians():
    cols = ["total_price", "weight_g", "customer_state_SP", "customer_state_RJ", "seller_state_SP"]
    medians = pd.Series({"total_price": 50.0, "weight_g": 800.0})
    df = pd.DataFrame([{"total_price": 120.0, "weight_g": None, "customer_state": "AM", "seller_state": "SP"}])
    X = to_model_matrix(df, cols, medians)
    assert list(X.columns) == cols
    row = X.iloc[0]
    assert row["weight_g"] == 800.0          # missing value filled with training median
    assert row["customer_state_SP"] == 0.0   # unseen state "AM" maps to all zeros
    assert row["customer_state_RJ"] == 0.0
    assert row["seller_state_SP"] == 1.0


def test_weekly_alerts_flags_spike_only():
    rates = [5.0, 5.5, 4.5, 5.0, 5.2, 4.8, 5.1, 4.9, 20.0, 5.0]
    out = weekly_alerts(pd.DataFrame({"late_pct": rates}))
    assert out["alert"].tolist() == [False] * 8 + [True, False]


def test_weekly_alerts_needs_history():
    # fewer than 4 previous weeks -> no baseline -> never alert
    out = weekly_alerts(pd.DataFrame({"late_pct": [5.0, 5.0, 50.0]}))
    assert out["baseline"].isna().all()
    assert not out["alert"].any()


def test_weekly_alerts_min_spread_prevents_alerts_on_flat_history():
    # perfectly flat history has std 0; the 1-point floor stops a tiny bump from alerting
    out = weekly_alerts(pd.DataFrame({"late_pct": [5.0] * 8 + [7.5]}))
    assert not out["alert"].iloc[-1]
