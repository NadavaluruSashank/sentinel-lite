"""API tests using a tiny stand-in model, so CI needs neither Postgres nor the real trained model."""

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app import api

COLUMNS = ["distance_km", "promised_days", "same_state", "customer_state_SP", "seller_state_SP"]


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Toy data where long distances are late, so the score direction is predictable
    rng = np.random.default_rng(0)
    dist = rng.uniform(0, 3000, 400)
    X = pd.DataFrame({
        "distance_km": dist,
        "promised_days": rng.uniform(5, 30, 400),
        "same_state": (dist < 300).astype(float),
        "customer_state_SP": (dist < 300).astype(float),
        "seller_state_SP": 1.0,
    })
    y = (dist > 1500).astype(int)
    model = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))]).fit(X, y)

    path = tmp_path / "model.joblib"
    joblib.dump({"model": model, "columns": COLUMNS,
                 "medians": X.median().to_dict(), "alert_threshold": 0.5}, path)
    monkeypatch.setenv("SENTINEL_MODEL_PATH", str(path))
    api.get_bundle.cache_clear()
    yield TestClient(api.app)
    api.get_bundle.cache_clear()


ORDER = {
    "customer_state": "AL", "seller_state": "SP", "total_price": 120, "total_freight": 35,
    "promised_days": 20, "purchase_dow": 2, "purchase_hour": 14,
}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "alert_threshold": 0.5}


def test_score_returns_valid_risk(client):
    r = client.post("/score", json={**ORDER, "distance_km": 2200})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["late_risk"] <= 1.0
    assert isinstance(body["high_risk_alert"], bool)


def test_long_route_scores_higher_than_local(client):
    far = client.post("/score", json={**ORDER, "distance_km": 2800}).json()
    near = client.post("/score", json={**ORDER, "customer_state": "SP", "distance_km": 20}).json()
    assert far["late_risk"] > near["late_risk"]
    assert far["high_risk_alert"] and not near["high_risk_alert"]


def test_missing_optional_fields_use_medians(client):
    r = client.post("/score", json=ORDER)  # no distance_km
    assert r.status_code == 200


def test_invalid_payload_rejected(client):
    r = client.post("/score", json={"customer_state": "SP"})
    assert r.status_code == 422
