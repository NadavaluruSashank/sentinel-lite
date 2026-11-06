import os
from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.features import add_order_features, to_model_matrix

MODEL_PATH = os.getenv("SENTINEL_MODEL_PATH", "models/late_risk_lr.joblib")


@lru_cache(maxsize=1)
def get_bundle():
    """Load the trained model bundle once, on first request (not at import time)."""
    bundle = joblib.load(os.getenv("SENTINEL_MODEL_PATH", MODEL_PATH))
    bundle["medians"] = pd.Series(bundle["medians"])
    return bundle


app = FastAPI(title="Sentinel late-delivery risk API")


class Order(BaseModel):
    customer_state: str
    seller_state: str
    n_items: int = 1
    n_sellers: int = 1
    total_price: float
    total_freight: float
    promised_days: float
    purchase_dow: int
    purchase_hour: int
    distance_km: float | None = None
    weight_g: float | None = None
    vol_cm3: float | None = None
    seller_late_rate: float | None = None
    seller_orders: float = 0


@app.get("/health")
def health():
    return {"status": "ok", "alert_threshold": round(get_bundle()["alert_threshold"], 3)}


@app.post("/score")
def score(o: Order):
    b = get_bundle()
    df = add_order_features(pd.DataFrame([o.model_dump()]))
    X = to_model_matrix(df, b["columns"], b["medians"])
    risk = float(b["model"].predict_proba(X)[0, 1])
    return {"late_risk": round(risk, 3), "high_risk_alert": risk >= b["alert_threshold"]}
