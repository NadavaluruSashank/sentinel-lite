from typing import Optional
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

bundle = joblib.load("models/late_risk_lr.joblib")
model, cols = bundle["model"], bundle["columns"]
medians = pd.Series(bundle["medians"])
threshold = bundle["alert_threshold"]

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
    distance_km: Optional[float] = None
    weight_g: Optional[float] = None
    vol_cm3: Optional[float] = None
    seller_late_rate: Optional[float] = None
    seller_orders: float = 0

@app.get("/health")
def health():
    return {"status": "ok", "alert_threshold": round(threshold, 3)}

@app.post("/score")
def score(o: Order):
    f = o.model_dump()
    f["same_state"] = int(o.customer_state == o.seller_state)
    f["freight_ratio"] = o.total_freight / (o.total_price + 1)
    X = pd.get_dummies(pd.DataFrame([f]), columns=["customer_state", "seller_state"])
    X = X.reindex(columns=cols, fill_value=0).astype(float).fillna(medians)
    risk = float(model.predict_proba(X)[0, 1])
    return {"late_risk": round(risk, 3), "high_risk_alert": risk >= threshold}
