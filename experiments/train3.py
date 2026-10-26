import json
import os
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
from xgboost import XGBClassifier

engine = create_engine("postgresql+psycopg2://sentinel:sentinel@localhost:5432/olist")

SQL = """
WITH geo AS (
    SELECT geolocation_zip_code_prefix AS z, AVG(geolocation_lat) AS lat,
           AVG(geolocation_lng) AS lng
    FROM geolocation GROUP BY 1
),
first_item AS (
    SELECT DISTINCT ON (order_id) order_id, seller_id
    FROM order_items ORDER BY order_id, order_item_id
),
w AS (
    SELECT oi.order_id, SUM(p.product_weight_g) AS weight_g,
           SUM(p.product_length_cm*p.product_height_cm*p.product_width_cm) AS vol_cm3
    FROM order_items oi JOIN products p USING (product_id) GROUP BY oi.order_id
)
SELECT d.*, fi.seller_id, w.weight_g, w.vol_cm3,
       gc.lat AS c_lat, gc.lng AS c_lng, gs.lat AS s_lat, gs.lng AS s_lng
FROM delivered_orders d
JOIN orders o ON o.order_id = d.order_id
JOIN customers c ON c.customer_id = o.customer_id
JOIN first_item fi ON fi.order_id = d.order_id
JOIN sellers s ON s.seller_id = fi.seller_id
LEFT JOIN w ON w.order_id = d.order_id
LEFT JOIN geo gc ON gc.z = c.customer_zip_code_prefix
LEFT JOIN geo gs ON gs.z = s.seller_zip_code_prefix
"""

df = pd.read_sql(SQL, engine)
df["purchase_ts"] = pd.to_datetime(df["purchase_ts"], errors="coerce")
df = df.dropna(subset=["purchase_ts", "is_late"]).sort_values("purchase_ts").reset_index(drop=True)

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

df["distance_km"] = haversine(df["s_lat"], df["s_lng"], df["c_lat"], df["c_lng"])
df["purchase_month"] = df["purchase_ts"].dt.month
df["purchase_dow"] = df["purchase_ts"].dt.dayofweek
df["purchase_hour"] = df["purchase_ts"].dt.hour
df["same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)
df["freight_ratio"] = df["total_freight"] / (df["total_price"] + 1)

# Chronological blocks: historical period -> training period -> test period.
n = len(df)
a, b = int(n * 0.5), int(n * 0.8)
hist, tr, te = df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()
prior, k = float(hist["is_late"].mean()), 10
seller_history = hist.groupby("seller_id")["is_late"].agg(["sum", "count"])

for part in (tr, te):
    sums = part["seller_id"].map(seller_history["sum"])
    counts = part["seller_id"].map(seller_history["count"])
    part["seller_late_rate"] = (sums.fillna(0) + prior*k) / (counts.fillna(0) + k)
    part["seller_orders"] = counts.fillna(0)

num = ["n_items", "n_sellers", "total_price", "total_freight", "promised_days",
       "purchase_month", "purchase_dow", "purchase_hour", "same_state",
       "distance_km", "weight_g", "vol_cm3", "freight_ratio",
       "seller_late_rate", "seller_orders"]
cat = ["customer_state", "seller_state"]
combined = pd.get_dummies(pd.concat([tr[num+cat], te[num+cat]]), columns=cat, dtype=int)
Xtr, Xte = combined.iloc[:len(tr)].copy(), combined.iloc[len(tr):].copy()
ytr, yte = tr["is_late"].astype(int), te["is_late"].astype(int)
base_rate = float(yte.mean())

print(f"history={len(hist)} train={len(tr)} test={len(te)} late rate: hist={hist['is_late'].mean():.3f} train={ytr.mean():.3f} test={base_rate:.3f}")

def evaluate(name, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    top_n = max(1, int(0.05 * len(yte)))
    top_labels = yte.to_numpy()[np.argsort(-probabilities)[:top_n]]
    top_precision = float(top_labels.mean())
    top_recall = float(top_labels.sum() / max(1, int(yte.sum())))
    result = {
        "model": name,
        "threshold": threshold,
        "roc_auc": round(float(roc_auc_score(yte, probabilities)), 3),
        "pr_auc": round(float(average_precision_score(yte, probabilities)), 3),
        "precision": round(float(precision_score(yte, predictions, zero_division=0)), 3),
        "recall": round(float(recall_score(yte, predictions)), 3),
        "top5pct_precision": round(top_precision, 3),
        "top5pct_recall": round(top_recall, 3),
        "top5pct_lift": round(float(top_precision/base_rate) if base_rate else 0.0, 2),
    }
    print(result)
    return result

results = []
medians = Xtr.median(numeric_only=True)
Xtr_lr, Xte_lr = Xtr.fillna(medians).fillna(0), Xte.fillna(medians).fillna(0)
lr = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
])
lr.fit(Xtr_lr, ytr)
results.append(evaluate("Logistic regression", lr.predict_proba(Xte_lr)[:, 1], 0.5))

positive_count, negative_count = int((ytr == 1).sum()), int((ytr == 0).sum())
spw = negative_count / max(1, positive_count)
xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                    scale_pos_weight=spw, eval_metric="aucpr", random_state=42)
xgb.fit(Xtr, ytr)
p = xgb.predict_proba(Xte)[:, 1]
results.append(evaluate("XGBoost", p, 0.5))
results.append(evaluate("XGBoost", p, 0.7))

os.makedirs("models", exist_ok=True)
with open("models/metrics.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
joblib.dump({"model": xgb, "columns": list(Xtr.columns)}, "models/xgb_late_v3.joblib")
print("Saved metrics to models/metrics.json")
print("Saved model to models/xgb_late_v3.joblib")
