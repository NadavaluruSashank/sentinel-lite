import json, os, joblib
import numpy as np, pandas as pd
from sqlalchemy import create_engine
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
from xgboost import XGBClassifier

engine = create_engine("postgresql+psycopg2://sentinel:sentinel@localhost:5432/olist")
SQL = """
WITH geo AS (
  SELECT geolocation_zip_code_prefix AS z, AVG(geolocation_lat) AS lat, AVG(geolocation_lng) AS lng
  FROM geolocation GROUP BY 1),
first_item AS (
  SELECT DISTINCT ON (order_id) order_id, seller_id FROM order_items ORDER BY order_id, order_item_id),
w AS (
  SELECT oi.order_id, SUM(p.product_weight_g) AS weight_g,
         SUM(p.product_length_cm*p.product_height_cm*p.product_width_cm) AS vol_cm3
  FROM order_items oi JOIN products p USING (product_id) GROUP BY oi.order_id)
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
df = pd.read_sql(SQL, engine).sort_values("purchase_ts").reset_index(drop=True)

def haversine(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, [la1, lo1, la2, lo2])
    a = np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return 6371 * 2 * np.arcsin(np.sqrt(a))

df["distance_km"] = haversine(df.s_lat, df.s_lng, df.c_lat, df.c_lng)
df["purchase_dow"] = df.purchase_ts.dt.dayofweek
df["purchase_hour"] = df.purchase_ts.dt.hour
df["same_state"] = (df.customer_state == df.seller_state).astype(int)
df["freight_ratio"] = df.total_freight / (df.total_price + 1)

n = len(df)
a, b = int(n * 0.5), int(n * 0.8)
hist, tr, te = df.iloc[:a], df.iloc[a:b].copy(), df.iloc[b:].copy()
prior, k = hist.is_late.mean(), 10
g = hist.groupby("seller_id").is_late.agg(["sum", "count"])
tr = tr.join(g, on="seller_id")
te = te.join(g, on="seller_id")
for d in (tr, te):
    d["seller_late_rate"] = (d["sum"].fillna(0) + prior * k) / (d["count"].fillna(0) + k)
    d["seller_orders"] = d["count"].fillna(0)

num = ["n_items","n_sellers","total_price","total_freight","promised_days",
       "purchase_dow","purchase_hour","same_state","distance_km","weight_g","vol_cm3",
       "freight_ratio","seller_late_rate","seller_orders"]
cat = ["customer_state", "seller_state"]
both = pd.get_dummies(pd.concat([tr[num+cat], te[num+cat]]), columns=cat)
Xtr, Xte = both.iloc[:len(tr)], both.iloc[len(tr):]
ytr, yte = tr.is_late, te.is_late
base_rate = yte.mean()
print(f"history={len(hist)} train={len(tr)} test={len(te)} late rate: hist={hist.is_late.mean():.3f} train={ytr.mean():.3f} test={base_rate:.3f}")

def metrics(name, p, thr):
    pred = (p >= thr).astype(int)
    kk = int(0.05 * len(yte)); idx = np.argsort(-p)[:kk]
    top = yte.values[idx]
    m = dict(model=name, threshold=thr,
             roc_auc=round(float(roc_auc_score(yte, p)), 3),
             pr_auc=round(float(average_precision_score(yte, p)), 3),
             precision=round(float(precision_score(yte, pred, zero_division=0)), 3),
             recall=round(float(recall_score(yte, pred)), 3),
             top5pct_precision=round(float(top.mean()), 3),
             top5pct_recall=round(float(top.sum() / yte.sum()), 3),
             top5pct_lift=round(float(top.mean() / base_rate), 2))
    print(m); return m

med = Xtr.median()
lr = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))])
lr.fit(Xtr.fillna(med), ytr)
p_lr = lr.predict_proba(Xte.fillna(med))[:, 1]

spw = (ytr == 0).sum() / (ytr == 1).sum()
xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03, subsample=0.8,
                    colsample_bytree=0.8, min_child_weight=20, scale_pos_weight=spw,
                    eval_metric="aucpr", random_state=42)
xgb.fit(Xtr, ytr)
p_xgb = xgb.predict_proba(Xte)[:, 1]

res = [metrics("Logistic regression (final)", p_lr, 0.5),
       metrics("XGBoost (compared)", p_xgb, 0.5)]

cut95 = float(np.quantile(p_lr, 0.95))
os.makedirs("models", exist_ok=True)
json.dump(res, open("models/metrics.json", "w"), indent=2)
joblib.dump({"model": lr, "columns": list(Xtr.columns), "medians": med.to_dict(),
             "alert_threshold": cut95}, "models/late_risk_lr.joblib")
print("LR top-5% score cutoff:", round(cut95, 3))
print("saved models/late_risk_lr.joblib and models/metrics.json")
