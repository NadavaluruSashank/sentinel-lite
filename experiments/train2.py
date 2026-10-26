import json
import os
import joblib
import numpy as np
import pandas as pd

from sqlalchemy import create_engine
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

DATABASE_URL = (
    "postgresql+psycopg2://sentinel:sentinel@localhost:5432/olist"
)

SQL = """
WITH geo AS (
    SELECT
        geolocation_zip_code_prefix AS zip_prefix,
        AVG(geolocation_lat) AS lat,
        AVG(geolocation_lng) AS lng
    FROM geolocation
    GROUP BY geolocation_zip_code_prefix
),
first_item AS (
    SELECT DISTINCT ON (order_id)
        order_id, seller_id
    FROM order_items
    ORDER BY order_id, order_item_id
),
product_features AS (
    SELECT
        oi.order_id,
        SUM(p.product_weight_g) AS weight_g,
        SUM(
            p.product_length_cm
            * p.product_height_cm
            * p.product_width_cm
        ) AS volume_cm3
    FROM order_items oi
    JOIN products p USING (product_id)
    GROUP BY oi.order_id
)
SELECT
    d.*,
    fi.seller_id AS feature_seller_id,
    pf.weight_g,
    pf.volume_cm3,
    gc.lat AS customer_lat,
    gc.lng AS customer_lng,
    gs.lat AS seller_lat,
    gs.lng AS seller_lng
FROM delivered_orders d
JOIN orders o USING (order_id)
JOIN customers c ON c.customer_id = o.customer_id
JOIN first_item fi ON fi.order_id = d.order_id
JOIN sellers s ON s.seller_id = fi.seller_id
LEFT JOIN product_features pf ON pf.order_id = d.order_id
LEFT JOIN geo gc
    ON gc.zip_prefix = c.customer_zip_code_prefix
LEFT JOIN geo gs
    ON gs.zip_prefix = s.seller_zip_code_prefix
"""


def haversine_km(lat1, lon1, lat2, lon2):
    """Approximate straight-line distance between two coordinates."""
    lat1, lon1, lat2, lon2 = [
        np.radians(np.asarray(v, dtype=float))
        for v in (lat1, lon1, lat2, lon2)
    ]
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin((lon2 - lon1) / 2) ** 2
    )
    a = np.clip(a, 0, 1)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


engine = create_engine(DATABASE_URL)
try:
    df = pd.read_sql(SQL, engine)
finally:
    engine.dispose()

df["purchase_ts"] = pd.to_datetime(df["purchase_ts"])
df = (
    df.dropna(subset=["purchase_ts", "is_late"])
    .sort_values("purchase_ts")
    .reset_index(drop=True)
)
df["is_late"] = df["is_late"].astype(int)

# Features that can be estimated from information available at purchase time.
df["distance_km"] = haversine_km(
    df["seller_lat"], df["seller_lng"],
    df["customer_lat"], df["customer_lng"],
)
df["purchase_month"] = df["purchase_ts"].dt.month
df["purchase_dow"] = df["purchase_ts"].dt.dayofweek
df["purchase_hour"] = df["purchase_ts"].dt.hour
df["same_state"] = (
    df["customer_state"] == df["seller_state"]
).astype(int)
df["freight_ratio"] = df["total_freight"] / (df["total_price"] + 1.0)

# Chronological split.
cut = int(len(df) * 0.8)
train = df.iloc[:cut].copy()
test = df.iloc[cut:].copy()

if train["is_late"].nunique() < 2 or test["is_late"].nunique() < 2:
    raise ValueError("Both train and test sets must contain both classes.")

# Seller history for each training row uses only prior rows for that seller.
# The smoothed global prior also uses only earlier training rows.
train["global_prior"] = (
    train["is_late"].shift(1).expanding().mean().fillna(0.08)
)
train["seller_previous_count"] = (
    train.groupby("feature_seller_id", sort=False).cumcount()
)
train["seller_previous_late"] = (
    train.groupby("feature_seller_id", sort=False)["is_late"]
    .transform(lambda values: values.shift(1).fillna(0).cumsum())
)
smoothing = 10.0
train["seller_late_rate"] = (
    train["seller_previous_late"] + train["global_prior"] * smoothing
) / (train["seller_previous_count"] + smoothing)

# Test seller history uses only outcomes from the training period.
global_train_rate = train["is_late"].mean()
seller_history = train.groupby("feature_seller_id")["is_late"].agg(
    seller_late_sum="sum",
    seller_order_count="count",
)
test = test.join(seller_history, on="feature_seller_id")
test["seller_late_sum"] = test["seller_late_sum"].fillna(0)
test["seller_order_count"] = test["seller_order_count"].fillna(0)
test["seller_late_rate"] = (
    test["seller_late_sum"] + global_train_rate * smoothing
) / (test["seller_order_count"] + smoothing)
test["seller_previous_count"] = test["seller_order_count"]

numeric_features = [
    "n_items", "n_sellers", "total_price", "total_freight",
    "promised_days", "purchase_month", "purchase_dow", "purchase_hour",
    "same_state", "distance_km", "weight_g", "volume_cm3",
    "freight_ratio", "seller_late_rate", "seller_previous_count",
]
categorical_features = ["customer_state", "seller_state"]
feature_columns = numeric_features + categorical_features

X_train = pd.get_dummies(
    train[feature_columns], columns=categorical_features, dtype=int
)
X_test = pd.get_dummies(
    test[feature_columns], columns=categorical_features, dtype=int
)
X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Impute missing numeric values using training-only medians.
medians = X_train[numeric_features].median().fillna(0)
X_train[numeric_features] = X_train[numeric_features].fillna(medians)
X_test[numeric_features] = X_test[numeric_features].fillna(medians)

y_train = train["is_late"]
y_test = test["is_late"]

print(
    f"Train={len(X_train)}, Test={len(X_test)}, "
    f"late rate train={y_train.mean():.3f}, "
    f"test={y_test.mean():.3f}"
)


def evaluate(name, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)
    result = {
        "model": name,
        "threshold": threshold,
        "roc_auc": round(roc_auc_score(y_test, probabilities), 4),
        "pr_auc": round(average_precision_score(y_test, probabilities), 4),
        "precision": round(
            precision_score(y_test, predictions, zero_division=0), 4
        ),
        "recall": round(recall_score(y_test, predictions, zero_division=0), 4),
    }
    print(result)
    return result


results = []

baseline = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42
    )),
])
baseline.fit(X_train, y_train)
baseline_probabilities = baseline.predict_proba(X_test)[:, 1]
results.append(evaluate("Logistic regression", baseline_probabilities, 0.5))

late_count = int(y_train.sum())
on_time_count = len(y_train) - late_count
model = XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.04,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=on_time_count / late_count,
    eval_metric="aucpr",
    random_state=42,
    n_jobs=2,
)
model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)[:, 1]

for threshold in (0.5, 0.7, 0.8):
    results.append(evaluate("XGBoost", probabilities, threshold))

os.makedirs("models", exist_ok=True)
with open("models/metrics.json", "w", encoding="utf-8") as file:
    json.dump(results, file, indent=2)

joblib.dump(
    {
        "model": model,
        "columns": list(X_train.columns),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "medians": medians.to_dict(),
    },
    "models/xgb_late_v2.joblib",
)
print("\nSaved models/metrics.json")
print("Saved models/xgb_late_v2.joblib")
