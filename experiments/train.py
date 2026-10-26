import os
import joblib
import pandas as pd
from sqlalchemy import create_engine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

engine = create_engine(
    "postgresql+psycopg2://sentinel:sentinel@localhost:5432/olist"
)

try:
    df = pd.read_sql("SELECT * FROM delivered_orders", engine)
finally:
    engine.dispose()

df = df.dropna(
    subset=["seller_state", "promised_days", "total_price", "is_late"]
).copy()

df["purchase_ts"] = pd.to_datetime(df["purchase_ts"])
df = df.sort_values("purchase_ts").reset_index(drop=True)

# Features available at purchase time
df["purchase_month"] = df["purchase_ts"].dt.month
df["purchase_dow"] = df["purchase_ts"].dt.dayofweek
df["purchase_hour"] = df["purchase_ts"].dt.hour
df["same_state"] = (
    df["customer_state"] == df["seller_state"]
).astype(int)
df["route"] = df["seller_state"] + "_" + df["customer_state"]

num = [
    "n_items", "n_sellers", "total_price", "total_freight",
    "promised_days", "purchase_month", "purchase_dow",
    "purchase_hour", "same_state",
]
cat = ["customer_state", "seller_state", "route"]

X = pd.get_dummies(df[num + cat], columns=cat, dtype=int)
y = df["is_late"].astype(int)

# Chronological split: oldest 80% for training, newest 20% for testing
cut = int(len(df) * 0.8)
X_train, X_test = X.iloc[:cut].copy(), X.iloc[cut:].copy()
y_train, y_test = y.iloc[:cut], y.iloc[cut:]

# Keep the test feature columns consistent with training.
X_train, X_test = X_train.align(
    X_test, join="left", axis=1, fill_value=0
)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print(
    f"Train={len(X_train)}, Test={len(X_test)}, "
    f"late rate train={y_train.mean():.3f}, "
    f"test={y_test.mean():.3f}"
)

def report(name, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)
    print(
        f"{name} (threshold={threshold:.2f}): "
        f"ROC-AUC={roc_auc_score(y_test, probabilities):.3f}, "
        f"PR-AUC={average_precision_score(y_test, probabilities):.3f}, "
        f"precision={precision_score(y_test, predictions, zero_division=0):.3f}, "
        f"recall={recall_score(y_test, predictions, zero_division=0):.3f}"
    )

# Baseline model
baseline = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=1000, class_weight="balanced"
    )),
])
baseline.fit(X_train, y_train)
baseline_probabilities = baseline.predict_proba(X_test)[:, 1]
report("Baseline", baseline_probabilities)

# XGBoost model
late_count = int(y_train.sum())
on_time_count = len(y_train) - late_count

if late_count == 0 or on_time_count == 0:
    raise ValueError("Training data must contain both classes.")

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=on_time_count / late_count,
    eval_metric="aucpr",
    random_state=42,
    n_jobs=2,
)
model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)[:, 1]
report("XGBoost", probabilities, threshold=0.5)
report("XGBoost", probabilities, threshold=0.7)

os.makedirs("models", exist_ok=True)
joblib.dump(
    {"model": model, "columns": list(X_train.columns)},
    "models/xgb_late.joblib",
)
print("Saved models/xgb_late.joblib")
