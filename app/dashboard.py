import pandas as pd
import json
from pathlib import Path
import streamlit as st
from sqlalchemy import create_engine, text

st.set_page_config(
    page_title="Sentinel | Delivery Intelligence",
    page_icon="📦",
    layout="wide",
)

st.title("📦 Sentinel — Delivery Intelligence")
st.caption("E-commerce delivery performance and late-order risk analytics")

@st.cache_resource
def get_engine():
    return create_engine(
        "postgresql+psycopg2://sentinel:sentinel@localhost:5432/olist"
    )

@st.cache_data(ttl=300)
def load_kpis():
    query = """
        SELECT
            COUNT(*) AS delivered_orders,
            ROUND(100.0 * AVG(is_late), 2) AS late_pct,
            ROUND(AVG(delivery_days)::numeric, 2) AS avg_delivery_days,
            ROUND(AVG(promised_days)::numeric, 2) AS avg_promised_days
        FROM delivered_orders
    """
    with get_engine().connect() as connection:
        return pd.read_sql(text(query), connection)

@st.cache_data(ttl=300)
def load_regions():
    query = """
        SELECT
            customer_state,
            COUNT(*) AS orders,
            ROUND(100.0 * AVG(is_late), 2) AS late_pct
        FROM delivered_orders
        GROUP BY customer_state
        HAVING COUNT(*) >= 100
        ORDER BY late_pct DESC
    """
    with get_engine().connect() as connection:
        return pd.read_sql(text(query), connection)

try:
    kpis = load_kpis().iloc[0]
    regions = load_regions()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Delivered orders", f"{int(kpis['delivered_orders']):,}")
    c2.metric("Late deliveries", f"{kpis['late_pct']}%")
    c3.metric("Average delivery days", f"{kpis['avg_delivery_days']}")
    c4.metric("Average promised days", f"{kpis['avg_promised_days']}")

    st.divider()
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Late-delivery rate by customer state")
        chart_data = regions.set_index("customer_state")[["late_pct"]]
        st.bar_chart(chart_data)
        st.caption("States with at least 100 delivered orders, ranked by late rate.")

    with right:
        st.subheader("Regional performance")
        st.dataframe(
            regions.rename(columns={
                "customer_state": "State",
                "orders": "Delivered orders",
                "late_pct": "Late rate (%)",
            }),
            use_container_width=True,
            hide_index=True,
        )


    st.divider()
    st.subheader("Model evaluation")

    metrics_path = Path(__file__).resolve().parents[1] / "models" / "metrics.json"

    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as file:
            results = json.load(file)

        model_results = pd.DataFrame(results)
        st.dataframe(
            model_results,
            width="stretch",
            hide_index=True,
        )
    else:
        st.warning("No model metrics found. Run the training script first.")

except Exception as e:
    st.error(f"Dashboard error: {e}")
