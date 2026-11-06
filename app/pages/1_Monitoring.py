import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

from src.features import weekly_alerts

st.title("Delivery performance monitoring")
st.caption("Weekly late-delivery rate with an automatic alert when it rises well above its recent level.")

@st.cache_data
def load():
    engine = create_engine("postgresql+psycopg2://sentinel:sentinel@localhost:5432/olist")
    q = """
    SELECT date_trunc('week', purchase_ts)::date AS week,
           COUNT(*) AS orders,
           100.0 * AVG(is_late) AS late_pct
    FROM delivered_orders GROUP BY 1 HAVING COUNT(*) >= 200 ORDER BY 1
    """
    return pd.read_sql(q, engine)

df = weekly_alerts(load())

c1, c2, c3 = st.columns(3)
c1.metric("Weeks analysed", len(df))
c2.metric("Alert weeks", int(df["alert"].sum()))
c3.metric("Worst week late rate", f"{df['late_pct'].max():.1f}%")

st.line_chart(df.set_index("week")[["late_pct", "baseline"]])

st.subheader("Alert weeks")
alerts = df[df["alert"]][["week", "orders", "late_pct", "baseline"]].round(1)
st.dataframe(alerts, width="stretch", hide_index=True)
st.info("Alert rule: weekly late rate above the previous 8-week average plus 3 standard deviations (minimum spread 1 point).")
