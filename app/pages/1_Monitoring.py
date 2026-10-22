import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

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

df = load()
prev = df["late_pct"].shift(1)
df["baseline"] = prev.rolling(8, min_periods=4).mean()
df["spread"] = prev.rolling(8, min_periods=4).std()
df["alert"] = df["late_pct"] > df["baseline"] + 3 * df["spread"].clip(lower=1.0)

c1, c2, c3 = st.columns(3)
c1.metric("Weeks analysed", len(df))
c2.metric("Alert weeks", int(df["alert"].sum()))
c3.metric("Worst week late rate", f"{df['late_pct'].max():.1f}%")

st.line_chart(df.set_index("week")[["late_pct", "baseline"]])

st.subheader("Alert weeks")
alerts = df[df["alert"]][["week", "orders", "late_pct", "baseline"]].round(1)
st.dataframe(alerts, width="stretch", hide_index=True)
st.info("Alert rule: weekly late rate above the previous 8-week average plus 3 standard deviations (minimum spread 1 point).")
