import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

import joblib

# torch imports for NN inference
import torch
import torch.nn as nn

load_dotenv("config/settings.env")

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

FEATURE_COLS = [
    "recency_days",
    "frequency",
    "monetary",
    "avg_delivery_days",
    "late_rate",
    "avg_review_score",
    "customer_state",
]

UPLOAD_REQUIRED_COLS = FEATURE_COLS[:]  # customer_unique_id optional


# -----------------------------
# DB + caching
# -----------------------------
@st.cache_resource
def get_engine():
    url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url)


@st.cache_resource
def load_model_rf():
    return joblib.load("models/churn_model.joblib")


@st.cache_data
def load_model_metrics_rf():
    path = "models/churn_metrics.json"
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Torch NN loader + helpers
# -----------------------------
class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  # logits


@st.cache_resource
def load_torch_bundle():
    bundle_path = "models/churn_torch_bundle.joblib"
    if not os.path.exists(bundle_path):
        return None

    bundle = joblib.load(bundle_path)
    preprocessor = bundle["preprocessor"]
    weights_path = bundle["model_state_dict_path"]
    input_dim = int(bundle["input_dim"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLP(input_dim).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    return {"preprocessor": preprocessor, "model": model, "device": device}


@st.cache_data
def load_torch_metrics():
    path = "models/churn_torch_metrics.json"
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def torch_predict_proba(torch_bundle, X_df: pd.DataFrame, batch_size: int = 8192) -> np.ndarray:
    """
    Batched inference to avoid GPU/CPU memory spikes.
    Returns probabilities shape (n_rows,).
    """
    pre = torch_bundle["preprocessor"]
    model = torch_bundle["model"]
    device = torch_bundle["device"]

    Xt = pre.transform(X_df)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    Xt = np.asarray(Xt, dtype=np.float32)

    probs = np.empty((Xt.shape[0],), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for start in range(0, Xt.shape[0], batch_size):
            end = min(start + batch_size, Xt.shape[0])
            x = torch.from_numpy(Xt[start:end]).to(device)
            logits = model(x)
            p = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
            probs[start:end] = p

    return probs


# -----------------------------
# Help / glossary
# -----------------------------
def render_glossary_and_help(threshold: float):
    with st.expander("Help / Glossary (How to read this dashboard)", expanded=False):
        st.markdown(
            f"""
### What is this dashboard?
This dashboard combines **business analytics** (RFM, cohorts, delivery, reviews, category sales)
with **AI/ML predictions** (reorder probability in the next 90 days).

### Key terms (simple)
- **Days since last purchase (recency_days)**: higher = customer inactive longer (often higher churn risk).
- **Total orders (frequency)**: higher = more loyal customer (often lower churn risk).
- **Total spend (monetary)**: higher = higher value customer (prioritize for retention).
- **Avg delivery days**: delivery speed.
- **Late delivery rate (late_rate, 0–1)**: share of orders delivered late. Example: 0.10 = 10% late.
- **Avg review score (1–5)**: satisfaction.

### AI prediction
- The model outputs a **probability** between **0 and 1**.
- We convert probability into a **risk band** using the selected threshold.

### Risk bands (used in this project)
- **High**: probability **>= threshold**
- **Medium**: probability **>= 0.5 × threshold** and < threshold
- **Low**: probability < 0.5 × threshold

Current threshold: **{threshold:.3f}**

### “Insights summary”
Converts predictions into decisions:
- where high-risk customers are (states)
- main drivers (late deliveries, low reviews, high recency)
- recommended action counts (retention outreach, delivery follow-up, win-back, upsell)
"""
        )


# -----------------------------
# Scoring + suggestions + UX
# -----------------------------
def risk_bucket(prob: float, thr: float) -> str:
    if prob >= thr:
        return "high"
    if prob >= (thr * 0.5):
        return "medium"
    return "low"


def risk_badge(bucket: str, prob: float, thr: float):
    label = f"Risk: {bucket.upper()} (p={prob:.4f}, threshold={thr:.3f})"
    if bucket == "high":
        st.error(label)
    elif bucket == "medium":
        st.warning(label)
    else:
        st.success(label)


def generate_suggestions(row: dict, prob: float, thr: float) -> list[str]:
    sugg = []
    bucket = risk_bucket(prob, thr)

    monetary = float(row.get("monetary") or 0)
    late_rate = float(row.get("late_rate") or 0)
    review = float(row.get("avg_review_score") or 0)
    recency = float(row.get("recency_days") or 0)
    freq = float(row.get("frequency") or 0)

    if bucket == "high":
        sugg.append("Retention outreach within 24–48 hours (WhatsApp/email/call).")
        if monetary >= 200:
            sugg.append("High value: offer targeted coupon/loyalty reward.")
        if late_rate >= 0.10:
            sugg.append("Delivery issue likely: apologize / compensate / investigate logistics.")
        if review and review <= 3.5:
            sugg.append("Low satisfaction: customer support follow-up.")
        if recency >= 120:
            sugg.append("Win-back: best-sellers + limited-time offer.")
    elif bucket == "medium":
        sugg.append("Nudge campaign: personalized recommendations + small incentive.")
        if review and review <= 3.5:
            sugg.append("Ask for feedback and resolve pain points.")
        if freq >= 3:
            sugg.append("Frequent buyer: upsell bundles / subscription-style offers.")
    else:
        sugg.append("Upsell/cross-sell, membership, and referral programs.")
        if freq >= 2:
            sugg.append("Recommend complementary products to increase AOV.")
    return sugg


def score_dataframe(model_choice: str, X: pd.DataFrame) -> np.ndarray:
    if model_choice == "RandomForest (joblib)":
        model = load_model_rf()
        return model.predict_proba(X)[:, 1]
    torch_bundle = load_torch_bundle()
    if torch_bundle is None:
        raise RuntimeError("Torch NN bundle not available (models/churn_torch_bundle.joblib missing).")
    return torch_predict_proba(torch_bundle, X)


def make_run_id(source: str, model_name: str, snapshot_date=None) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    snap = f"_{snapshot_date}" if snapshot_date else ""
    return f"{source}_{model_name}{snap}_{ts}"


def insert_predictions_to_db(
    engine,
    df_scored: pd.DataFrame,
    *,
    run_id: str,
    source: str,
    snapshot_date,
    model_name: str,
    threshold: float,
):
    for c in FEATURE_COLS:
        if c not in df_scored.columns:
            df_scored[c] = None

    if "customer_unique_id" not in df_scored.columns:
        df_scored["customer_unique_id"] = None

    rows = []
    for _, r in df_scored.iterrows():
        rows.append(
            {
                "run_id": run_id,
                "source": source,
                "snapshot_date": snapshot_date,
                "model_name": model_name,
                "threshold": float(threshold),
                "customer_unique_id": r.get("customer_unique_id"),
                "recency_days": r.get("recency_days"),
                "frequency": r.get("frequency"),
                "monetary": r.get("monetary"),
                "avg_delivery_days": r.get("avg_delivery_days"),
                "late_rate": r.get("late_rate"),
                "avg_review_score": r.get("avg_review_score"),
                "customer_state": r.get("customer_state"),
                "reorder_proba_90d": float(r.get("reorder_proba_90d")),
                "risk_bucket": r.get("risk_bucket"),
            }
        )

    stmt = text("""
        INSERT INTO mart.customer_churn_predictions_log (
          run_id, source, snapshot_date, model_name, threshold, customer_unique_id,
          recency_days, frequency, monetary, avg_delivery_days, late_rate, avg_review_score, customer_state,
          reorder_proba_90d, risk_bucket
        ) VALUES (
          :run_id, :source, :snapshot_date, :model_name, :threshold, :customer_unique_id,
          :recency_days, :frequency, :monetary, :avg_delivery_days, :late_rate, :avg_review_score, :customer_state,
          :reorder_proba_90d, :risk_bucket
        )
    """)

    with engine.begin() as conn:
        conn.execute(stmt, rows)


def compute_insights(scored_df: pd.DataFrame, thr: float) -> dict:
    df = scored_df.copy()

    for c in FEATURE_COLS + ["reorder_proba_90d", "risk_bucket"]:
        if c not in df.columns:
            df[c] = np.nan

    total = len(df)
    high = int((df["risk_bucket"] == "high").sum())
    med = int((df["risk_bucket"] == "medium").sum())
    low = int((df["risk_bucket"] == "low").sum())

    top_states = (
        df[df["risk_bucket"] == "high"]
        .assign(customer_state=lambda x: x["customer_state"].fillna("Unknown"))
        .groupby("customer_state")
        .size()
        .reset_index(name="high_risk_customers")
        .sort_values("high_risk_customers", ascending=False)
        .head(10)
    )

    df["flag_late_delivery"] = (df["late_rate"].fillna(0) >= 0.10)
    df["flag_low_reviews"] = (df["avg_review_score"].fillna(5) <= 3.5)
    df["flag_high_recency"] = (df["recency_days"].fillna(0) >= 120)
    df["flag_low_frequency"] = (df["frequency"].fillna(0) <= 1)
    df["flag_high_value"] = df["monetary"].fillna(0) >= float(df["monetary"].quantile(0.95) if total else 0)

    flags = {
        "Late delivery problems (late_rate >= 10%)": int(df["flag_late_delivery"].sum()),
        "Low reviews (avg_review_score <= 3.5)": int(df["flag_low_reviews"].sum()),
        "Long time since last purchase (recency_days >= 120)": int(df["flag_high_recency"].sum()),
        "Low frequency (<= 1 order)": int(df["flag_low_frequency"].sum()),
        "High value (top 5% spend)": int(df["flag_high_value"].sum()),
    }
    flags_df = (
        pd.DataFrame([{"driver": k, "customers": v} for k, v in flags.items()])
        .sort_values("customers", ascending=False)
    )

    df["act_retention_outreach"] = (df["risk_bucket"] == "high")
    df["act_delivery_followup"] = (df["risk_bucket"].isin(["high", "medium"])) & df["flag_late_delivery"]
    df["act_support_followup"] = (df["risk_bucket"].isin(["high", "medium"])) & df["flag_low_reviews"]
    df["act_winback_campaign"] = (df["risk_bucket"] == "high") & df["flag_high_recency"]
    df["act_upsell"] = (df["risk_bucket"] == "low")

    actions = {
        "Retention outreach (high risk)": int(df["act_retention_outreach"].sum()),
        "Delivery issue follow-up (late delivery flagged)": int(df["act_delivery_followup"].sum()),
        "Customer support follow-up (low reviews flagged)": int(df["act_support_followup"].sum()),
        "Win-back campaign (high risk + high recency)": int(df["act_winback_campaign"].sum()),
        "Upsell / cross-sell (low risk)": int(df["act_upsell"].sum()),
    }
    actions_df = (
        pd.DataFrame([{"action": k, "customers": v} for k, v in actions.items()])
        .sort_values("customers", ascending=False)
    )

    return {
        "total": total,
        "high": high,
        "medium": med,
        "low": low,
        "top_states": top_states,
        "drivers": flags_df,
        "actions": actions_df,
        "threshold": float(thr),
    }


def render_insights(title: str, insights: dict):
    st.subheader(title)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", int(insights["total"]))
    c2.metric("High risk", int(insights["high"]))
    c3.metric("Medium risk", int(insights["medium"]))
    c4.metric("Low risk", int(insights["low"]))

    st.caption(
        f"Risk bands are based on threshold = {insights['threshold']:.3f} "
        "(High >= thr, Medium >= 0.5×thr, Low < 0.5×thr)"
    )

    colA, colB = st.columns([2, 3])
    with colA:
        st.write("Top states by HIGH risk customers")
        st.dataframe(insights["top_states"], use_container_width=True, height=320)
    with colB:
        st.write("Main drivers (simple rule-based flags)")
        fig = px.bar(insights["drivers"], x="customers", y="driver", orientation="h")
        fig.update_layout(height=320, xaxis_title="Customers", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.write("Recommended actions (counts)")
    fig2 = px.bar(insights["actions"], x="customers", y="action", orientation="h")
    fig2.update_layout(height=320, xaxis_title="Customers", yaxis_title="")
    st.plotly_chart(fig2, use_container_width=True)


def plot_risk_bands(scored_df: pd.DataFrame):
    st.subheader("Risk bands (Low / Medium / High)")
    band_counts = (
        scored_df["risk_bucket"]
        .value_counts()
        .reindex(["low", "medium", "high"])
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    band_counts.columns = ["risk_bucket", "customers"]
    fig_band = px.bar(band_counts, x="risk_bucket", y="customers", text="customers", color="risk_bucket")
    fig_band.update_layout(xaxis_title="Risk band", yaxis_title="Customers")
    st.plotly_chart(fig_band, use_container_width=True)


def main():
    st.set_page_config(page_title="Olist Customer Analytics", layout="wide")
    st.title("Olist Customer Analytics (RFM + Cohort + Delivery + Categories + Reviews + AI/ML)")

    engine = get_engine()

    # Latest RFM snapshot date
    with engine.connect() as conn:
        snapshot_date = conn.execute(
            text("SELECT MAX(snapshot_date) FROM mart.customer_segment_geo_snapshot")
        ).scalar_one()

    st.caption(f"RFM snapshot date: {snapshot_date}")

    # ------------------------
    # Sidebar filters
    # ------------------------
    st.sidebar.header("Filters")

    states = pd.read_sql(
        text("""
            SELECT DISTINCT customer_state
            FROM mart.customer_segment_geo_snapshot
            WHERE snapshot_date = :d AND customer_state IS NOT NULL
            ORDER BY customer_state
        """),
        engine,
        params={"d": snapshot_date},
    )["customer_state"].tolist()

    segments = pd.read_sql(
        text("""
            SELECT DISTINCT segment_name
            FROM mart.customer_segment_geo_snapshot
            WHERE snapshot_date = :d
            ORDER BY segment_name
        """),
        engine,
        params={"d": snapshot_date},
    )["segment_name"].tolist()

    state_choice = st.sidebar.selectbox("Customer state (RFM)", ["All"] + states, index=0)
    segment_choice = st.sidebar.selectbox("Segment (RFM)", ["All"] + segments, index=0)

    where = "snapshot_date = :d"
    params = {"d": snapshot_date}

    if state_choice != "All":
        where += " AND customer_state = :state"
        params["state"] = state_choice

    if segment_choice != "All":
        where += " AND segment_name = :segment"
        params["segment"] = segment_choice

    # Category filter (for category sales section)
    categories = pd.read_sql(
        text("""
            SELECT DISTINCT product_category_name
            FROM mart.category_sales_monthly
            ORDER BY product_category_name
        """),
        engine,
    )["product_category_name"].tolist()

    category_choice = st.sidebar.selectbox("Category (Sales)", ["All"] + categories, index=0)

    # AI/ML scoring snapshot selector
    ml_snapshots = pd.read_sql(
        text("""
            SELECT DISTINCT snapshot_date
            FROM mart.customer_churn_features_snapshot
            ORDER BY snapshot_date DESC
        """),
        engine,
    )["snapshot_date"].tolist()

    if ml_snapshots:
        ml_snapshot_choice = st.sidebar.selectbox("AI snapshot date", ml_snapshots, index=0)
    else:
        ml_snapshot_choice = None

    # Model selector
    model_choice = st.sidebar.selectbox(
        "AI model",
        ["RandomForest (joblib)", "Neural Network (PyTorch)"],
        index=0,
    )

    # Threshold defaults
    torch_metrics_for_threshold = load_torch_metrics()
    nn_default_thr = None
    if torch_metrics_for_threshold and "best_threshold_f1" in torch_metrics_for_threshold:
        try:
            nn_default_thr = float(torch_metrics_for_threshold["best_threshold_f1"])
        except Exception:
            nn_default_thr = None

    default_thr = (
        0.20
        if model_choice == "RandomForest (joblib)"
        else (nn_default_thr if nn_default_thr is not None else 0.50)
    )

    threshold = st.sidebar.slider(
        "AI threshold (high risk)",
        0.0,
        1.0,
        float(default_thr),
        0.01,
        help="Risk bands: High >= threshold, Medium >= 0.5×threshold, Low < 0.5×threshold.",
    )

    # Glossary / help
    render_glossary_and_help(float(threshold))

    # ------------------------
    # Filtered KPIs (RFM)
    # ------------------------
    kpis = pd.read_sql(
        text(f"""
            SELECT
              COUNT(*) AS customers,
              ROUND(AVG(monetary)::numeric, 2) AS avg_monetary,
              ROUND(AVG(frequency)::numeric, 2) AS avg_frequency,
              ROUND(AVG(recency_days)::numeric, 2) AS avg_recency_days
            FROM mart.customer_segment_geo_snapshot
            WHERE {where}
        """),
        engine,
        params=params,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", int(kpis.loc[0, "customers"]))
    c2.metric("Avg monetary", float(kpis.loc[0, "avg_monetary"] or 0))
    c3.metric("Avg frequency", float(kpis.loc[0, "avg_frequency"] or 0))
    c4.metric("Avg recency (days)", float(kpis.loc[0, "avg_recency_days"] or 0))

    # ------------------------
    # RFM Segments (Filtered)
    # ------------------------
    st.header("RFM Segments (Filtered)")

    seg_counts = pd.read_sql(
        text(f"""
            SELECT segment_name, COUNT(*) AS customers
            FROM mart.customer_segment_geo_snapshot
            WHERE {where}
            GROUP BY segment_name
            ORDER BY customers DESC
        """),
        engine,
        params=params,
    )

    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("Segment distribution")
        fig = px.bar(seg_counts, x="segment_name", y="customers", text="customers")
        fig.update_layout(xaxis_title="Segment", yaxis_title="Customers")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top states (by customers)")
        top_states = pd.read_sql(
            text(f"""
                SELECT COALESCE(customer_state,'Unknown') AS customer_state, COUNT(*) AS customers
                FROM mart.customer_segment_geo_snapshot
                WHERE {where}
                GROUP BY 1
                ORDER BY customers DESC
                LIMIT 15
            """),
            engine,
            params=params,
        )
        fig2 = px.bar(top_states, x="customer_state", y="customers", text="customers")
        fig2.update_layout(xaxis_title="State", yaxis_title="Customers")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Top 50 customers by monetary value (filtered)")
    top = pd.read_sql(
        text(f"""
            SELECT customer_unique_id, segment_name, customer_state, customer_city,
                   monetary, frequency, recency_days
            FROM mart.customer_segment_geo_snapshot
            WHERE {where}
            ORDER BY monetary DESC
            LIMIT 50
        """),
        engine,
        params=params,
    )
    st.dataframe(top, use_container_width=True)

    # ------------------------
    # Cohort Retention (Global)
    # ------------------------
    st.header("Cohort Retention (Global)")

    retention = pd.read_sql(
        text("""
            SELECT cohort_month, period_number, customers, cohort_size, retention_rate
            FROM mart.cohort_retention
            ORDER BY cohort_month, period_number
        """),
        engine,
    )

    rate_df = retention[(retention["period_number"] >= 1) & (retention["period_number"] <= 6)].copy()
    rate_pivot = (
        rate_df.pivot(index="cohort_month", columns="period_number", values="retention_rate")
        .fillna(0)
        .sort_index()
    )
    rate_log = np.log1p(rate_pivot)

    st.subheader("Retention rate heatmap (months 1–6, log-scaled)")
    fig_rate = px.imshow(
        rate_log,
        aspect="auto",
        color_continuous_scale="YlGnBu",
        labels=dict(x="Period (months)", y="Cohort month", color="log1p(rate)"),
    )
    fig_rate.update_layout(height=520)
    st.plotly_chart(fig_rate, use_container_width=True)

    st.subheader("Retention table (real %, months 1–6)")
    st.dataframe((rate_pivot * 100).round(2), use_container_width=True)

    # ------------------------
    # Delivery Performance (Monthly)
    # ------------------------
    st.header("Delivery Performance (Monthly)")

    delivery = pd.read_sql(
        text("""
            SELECT order_month, orders, avg_delivery_days, avg_estimated_days, late_orders, late_rate
            FROM mart.delivery_performance_monthly
            ORDER BY order_month
        """),
        engine,
    )

    colA, colB = st.columns([3, 2])

    with colA:
        st.subheader("Avg delivery time vs estimated (days)")
        line_df = delivery.melt(
            id_vars=["order_month"],
            value_vars=["avg_delivery_days", "avg_estimated_days"],
            var_name="metric",
            value_name="days",
        )
        line_df["metric"] = line_df["metric"].replace(
            {"avg_delivery_days": "Avg delivery days", "avg_estimated_days": "Avg estimated days"}
        )
        fig3 = px.line(line_df, x="order_month", y="days", color="metric", markers=True)
        fig3.update_layout(xaxis_title="Order month", yaxis_title="Days")
        st.plotly_chart(fig3, use_container_width=True)

    with colB:
        st.subheader("Late delivery rate (%)")
        late_df = delivery.copy()
        late_df["late_rate_pct"] = (late_df["late_rate"] * 100).round(2)
        fig4 = px.bar(late_df, x="order_month", y="late_rate_pct")
        fig4.update_layout(xaxis_title="Order month", yaxis_title="Late rate (%)")
        st.plotly_chart(fig4, use_container_width=True)

    # ------------------------
    # Category Sales (Monthly)
    # ------------------------
    st.header("Category Sales (Monthly)")

    top_cat = pd.read_sql(
        text("""
            SELECT product_category_name,
                   ROUND(SUM(revenue)::numeric, 2) AS revenue,
                   SUM(orders) AS orders,
                   SUM(items) AS items
            FROM mart.category_sales_monthly
            GROUP BY product_category_name
            ORDER BY revenue DESC
            LIMIT 15
        """),
        engine,
    )

    colX, colY = st.columns([2, 3])

    with colX:
        st.subheader("Top categories by revenue (overall)")
        fig5 = px.bar(top_cat, x="product_category_name", y="revenue", text="revenue")
        fig5.update_layout(xaxis_title="Category", yaxis_title="Revenue")
        st.plotly_chart(fig5, use_container_width=True)

    with colY:
        st.subheader("Revenue trend (monthly)")
        if category_choice == "All":
            trend = pd.read_sql(
                text("""
                    SELECT order_month, ROUND(SUM(revenue)::numeric, 2) AS revenue
                    FROM mart.category_sales_monthly
                    GROUP BY order_month
                    ORDER BY order_month
                """),
                engine,
            )
            title = "All categories"
        else:
            trend = pd.read_sql(
                text("""
                    SELECT order_month, revenue
                    FROM mart.category_sales_monthly
                    WHERE product_category_name = :cat
                    ORDER BY order_month
                """),
                engine,
                params={"cat": category_choice},
            )
            title = category_choice

        fig6 = px.line(trend, x="order_month", y="revenue", markers=True, title=title)
        fig6.update_layout(xaxis_title="Order month", yaxis_title="Revenue")
        st.plotly_chart(fig6, use_container_width=True)

    # ------------------------
    # Reviews / Customer Satisfaction (Monthly)
    # ------------------------
    st.header("Customer Satisfaction (Reviews)")

    reviews = pd.read_sql(
        text("""
            SELECT review_month, reviews, avg_review_score, pct_1_star, pct_5_star
            FROM mart.review_monthly
            ORDER BY review_month
        """),
        engine,
    )

    colR1, colR2 = st.columns([3, 2])

    with colR1:
        st.subheader("Avg review score over time")
        fig7 = px.line(reviews, x="review_month", y="avg_review_score", markers=True)
        fig7.update_layout(xaxis_title="Month", yaxis_title="Avg review score (1-5)")
        st.plotly_chart(fig7, use_container_width=True)

    with colR2:
        st.subheader("1-star vs 5-star share (%)")
        r = reviews.copy()
        r["pct_1_star_pct"] = (r["pct_1_star"] * 100).round(2)
        r["pct_5_star_pct"] = (r["pct_5_star"] * 100).round(2)

        rr = r.melt(
            id_vars=["review_month"],
            value_vars=["pct_1_star_pct", "pct_5_star_pct"],
            var_name="metric",
            value_name="pct",
        )
        rr["metric"] = rr["metric"].replace(
            {"pct_1_star_pct": "% 1-star", "pct_5_star_pct": "% 5-star"}
        )
        fig8 = px.line(rr, x="review_month", y="pct", color="metric", markers=True)
        fig8.update_layout(xaxis_title="Month", yaxis_title="Percent (%)")
        st.plotly_chart(fig8, use_container_width=True)

    st.subheader("Late delivery rate vs Avg review score (by month)")
    join = pd.merge(
        delivery[["order_month", "late_rate"]],
        reviews.rename(columns={"review_month": "order_month"})[["order_month", "avg_review_score", "reviews"]],
        on="order_month",
        how="inner",
    )
    join["late_rate_pct"] = (join["late_rate"] * 100).round(2)

    fig9 = px.scatter(
        join,
        x="late_rate_pct",
        y="avg_review_score",
        size="reviews",
        hover_data=["order_month", "reviews"],
    )
    fig9.update_layout(xaxis_title="Late delivery rate (%)", yaxis_title="Avg review score (1-5)")
    st.plotly_chart(fig9, use_container_width=True)

    # ------------------------
    # AI / ML: Reorder Prediction (90 days)
    # ------------------------
    st.header("AI / ML: Reorder Prediction (90 days)")

    if model_choice == "Neural Network (PyTorch)" and nn_default_thr is not None:
        st.caption(f"NN recommended threshold (best F1 from training): {nn_default_thr:.3f}")

    # RF metrics
    metrics_rf = load_model_metrics_rf()
    if metrics_rf is None:
        st.warning("RF metrics not found (models/churn_metrics.json). Train the model first.")
    else:
        with st.expander("RandomForest model details / metrics"):
            st.json(metrics_rf)

    # NN metrics + training plots
    torch_metrics = load_torch_metrics()
    torch_bundle = load_torch_bundle() if model_choice == "Neural Network (PyTorch)" else None

    if model_choice == "Neural Network (PyTorch)":
        if torch_bundle is None:
            st.warning("Torch NN bundle not found. Train NN first (models/churn_torch_bundle.joblib).")
        else:
            with st.expander("Neural Network (PyTorch) details / metrics"):
                st.json(torch_metrics or {"info": "NN metrics not found (models/churn_torch_metrics.json)"})

            # IMPORTANT: The 4 NN graphs (restored)
            st.subheader("Neural Network training plots")
            cimg1, cimg2 = st.columns(2)
            with cimg1:
                st.image("reports/torch_nn_loss.png", caption="Training loss", use_container_width=True)
                st.image("reports/torch_nn_pr_curve.png", caption="Precision-Recall curve", use_container_width=True)
            with cimg2:
                st.image("reports/torch_nn_val_pr_auc.png", caption="Validation PR-AUC", use_container_width=True)
                st.image("reports/torch_nn_confusion_matrix.png", caption="Confusion matrix", use_container_width=True)

    if ml_snapshot_choice is None:
        st.warning("No ML feature snapshots found. Run etl/build_customer_churn_features_snapshot.py")
        return

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Snapshot scoring (DB)", "Single customer predictor", "Batch upload (CSV)", "Prediction log (DB)"]
    )

    # Tab 1: Snapshot
    with tab1:
        feat = pd.read_sql(
            text("""
                SELECT
                  customer_unique_id,
                  recency_days,
                  frequency,
                  monetary,
                  avg_delivery_days,
                  late_rate,
                  avg_review_score,
                  customer_state
                FROM mart.customer_churn_features_snapshot
                WHERE snapshot_date = :d
            """),
            engine,
            params={"d": ml_snapshot_choice},
        )

        if feat.empty:
            st.warning(f"No features found for snapshot_date={ml_snapshot_choice}")
        else:
            X = feat.drop(columns=["customer_unique_id"])
            proba = score_dataframe(model_choice, X)

            scored = feat.copy()
            scored["reorder_proba_90d"] = proba
            scored["risk_bucket"] = [risk_bucket(float(p), float(threshold)) for p in proba]

            plot_risk_bands(scored)

            st.subheader("Predicted probability distribution")
            fig_h = px.histogram(pd.DataFrame({"reorder_proba_90d": proba}), x="reorder_proba_90d", nbins=50)
            fig_h.add_vline(x=float(threshold), line_width=2, line_dash="dash")
            st.plotly_chart(fig_h, use_container_width=True)

            render_insights("Insights summary (Snapshot scoring)", compute_insights(scored, float(threshold)))

            st.divider()
            st.subheader("Store predictions to database (keeps history via run_id)")
            model_name = "rf" if model_choice.startswith("RandomForest") else "torch_nn"
            if st.button("Save ALL scored predictions to DB", type="primary"):
                run_id = make_run_id("snapshot", model_name, snapshot_date=ml_snapshot_choice)
                insert_predictions_to_db(
                    engine,
                    scored,
                    run_id=run_id,
                    source="snapshot",
                    snapshot_date=ml_snapshot_choice,
                    model_name=model_name,
                    threshold=float(threshold),
                )
                st.success(f"Saved predictions. run_id={run_id}")

    # Tab 2: Single customer
    with tab2:
        st.subheader("Predict for a single customer (manual input)")
        with st.form("single_customer_form", clear_on_submit=False):
            c_state = st.selectbox(
                "Customer state",
                ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "DF", "ES", "GO", "PE", "CE", "PA", "MT", "MA", "MS", "PB", "RN", "PI", "AL", "SE", "RO", "TO", "AM", "AP", "RR", "AC", "Unknown"],
            )
            c1, c2, c3 = st.columns(3)
            recency_days = c1.number_input("Days since last purchase", min_value=0.0, value=60.0, step=1.0)
            frequency = c2.number_input("Total orders", min_value=0.0, value=1.0, step=1.0)
            monetary = c3.number_input("Total spend", min_value=0.0, value=150.0, step=10.0)

            d1, d2, d3 = st.columns(3)
            avg_delivery_days = d1.number_input("Avg delivery days", min_value=0.0, value=10.0, step=1.0)
            late_rate = d2.number_input("Late delivery rate (0 to 1)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
            avg_review_score = d3.number_input("Avg review score (1-5)", min_value=1.0, max_value=5.0, value=4.2, step=0.1)

            submitted = st.form_submit_button("Predict")

        if submitted:
            one = pd.DataFrame([{
                "recency_days": recency_days,
                "frequency": frequency,
                "monetary": monetary,
                "avg_delivery_days": avg_delivery_days,
                "late_rate": late_rate,
                "avg_review_score": avg_review_score,
                "customer_state": c_state,
            }])
            p = float(score_dataframe(model_choice, one)[0])
            bucket = risk_bucket(p, float(threshold))

            st.metric("Predicted reorder probability (90d)", f"{p:.4f}")
            risk_badge(bucket, p, float(threshold))

            st.subheader("Suggestions")
            for s in generate_suggestions(one.iloc[0].to_dict(), p, float(threshold)):
                st.write("-", s)

            st.divider()
            if st.button("Save this prediction to DB"):
                to_save = one.copy()
                to_save["customer_unique_id"] = None
                to_save["reorder_proba_90d"] = p
                to_save["risk_bucket"] = bucket
                model_name = "rf" if model_choice.startswith("RandomForest") else "torch_nn"
                run_id = make_run_id("manual", model_name)
                insert_predictions_to_db(
                    engine,
                    to_save,
                    run_id=run_id,
                    source="manual",
                    snapshot_date=None,
                    model_name=model_name,
                    threshold=float(threshold),
                )
                st.success(f"Saved. run_id={run_id}")

    # Tab 3: Batch upload
    with tab3:
        st.subheader("Batch prediction from uploaded CSV (company data)")

        template = pd.DataFrame([{
            "customer_unique_id": "cust_001",
            "recency_days": 45,
            "frequency": 2,
            "monetary": 320.5,
            "avg_delivery_days": 8.0,
            "late_rate": 0.02,
            "avg_review_score": 4.6,
            "customer_state": "SP",
        }])
        st.download_button(
            "Download CSV template",
            data=template.to_csv(index=False).encode("utf-8"),
            file_name="batch_prediction_template.csv",
            mime="text/csv",
        )

        up = st.file_uploader("Upload CSV", type=["csv"])
        if up is not None:
            df_up = pd.read_csv(up)
            missing = [c for c in UPLOAD_REQUIRED_COLS if c not in df_up.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                X_up = df_up[FEATURE_COLS].copy()
                proba_up = score_dataframe(model_choice, X_up)

                scored_up = df_up.copy()
                scored_up["reorder_proba_90d"] = proba_up
                scored_up["risk_bucket"] = [risk_bucket(float(p), float(threshold)) for p in proba_up]

                plot_risk_bands(scored_up)
                render_insights("Insights summary (Batch upload)", compute_insights(scored_up, float(threshold)))

                st.subheader("Batch results (preview)")
                st.dataframe(scored_up.head(50), use_container_width=True)

                st.download_button(
                    "Download scored CSV",
                    data=scored_up.to_csv(index=False).encode("utf-8"),
                    file_name=f"scored_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

                st.divider()
                st.subheader("Store batch predictions to database (keeps history via run_id)")
                model_name = "rf" if model_choice.startswith("RandomForest") else "torch_nn"
                if st.button("Save ALL uploaded predictions to DB", type="primary"):
                    run_id = make_run_id("upload", model_name)
                    insert_predictions_to_db(
                        engine,
                        scored_up,
                        run_id=run_id,
                        source="upload",
                        snapshot_date=None,
                        model_name=model_name,
                        threshold=float(threshold),
                    )
                    st.success(f"Saved uploaded predictions. run_id={run_id}")

    # Tab 4: Run summary/log
    with tab4:
        st.subheader("Prediction runs (summary)")
        run_limit = st.number_input("How many latest runs to show", min_value=5, max_value=200, value=50, step=5)

        runs_df = pd.read_sql(
            text("""
                SELECT
                  run_id,
                  MAX(scored_at) AS scored_at,
                  MAX(source) AS source,
                  MAX(snapshot_date) AS snapshot_date,
                  MAX(model_name) AS model_name,
                  MAX(threshold) AS threshold,
                  COUNT(*) AS rows,
                  AVG(reorder_proba_90d) AS avg_prob,
                  AVG(CASE WHEN risk_bucket = 'high' THEN 1 ELSE 0 END) * 100.0 AS high_risk_pct
                FROM mart.customer_churn_predictions_log
                WHERE run_id IS NOT NULL
                GROUP BY run_id
                ORDER BY MAX(scored_at) DESC
                LIMIT :lim
            """),
            engine,
            params={"lim": int(run_limit)},
        )

        if runs_df.empty:
            st.info("No runs found yet. Save a snapshot/upload/manual prediction first.")
            return

        st.dataframe(runs_df, use_container_width=True)

        selected_run_id = st.selectbox("Select run_id", runs_df["run_id"].tolist(), index=0)
        limit = st.number_input("Rows to show (detail)", min_value=50, max_value=5000, value=500, step=50)

        log_df = pd.read_sql(
            text("""
                SELECT run_id, scored_at, source, snapshot_date, model_name, threshold, customer_unique_id,
                       recency_days, frequency, monetary, avg_delivery_days, late_rate, avg_review_score, customer_state,
                       reorder_proba_90d, risk_bucket
                FROM mart.customer_churn_predictions_log
                WHERE run_id = :run_id
                ORDER BY reorder_proba_90d DESC
                LIMIT :lim
            """),
            engine,
            params={"run_id": selected_run_id, "lim": int(limit)},
        )

        st.subheader("Prediction log (details)")
        st.dataframe(log_df, use_container_width=True)

        if not log_df.empty:
            st.subheader("Monitoring charts (selected run)")
            st.plotly_chart(px.histogram(log_df, x="risk_bucket"), use_container_width=True)
            fig_prob = px.histogram(log_df, x="reorder_proba_90d", nbins=40)
            fig_prob.add_vline(x=float(log_df["threshold"].iloc[0]), line_width=2, line_dash="dash")
            st.plotly_chart(fig_prob, use_container_width=True)


if __name__ == "__main__":
    main()