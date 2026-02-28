import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv("config/settings.env")

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# Choose a "latest-ish" date for demo scoring
SNAPSHOT_DATE = "2018-08-01"


def main():
    engine = create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    snap = pd.to_datetime(SNAPSHOT_DATE).date()

    sql = """
    WITH params AS (
      SELECT (%(snap)s::date + INTERVAL '1 day')::timestamp AS cutoff_ts
    ),
    base_orders AS (
      SELECT
        c.customer_unique_id,
        c.customer_state,
        o.order_id,
        o.order_purchase_timestamp::timestamp AS purchase_ts,
        o.order_delivered_customer_date::timestamp AS delivered_ts,
        o.order_estimated_delivery_date::timestamp AS estimated_ts
      FROM raw.olist_orders_dataset o
      JOIN raw.olist_customers_dataset c
        ON c.customer_id = o.customer_id
      WHERE o.order_status = 'delivered'
        AND o.order_purchase_timestamp IS NOT NULL
        AND o.order_delivered_customer_date IS NOT NULL
        AND o.order_estimated_delivery_date IS NOT NULL
    ),
    hist AS (
      SELECT bo.*
      FROM base_orders bo
      CROSS JOIN params p
      WHERE bo.delivered_ts < p.cutoff_ts
    ),
    rfm AS (
      SELECT
        h.customer_unique_id,
        MAX(h.customer_state) AS customer_state,
        (DATE(%(snap)s) - DATE(MAX(h.delivered_ts)))::int AS recency_days,
        COUNT(DISTINCT h.order_id)::int AS frequency
      FROM hist h
      GROUP BY h.customer_unique_id
    ),
    monetary AS (
      SELECT
        c.customer_unique_id,
        ROUND(SUM(oi.price + oi.freight_value)::numeric, 2) AS monetary
      FROM raw.olist_order_items_dataset oi
      JOIN raw.olist_orders_dataset o
        ON o.order_id = oi.order_id
      JOIN raw.olist_customers_dataset c
        ON c.customer_id = o.customer_id
      CROSS JOIN params p
      WHERE o.order_status = 'delivered'
        AND o.order_delivered_customer_date IS NOT NULL
        AND o.order_delivered_customer_date::timestamp < p.cutoff_ts
      GROUP BY c.customer_unique_id
    ),
    ops AS (
      SELECT
        h.customer_unique_id,
        AVG(EXTRACT(EPOCH FROM (h.delivered_ts - h.purchase_ts))/86400.0)::numeric(10,2) AS avg_delivery_days,
        AVG(CASE WHEN h.delivered_ts > h.estimated_ts THEN 1 ELSE 0 END)::numeric(10,4) AS late_rate
      FROM hist h
      GROUP BY h.customer_unique_id
    ),
    reviews AS (
      SELECT
        c.customer_unique_id,
        AVG(r.review_score)::numeric(10,2) AS avg_review_score
      FROM raw.olist_order_reviews_dataset r
      JOIN raw.olist_orders_dataset o
        ON o.order_id = r.order_id
      JOIN raw.olist_customers_dataset c
        ON c.customer_id = o.customer_id
      CROSS JOIN params p
      WHERE o.order_status = 'delivered'
        AND o.order_delivered_customer_date IS NOT NULL
        AND o.order_delivered_customer_date::timestamp < p.cutoff_ts
        AND r.review_score IS NOT NULL
      GROUP BY c.customer_unique_id
    )
    SELECT
      %(snap)s::date AS snapshot_date,
      r.customer_unique_id,
      r.recency_days,
      r.frequency,
      COALESCE(m.monetary, 0)::numeric(18,2) AS monetary,
      o.avg_delivery_days,
      o.late_rate,
      rv.avg_review_score,
      COALESCE(r.customer_state, 'Unknown') AS customer_state
    FROM rfm r
    LEFT JOIN monetary m ON m.customer_unique_id = r.customer_unique_id
    LEFT JOIN ops o ON o.customer_unique_id = r.customer_unique_id
    LEFT JOIN reviews rv ON rv.customer_unique_id = r.customer_unique_id
    ;
    """

    df = pd.read_sql(sql, engine, params={"snap": snap})

    if df.empty:
        raise RuntimeError(f"No rows produced for features snapshot {snap}. Try an earlier date like 2018-07-01.")

    df["avg_delivery_days"] = df["avg_delivery_days"].fillna(df["avg_delivery_days"].median())
    df["late_rate"] = df["late_rate"].fillna(0)
    df["avg_review_score"] = df["avg_review_score"].fillna(df["avg_review_score"].median())
    df["customer_state"] = df["customer_state"].fillna("Unknown")

    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM mart.customer_churn_features_snapshot WHERE snapshot_date = :snap"),
            {"snap": snap},
        )

    df.to_sql(
        "customer_churn_features_snapshot",
        engine,
        schema="mart",
        if_exists="append",
        index=False,
        method="multi",
        chunksize=50_000,
    )

    print(f"Built mart.customer_churn_features_snapshot for snapshot_date={snap}")
    print("Rows:", len(df))
    print(df.head(3))


if __name__ == "__main__":
    main()