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

def main():
    engine = create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    # NOTE: timestamps may be stored as text in your raw tables, so we cast to timestamp
    query = """
    SELECT
      DATE_TRUNC('month', o.order_purchase_timestamp::timestamp)::date AS order_month,
      COUNT(*) AS orders,
      AVG((o.order_delivered_customer_date::timestamp - o.order_purchase_timestamp::timestamp)) AS avg_delivery_interval,
      AVG((o.order_estimated_delivery_date::timestamp - o.order_purchase_timestamp::timestamp)) AS avg_estimated_interval,
      SUM(
        CASE
          WHEN o.order_delivered_customer_date IS NOT NULL
           AND o.order_estimated_delivery_date IS NOT NULL
           AND o.order_delivered_customer_date::timestamp > o.order_estimated_delivery_date::timestamp
          THEN 1 ELSE 0
        END
      ) AS late_orders
    FROM raw.olist_orders_dataset o
    WHERE o.order_status = 'delivered'
      AND o.order_purchase_timestamp IS NOT NULL
      AND o.order_delivered_customer_date IS NOT NULL
      AND o.order_estimated_delivery_date IS NOT NULL
    GROUP BY 1
    ORDER BY 1;
    """

    df = pd.read_sql(query, engine)

    # Convert intervals to days
    df["avg_delivery_days"] = df["avg_delivery_interval"].dt.total_seconds() / 86400
    df["avg_estimated_days"] = df["avg_estimated_interval"].dt.total_seconds() / 86400

    df["avg_delivery_days"] = df["avg_delivery_days"].round(2)
    df["avg_estimated_days"] = df["avg_estimated_days"].round(2)

    df["late_orders"] = df["late_orders"].astype(int)
    df["late_rate"] = (df["late_orders"] / df["orders"]).round(4)

    out = df[[
        "order_month",
        "orders",
        "avg_delivery_days",
        "avg_estimated_days",
        "late_orders",
        "late_rate"
    ]].copy()

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE mart.delivery_performance_monthly"))

    out.to_sql(
        "delivery_performance_monthly",
        engine,
        schema="mart",
        if_exists="append",
        index=False,
        method="multi",
        chunksize=10_000,
    )

    print("Built mart.delivery_performance_monthly")
    print("Rows:", len(out))
    print(out.head(5))

if __name__ == "__main__":
    main()