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

    # snapshot_date = max purchase date in dataset
    with engine.connect() as conn:
        snapshot_date = conn.execute(
            text("""
                SELECT (MAX(order_purchase_timestamp))::date
                FROM raw.olist_orders_dataset
                WHERE order_purchase_timestamp IS NOT NULL
            """)
        ).scalar_one()

    query = """
    WITH delivered_orders AS (
        SELECT
            o.order_id,
            c.customer_unique_id,
            o.order_purchase_timestamp
        FROM raw.olist_orders_dataset o
        JOIN raw.olist_customers_dataset c
          ON c.customer_id = o.customer_id
        WHERE o.order_status = 'delivered'
          AND o.order_purchase_timestamp IS NOT NULL
    ),
    order_value AS (
        SELECT
            p.order_id,
            SUM(p.payment_value) AS order_payment_value
        FROM raw.olist_order_payments_dataset p
        GROUP BY p.order_id
    )
    SELECT
        d.customer_unique_id,
        d.order_id,
        d.order_purchase_timestamp,
        COALESCE(v.order_payment_value, 0) AS order_value
    FROM delivered_orders d
    LEFT JOIN order_value v ON v.order_id = d.order_id;
    """

    df = pd.read_sql(query, engine)

    snap = pd.to_datetime(snapshot_date) + pd.Timedelta(days=1)

    rfm = df.groupby("customer_unique_id").agg(
        last_purchase_timestamp=("order_purchase_timestamp", "max"),
        first_purchase_timestamp=("order_purchase_timestamp", "min"),
        frequency=("order_id", "nunique"),
        monetary=("order_value", "sum"),
    ).reset_index()

    rfm["last_purchase_timestamp"] = pd.to_datetime(rfm["last_purchase_timestamp"])
    rfm["first_purchase_timestamp"] = pd.to_datetime(rfm["first_purchase_timestamp"])
    rfm["recency_days"] = (snap - rfm["last_purchase_timestamp"]).dt.days
    rfm["tenure_days"] = (snap - rfm["first_purchase_timestamp"]).dt.days
    rfm["snapshot_date"] = pd.to_datetime(snapshot_date).date()

    out = rfm[
        [
            "snapshot_date",
            "customer_unique_id",
            "recency_days",
            "frequency",
            "monetary",
            "last_purchase_timestamp",
            "first_purchase_timestamp",
            "tenure_days",
        ]
    ]

    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM mart.customer_rfm_snapshot WHERE snapshot_date = :d"),
            {"d": out["snapshot_date"].iloc[0]},
        )

    out.to_sql(
        "customer_rfm_snapshot",
        engine,
        schema="mart",
        if_exists="append",
        index=False,
        method="multi",
        chunksize=50_000,
    )

    print(f"Saved RFM snapshot_date={out['snapshot_date'].iloc[0]} customers={len(out)}")

if __name__ == "__main__":
    main()