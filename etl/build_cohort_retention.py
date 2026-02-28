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

    # Build customer-month activity (delivered orders)
    query = """
    SELECT
      c.customer_unique_id,
      DATE_TRUNC('month', o.order_purchase_timestamp::timestamp)::date AS order_month
    FROM raw.olist_orders_dataset o
    JOIN raw.olist_customers_dataset c
      ON c.customer_id = o.customer_id
    WHERE o.order_status = 'delivered'
      AND o.order_purchase_timestamp IS NOT NULL;
    """
    df = pd.read_sql(query, engine)

    # cohort_month = first order month
    first = df.groupby("customer_unique_id", as_index=False)["order_month"].min()
    first = first.rename(columns={"order_month": "cohort_month"})

    merged = df.merge(first, on="customer_unique_id", how="left")

    # period_number: months since cohort_month (0 = first month)
    merged["period_number"] = (
        (pd.to_datetime(merged["order_month"]).dt.year - pd.to_datetime(merged["cohort_month"]).dt.year) * 12
        + (pd.to_datetime(merged["order_month"]).dt.month - pd.to_datetime(merged["cohort_month"]).dt.month)
    )

    cohort_counts = (
        merged.groupby(["cohort_month", "order_month", "period_number"])["customer_unique_id"]
        .nunique()
        .reset_index(name="customers")
        .sort_values(["cohort_month", "period_number"])
    )

    cohort_sizes = (
        merged[merged["period_number"] == 0]
        .groupby("cohort_month")["customer_unique_id"]
        .nunique()
        .reset_index(name="cohort_size")
    )

    out = cohort_counts.merge(cohort_sizes, on="cohort_month", how="left")
    out["retention_rate"] = (out["customers"] / out["cohort_size"]).round(4)

    # Replace table contents (rerunnable)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE mart.cohort_retention"))

    out.to_sql(
        "cohort_retention",
        engine,
        schema="mart",
        if_exists="append",
        index=False,
        method="multi",
        chunksize=50_000,
    )

    print("Cohort retention built.")
    print("Rows:", len(out))
    print("Cohorts:", out["cohort_month"].nunique())

if __name__ == "__main__":
    main()