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

    # Cast timestamps because raw columns may be text
    query = """
    SELECT
      DATE_TRUNC('month', o.order_purchase_timestamp::timestamp)::date AS order_month,
      COALESCE(p.product_category_name, 'unknown') AS product_category_name,
      COUNT(DISTINCT o.order_id) AS orders,
      COUNT(*) AS items,
      ROUND(SUM(oi.price + oi.freight_value)::numeric, 2) AS revenue
    FROM raw.olist_orders_dataset o
    JOIN raw.olist_order_items_dataset oi
      ON oi.order_id = o.order_id
    LEFT JOIN raw.olist_products_dataset p
      ON p.product_id = oi.product_id
    WHERE o.order_status = 'delivered'
      AND o.order_purchase_timestamp IS NOT NULL
    GROUP BY 1, 2
    ORDER BY 1, 2;
    """

    df = pd.read_sql(query, engine)

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE mart.category_sales_monthly"))

    df.to_sql(
        "category_sales_monthly",
        engine,
        schema="mart",
        if_exists="append",
        index=False,
        method="multi",
        chunksize=50_000,
    )

    print("Built mart.category_sales_monthly")
    print("Rows:", len(df))
    print(df.head(10))

if __name__ == "__main__":
    main()