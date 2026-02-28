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

    query = """
    SELECT
      DATE_TRUNC('month', o.order_purchase_timestamp::timestamp)::date AS review_month,
      COUNT(r.review_id) AS reviews,
      ROUND(AVG(r.review_score)::numeric, 2) AS avg_review_score,
      ROUND(AVG(CASE WHEN r.review_score = 1 THEN 1 ELSE 0 END)::numeric, 4) AS pct_1_star,
      ROUND(AVG(CASE WHEN r.review_score = 5 THEN 1 ELSE 0 END)::numeric, 4) AS pct_5_star
    FROM raw.olist_orders_dataset o
    JOIN raw.olist_order_reviews_dataset r
      ON r.order_id = o.order_id
    WHERE o.order_status = 'delivered'
      AND o.order_purchase_timestamp IS NOT NULL
      AND r.review_score IS NOT NULL
    GROUP BY 1
    ORDER BY 1;
    """

    df = pd.read_sql(query, engine)

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE mart.review_monthly"))

    df.to_sql(
        "review_monthly",
        engine,
        schema="mart",
        if_exists="append",
        index=False,
        method="multi",
        chunksize=50_000,
    )

    print("Built mart.review_monthly")
    print("Rows:", len(df))
    print(df.head(10))

if __name__ == "__main__":
    main()