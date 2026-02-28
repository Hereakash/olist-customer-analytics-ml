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

    # latest snapshot_date
    with engine.connect() as conn:
        snap = conn.execute(text("SELECT MAX(snapshot_date) FROM mart.customer_segments_snapshot")).scalar_one()

    query = """
    SELECT
      s.snapshot_date,
      s.customer_unique_id,
      s.segment_name,
      c.customer_state,
      c.customer_city,
      s.monetary,
      s.frequency,
      s.recency_days
    FROM mart.customer_segments_snapshot s
    LEFT JOIN (
        SELECT customer_unique_id,
               MAX(customer_state) AS customer_state,
               MAX(customer_city) AS customer_city
        FROM raw.olist_customers_dataset
        GROUP BY customer_unique_id
    ) c
      ON c.customer_unique_id = s.customer_unique_id
    WHERE s.snapshot_date = :d
    """
    df = pd.read_sql(text(query), engine, params={"d": snap})

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM mart.customer_segment_geo_snapshot WHERE snapshot_date = :d"), {"d": snap})

    df.to_sql(
        "customer_segment_geo_snapshot",
        engine,
        schema="mart",
        if_exists="append",
        index=False,
        method="multi",
        chunksize=50_000,
    )

    print(f"Saved mart.customer_segment_geo_snapshot for snapshot_date={snap} rows={len(df)}")

if __name__ == "__main__":
    main()