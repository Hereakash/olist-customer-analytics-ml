import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv("config/settings.env")

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

DATA_DIR = Path("data")

FILES = {
    "olist_customers_dataset.csv": "olist_customers_dataset",
    "olist_orders_dataset.csv": "olist_orders_dataset",
    "olist_order_items_dataset.csv": "olist_order_items_dataset",
    "olist_order_payments_dataset.csv": "olist_order_payments_dataset",
    "olist_products_dataset.csv": "olist_products_dataset",
    "olist_order_reviews_dataset.csv": "olist_order_reviews_dataset",
    "olist_sellers_dataset.csv": "olist_sellers_dataset",
    "olist_geolocation_dataset.csv": "olist_geolocation_dataset",
    "product_category_name_translation.csv": "product_category_name_translation",
}

def main():
    url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(url)

    for csv_name, table_name in FILES.items():
        path = DATA_DIR / csv_name
        if not path.exists():
            print(f"SKIP (missing): {csv_name}")
            continue

        print(f"Loading {csv_name} -> raw.{table_name}")
        df = pd.read_csv(path)

        df.to_sql(
            table_name,
            engine,
            schema="raw",
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=50_000,
        )
        print(f"  rows inserted: {len(df)}")

    print("Done. Raw tables created in schema 'raw'.")

if __name__ == "__main__":
    main()