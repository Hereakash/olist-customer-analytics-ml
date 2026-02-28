# Database Documentation

## Overview

The project uses **PostgreSQL (≥ 13)** as its sole database. Two schemas are used:

| Schema | Purpose |
|---|---|
| `raw` | Raw CSV data loaded 1-to-1 from the Olist dataset files |
| `mart` | Analytics-ready tables and ML datasets produced by ETL scripts |

---

## Connection Approach

All Python scripts (ETL, ML training, and the Streamlit dashboard) read credentials from **`config/settings.env`** using `python-dotenv`:

```python
from dotenv import load_dotenv
load_dotenv("config/settings.env")

DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = os.getenv("DB_PORT", "5432")
DB_NAME     = os.getenv("DB_NAME")
DB_USER     = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
```

A SQLAlchemy engine is then constructed with:

```python
from sqlalchemy import create_engine
engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
```

**Files that create a DB connection:**

| File | How it connects |
|---|---|
| `etl/ingest_raw.py` | `create_engine(url)` inside `main()` |
| `etl/build_rfm.py` | `create_engine(url)` inside `main()` |
| `etl/build_rfm_segments.py` | `create_engine(url)` inside `main()` |
| `etl/build_customer_segment_geo.py` | `create_engine(url)` inside `main()` |
| `etl/build_cohort_retention.py` | `create_engine(url)` inside `main()` |
| `etl/build_delivery_performance.py` | `create_engine(url)` inside `main()` |
| `etl/build_category_sales.py` | `create_engine(url)` inside `main()` |
| `etl/build_review_monthly.py` | `create_engine(url)` inside `main()` |
| `etl/build_customer_churn_ml_dataset.py` | `create_engine(url)` inside `main()` |
| `etl/build_customer_churn_features_snapshot.py` | `create_engine(url)` inside `main()` |
| `ml/train_churn_model.py` | `create_engine(url)` inside `main()` |
| `ml/train_churn_torch_nn.py` | `create_engine(url)` inside `main()` |
| `dashboard/streamlit_app.py` | `get_engine()` cached with `@st.cache_resource` |

> **Security note:** `config/settings.env` is listed in `.gitignore` and must never be committed. Copy `.env.example` as a starting point.

---

## Raw Schema Tables

Loaded from CSV files by `etl/ingest_raw.py` using `pd.to_sql(if_exists="replace")`. All tables are in schema `raw`.

| Table | Source CSV | Key columns |
|---|---|---|
| `raw.olist_customers_dataset` | `olist_customers_dataset.csv` | `customer_id` (PK), `customer_unique_id`, `customer_zip_code_prefix`, `customer_city`, `customer_state` |
| `raw.olist_orders_dataset` | `olist_orders_dataset.csv` | `order_id` (PK), `customer_id` (FK), `order_status`, `order_purchase_timestamp`, `order_approved_at`, `order_delivered_carrier_date`, `order_delivered_customer_date`, `order_estimated_delivery_date` |
| `raw.olist_order_items_dataset` | `olist_order_items_dataset.csv` | `order_id` (FK), `order_item_id`, `product_id` (FK), `seller_id` (FK), `price`, `freight_value` |
| `raw.olist_order_payments_dataset` | `olist_order_payments_dataset.csv` | `order_id` (FK), `payment_sequential`, `payment_type`, `payment_installments`, `payment_value` |
| `raw.olist_products_dataset` | `olist_products_dataset.csv` | `product_id` (PK), `product_category_name` (FK), `product_name_lenght`, `product_photos_qty`, `product_weight_g` |
| `raw.olist_order_reviews_dataset` | `olist_order_reviews_dataset.csv` | `review_id` (PK), `order_id` (FK), `review_score`, `review_creation_date`, `review_answer_timestamp` |
| `raw.olist_sellers_dataset` | `olist_sellers_dataset.csv` | `seller_id` (PK), `seller_zip_code_prefix`, `seller_city`, `seller_state` |
| `raw.olist_geolocation_dataset` | `olist_geolocation_dataset.csv` | `geolocation_zip_code_prefix`, `geolocation_lat`, `geolocation_lng`, `geolocation_city`, `geolocation_state` |
| `raw.product_category_name_translation` | `product_category_name_translation.csv` | `product_category_name` (PK), `product_category_name_english` |

---

## Mart Schema Tables

Analytics-ready tables produced by ETL scripts. All tables are in schema `mart`.

### `mart.customer_rfm_snapshot`
**Produced by:** `etl/build_rfm.py`

| Column | Type | Description |
|---|---|---|
| `snapshot_date` | DATE | Date of this RFM snapshot |
| `customer_unique_id` | TEXT | Unique customer identifier |
| `recency_days` | INT | Days since last delivered order |
| `frequency` | INT | Number of distinct delivered orders |
| `monetary` | NUMERIC(18,2) | Total spend (price + freight) |
| `last_purchase_timestamp` | TIMESTAMP | Most recent delivered order timestamp |
| `first_purchase_timestamp` | TIMESTAMP | Earliest delivered order timestamp |
| `tenure_days` | INT | Days between first and last purchase |

---

### `mart.customer_rfm_scored`
**Produced by:** `etl/build_rfm_segments.py`

| Column | Type | Description |
|---|---|---|
| `snapshot_date` | DATE | Date of this snapshot |
| `customer_unique_id` | TEXT | Unique customer identifier |
| `recency_days` | INT | Raw recency value |
| `frequency` | INT | Raw frequency value |
| `monetary` | NUMERIC(18,2) | Raw monetary value |
| `r_score` | INT | Recency quintile (1=worst, 5=best) |
| `f_score` | INT | Frequency quintile (1=worst, 5=best) |
| `m_score` | INT | Monetary quintile (1=worst, 5=best) |
| `rfm_score` | TEXT | Composite score string, e.g. `"555"` |

---

### `mart.customer_segments_snapshot`
**Produced by:** `etl/build_rfm_segments.py`

| Column | Type | Description |
|---|---|---|
| `snapshot_date` | DATE | Date of this snapshot |
| `customer_unique_id` | TEXT | Unique customer identifier |
| `segment_name` | TEXT | Business label (Champions, Loyal Customers, At Risk, …) |
| `r_score` | INT | Recency quintile |
| `f_score` | INT | Frequency quintile |
| `m_score` | INT | Monetary quintile |
| `recency_days` | INT | Raw recency value |
| `frequency` | INT | Raw frequency value |
| `monetary` | NUMERIC(18,2) | Raw monetary value |

---

### `mart.customer_segment_geo_snapshot`
**Produced by:** `etl/build_customer_segment_geo.py`

| Column | Type | Description |
|---|---|---|
| `snapshot_date` | DATE | Date of this snapshot |
| `customer_unique_id` | TEXT | Unique customer identifier |
| `segment_name` | TEXT | Business segment label |
| `customer_state` | TEXT | Brazilian state code |
| `customer_city` | TEXT | Customer city |
| `monetary` | NUMERIC(18,2) | Total spend |
| `frequency` | INT | Order count |
| `recency_days` | INT | Days since last order |

---

### `mart.cohort_retention`
**Produced by:** `etl/build_cohort_retention.py`

| Column | Type | Description |
|---|---|---|
| `cohort_month` | DATE | First-order month for the cohort |
| `order_month` | DATE | Month customers were active |
| `period_number` | INT | Months since cohort start (0 = acquisition month) |
| `customers` | INT | Distinct customers active in this period |
| `cohort_size` | INT | Total customers in the cohort |
| `retention_rate` | NUMERIC(8,4) | `customers / cohort_size` |

---

### `mart.delivery_performance_monthly`
**Produced by:** `etl/build_delivery_performance.py`

| Column | Type | Description |
|---|---|---|
| `order_month` | DATE | Calendar month (truncated to first day) |
| `orders` | INT | Delivered orders in this month |
| `avg_delivery_days` | NUMERIC(8,2) | Mean actual delivery days |
| `avg_estimated_days` | NUMERIC(8,2) | Mean estimated delivery days |
| `late_orders` | INT | Orders delivered after estimated date |
| `late_rate` | NUMERIC(8,4) | `late_orders / orders` |

---

### `mart.category_sales_monthly`
**Produced by:** `etl/build_category_sales.py`

| Column | Type | Description |
|---|---|---|
| `order_month` | DATE | Calendar month |
| `product_category_name` | TEXT | English product category name |
| `orders` | INT | Distinct orders containing items in this category |
| `items` | INT | Total item count |
| `revenue` | NUMERIC(18,2) | Total revenue (price + freight) |

---

### `mart.review_monthly`
**Produced by:** `etl/build_review_monthly.py`

| Column | Type | Description |
|---|---|---|
| `review_month` | DATE | Calendar month |
| `reviews` | INT | Total reviews received |
| `avg_review_score` | NUMERIC(8,2) | Mean review score (1–5) |
| `pct_1_star` | NUMERIC(8,4) | Fraction of 1-star reviews |
| `pct_5_star` | NUMERIC(8,4) | Fraction of 5-star reviews |

---

### `mart.customer_churn_ml_dataset`
**Produced by:** `etl/build_customer_churn_ml_dataset.py`

Labelled ML training dataset. The default `CUTOFF_DATE` is `2018-06-01`.

| Column | Type | Description |
|---|---|---|
| `snapshot_date` | DATE | The `CUTOFF_DATE` value used when building features (e.g., `2018-06-01`) |
| `customer_unique_id` | TEXT | Unique customer identifier |
| `recency_days` | INT | Days between cutoff and last delivered order |
| `frequency` | INT | Distinct delivered orders before cutoff |
| `monetary` | NUMERIC(18,2) | Total spend before cutoff |
| `avg_delivery_days` | NUMERIC(10,2) | Average actual delivery days |
| `late_rate` | NUMERIC(10,4) | Share of late orders (0–1) |
| `avg_review_score` | NUMERIC(10,2) | Average review score (1–5) |
| `customer_state` | TEXT | Brazilian state code |
| `will_reorder_90d` | INT | **Target label** – 1 if customer reordered within 90 days after cutoff |

---

### `mart.customer_churn_features_snapshot`
**Produced by:** `etl/build_customer_churn_features_snapshot.py`

Unlabelled feature snapshot for live dashboard scoring. The default `SNAPSHOT_DATE` is `2018-08-01`.

Identical columns to `mart.customer_churn_ml_dataset` **except** `will_reorder_90d` is absent.

| Column | Type | Description |
|---|---|---|
| `snapshot_date` | DATE | Snapshot date |
| `customer_unique_id` | TEXT | Unique customer identifier |
| `recency_days` | INT | Days since last delivered order |
| `frequency` | INT | Distinct delivered orders |
| `monetary` | NUMERIC(18,2) | Total spend |
| `avg_delivery_days` | NUMERIC(10,2) | Average actual delivery days |
| `late_rate` | NUMERIC(10,4) | Share of late orders |
| `avg_review_score` | NUMERIC(10,2) | Average review score |
| `customer_state` | TEXT | Brazilian state code |

---

### `mart.customer_churn_predictions_log`
**Written by:** `dashboard/streamlit_app.py` (on user request to save predictions)

Audit log of every prediction run. Each run is identified by a unique `run_id` (UUID).

| Column | Type | Description |
|---|---|---|
| `id` | BIGSERIAL PK | Auto-incrementing surrogate key |
| `run_id` | TEXT | UUID identifying a batch of predictions saved together |
| `source` | TEXT | `"snapshot"`, `"single"`, or `"batch_upload"` |
| `snapshot_date` | DATE | Snapshot date used for DB-sourced predictions |
| `model_name` | TEXT | `"RandomForest"` or `"PyTorch NN"` |
| `threshold` | NUMERIC(8,4) | Probability threshold used for risk banding |
| `customer_unique_id` | TEXT | Customer identifier |
| `recency_days` | NUMERIC | Feature value |
| `frequency` | NUMERIC | Feature value |
| `monetary` | NUMERIC | Feature value |
| `avg_delivery_days` | NUMERIC | Feature value |
| `late_rate` | NUMERIC | Feature value |
| `avg_review_score` | NUMERIC | Feature value |
| `customer_state` | TEXT | Feature value |
| `reorder_proba_90d` | NUMERIC(10,6) | Predicted reorder probability |
| `risk_bucket` | TEXT | `"High"`, `"Medium"`, or `"Low"` |
| `created_at` | TIMESTAMP | Row insertion timestamp (default: `NOW()`) |

**DDL:**

```sql
CREATE TABLE IF NOT EXISTS mart.customer_churn_predictions_log (
    id                BIGSERIAL PRIMARY KEY,
    run_id            TEXT,
    source            TEXT,
    snapshot_date     DATE,
    model_name        TEXT,
    threshold         NUMERIC(8,4),
    customer_unique_id TEXT,
    recency_days      NUMERIC,
    frequency         NUMERIC,
    monetary          NUMERIC,
    avg_delivery_days NUMERIC,
    late_rate         NUMERIC,
    avg_review_score  NUMERIC,
    customer_state    TEXT,
    reorder_proba_90d NUMERIC(10,6),
    risk_bucket       TEXT,
    created_at        TIMESTAMP DEFAULT NOW()
);
```

---

## How Prediction Logging Works

1. The user clicks **"Save predictions to DB"** in one of the Streamlit AI tabs.
2. `dashboard/streamlit_app.py` generates a UUID `run_id` with `uuid.uuid4()`.
3. Feature columns plus model outputs (`reorder_proba_90d`, `risk_bucket`) are assembled into a DataFrame with metadata columns (`run_id`, `source`, `snapshot_date`, `model_name`, `threshold`, `created_at`).
4. The DataFrame is written to `mart.customer_churn_predictions_log` using `df.to_sql(if_exists="append")`.
5. The **Prediction Log** tab loads this table back with `pd.read_sql(...)` and allows filtering by `run_id`, `model_name`, and `risk_bucket`.

---

## Schema Setup

Run once in PostgreSQL before the first ETL execution:

```sql
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS mart;
```

All mart tables are created with `IF NOT EXISTS` DDL. See the full DDL in the root [README.md](../README.md#8-setup--installation).
