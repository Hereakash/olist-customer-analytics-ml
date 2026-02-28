# Olist Customer Analytics & ML

> **End-to-end customer analytics platform** built on the [Olist Brazilian e-commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).  
> Covers the full data-engineering → analytics → machine-learning → dashboard lifecycle.

---

## Project Overview

This project turns the raw Olist CSV files into a fully-featured analytics and ML platform:

| Layer | What it does |
|---|---|
| **ETL** | Ingests 9 raw CSV files into PostgreSQL; builds 10+ analytics mart tables |
| **Analytics** | RFM segmentation, cohort retention, delivery performance, category sales, review trends |
| **Machine Learning** | Binary reorder/churn prediction (will the customer reorder within 90 days?) using RandomForest and a PyTorch MLP |
| **Dashboard** | Interactive Streamlit app with filters, charts, single-customer prediction, batch CSV scoring, and a persistent prediction log |

**Key features**

- Rerunnable, idempotent ETL scripts (safe to run multiple times)
- Two ML models selectable at runtime: scikit-learn RandomForest and a PyTorch MLP
- Adjustable probability threshold with risk banding (High / Medium / Low)
- Batch CSV upload for scoring your own customer data
- Every prediction run is logged to the database with a unique `run_id` for auditability

---

## Tech Stack

| Category | Libraries / Tools |
|---|---|
| Language | Python 3.10+ |
| Database | PostgreSQL (≥ 13) |
| ORM / DB connector | SQLAlchemy, psycopg2-binary |
| Data processing | pandas, numpy |
| Machine learning | scikit-learn (RandomForest, LogisticRegression, pipelines, preprocessors) |
| Deep learning | PyTorch (MLP, BCEWithLogitsLoss, Adam) |
| Model persistence | joblib |
| Visualisation | Plotly |
| Dashboard | Streamlit |
| Configuration | python-dotenv |

---

## Directory Structure

```
olist-customer-analytics-ml/
│
├── data/                          # ← raw Olist CSV files (git-ignored)
│
├── etl/                           # ETL scripts (run in order)
│   ├── ingest_raw.py              # Load CSVs → raw schema
│   ├── build_rfm.py               # RFM metrics snapshot
│   ├── build_rfm_segments.py      # RFM scoring + segment labels
│   ├── build_customer_segment_geo.py
│   ├── build_cohort_retention.py
│   ├── build_delivery_performance.py
│   ├── build_category_sales.py
│   ├── build_review_monthly.py
│   ├── build_customer_churn_ml_dataset.py
│   └── build_customer_churn_features_snapshot.py
│
├── ml/                            # ML training scripts
│   ├── train_churn_model.py       # RandomForest + LogisticRegression
│   └── train_churn_torch_nn.py    # PyTorch MLP
│
├── models/                        # Saved model artefacts
├── reports/                       # Training diagnostic plots
│
├── dashboard/
│   └── streamlit_app.py           # Main Streamlit application
│
├── docs/                          # Detailed documentation
│   ├── DATABASE.md                # DB schema, connection, prediction logging
│   ├── ER_DIAGRAM.md              # Mermaid ER diagrams (raw + mart schemas)
│   └── FLOWCHART.md               # Mermaid flowcharts (ETL, ML, dashboard)
│
├── config/                        # Runtime configuration (git-ignored)
│   └── settings.env               # DB credentials (copy from .env.example)
│
├── .env.example                   # Template for settings.env
├── requirements.txt               # Python dependencies
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL 13+ (local or remote)
- Olist CSV files in `data/` (download from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce))

### 1. Install dependencies

```bash
# Optional but recommended: create a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure database credentials

```bash
mkdir -p config
cp .env.example config/settings.env
# Edit config/settings.env and fill in DB_HOST, DB_NAME, DB_USER, DB_PASSWORD
```

> ⚠️ **Never commit `config/settings.env`** — it is listed in `.gitignore`.  
> The `data/` folder is also git-ignored; raw CSV files must be downloaded separately.

### 3. Set up the database schemas

Run the following SQL once in PostgreSQL:

```sql
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS mart;
```

For full mart table DDL, see [docs/DATABASE.md](docs/DATABASE.md) or the setup SQL below in the [Database Setup](#database-setup) section.

### 4. Run the ETL pipeline

```bash
python etl/ingest_raw.py
python etl/build_rfm.py
python etl/build_rfm_segments.py
python etl/build_customer_segment_geo.py
python etl/build_cohort_retention.py
python etl/build_delivery_performance.py
python etl/build_category_sales.py
python etl/build_review_monthly.py
python etl/build_customer_churn_ml_dataset.py
python etl/build_customer_churn_features_snapshot.py
```

### 5. Train ML models (optional)

Pre-trained model artefacts are included in `models/`. To retrain:

```bash
python ml/train_churn_model.py
python ml/train_churn_torch_nn.py
```

### 6. Launch the dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

---

## Configuration Reference

Copy `.env.example` to `config/settings.env` and fill in your values.

| Variable | Default | Description |
|---|---|---|
| `DB_HOST` | `localhost` | PostgreSQL host |
| `DB_PORT` | `5432` | PostgreSQL port |
| `DB_NAME` | *(required)* | Database name |
| `DB_USER` | *(required)* | Database user |
| `DB_PASSWORD` | *(required)* | Database password |

ML cutoff/snapshot dates can be edited directly in the ETL scripts (`CUTOFF_DATE`, `SNAPSHOT_DATE`).

---

## Database Setup

Run this SQL to create all mart tables before running ETL:

```sql
CREATE TABLE IF NOT EXISTS mart.customer_rfm_snapshot (
    snapshot_date DATE, customer_unique_id TEXT,
    recency_days INT, frequency INT, monetary NUMERIC(18,2),
    last_purchase_timestamp TIMESTAMP, first_purchase_timestamp TIMESTAMP, tenure_days INT
);
CREATE TABLE IF NOT EXISTS mart.customer_rfm_scored (
    snapshot_date DATE, customer_unique_id TEXT,
    recency_days INT, frequency INT, monetary NUMERIC(18,2),
    r_score INT, f_score INT, m_score INT, rfm_score TEXT
);
CREATE TABLE IF NOT EXISTS mart.customer_segments_snapshot (
    snapshot_date DATE, customer_unique_id TEXT, segment_name TEXT,
    r_score INT, f_score INT, m_score INT,
    recency_days INT, frequency INT, monetary NUMERIC(18,2)
);
CREATE TABLE IF NOT EXISTS mart.customer_segment_geo_snapshot (
    snapshot_date DATE, customer_unique_id TEXT, segment_name TEXT,
    customer_state TEXT, customer_city TEXT,
    monetary NUMERIC(18,2), frequency INT, recency_days INT
);
CREATE TABLE IF NOT EXISTS mart.cohort_retention (
    cohort_month DATE, order_month DATE, period_number INT,
    customers INT, cohort_size INT, retention_rate NUMERIC(8,4)
);
CREATE TABLE IF NOT EXISTS mart.delivery_performance_monthly (
    order_month DATE, orders INT,
    avg_delivery_days NUMERIC(8,2), avg_estimated_days NUMERIC(8,2),
    late_orders INT, late_rate NUMERIC(8,4)
);
CREATE TABLE IF NOT EXISTS mart.category_sales_monthly (
    order_month DATE, product_category_name TEXT,
    orders INT, items INT, revenue NUMERIC(18,2)
);
CREATE TABLE IF NOT EXISTS mart.review_monthly (
    review_month DATE, reviews INT,
    avg_review_score NUMERIC(8,2), pct_1_star NUMERIC(8,4), pct_5_star NUMERIC(8,4)
);
CREATE TABLE IF NOT EXISTS mart.customer_churn_ml_dataset (
    snapshot_date DATE, customer_unique_id TEXT,
    recency_days INT, frequency INT, monetary NUMERIC(18,2),
    avg_delivery_days NUMERIC(10,2), late_rate NUMERIC(10,4),
    avg_review_score NUMERIC(10,2), customer_state TEXT,
    will_reorder_90d INT
);
CREATE TABLE IF NOT EXISTS mart.customer_churn_features_snapshot (
    snapshot_date DATE, customer_unique_id TEXT,
    recency_days INT, frequency INT, monetary NUMERIC(18,2),
    avg_delivery_days NUMERIC(10,2), late_rate NUMERIC(10,4),
    avg_review_score NUMERIC(10,2), customer_state TEXT
);
CREATE TABLE IF NOT EXISTS mart.customer_churn_predictions_log (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT, source TEXT, snapshot_date DATE,
    model_name TEXT, threshold NUMERIC(8,4),
    customer_unique_id TEXT,
    recency_days NUMERIC, frequency NUMERIC, monetary NUMERIC,
    avg_delivery_days NUMERIC, late_rate NUMERIC, avg_review_score NUMERIC,
    customer_state TEXT, reorder_proba_90d NUMERIC(10,6), risk_bucket TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## Documentation

Detailed documentation is in the [`docs/`](docs/) directory:

| Document | Contents |
|---|---|
| [docs/DATABASE.md](docs/DATABASE.md) | DB connection approach, all table/column descriptions, prediction logging flow |
| [docs/ER_DIAGRAM.md](docs/ER_DIAGRAM.md) | Mermaid ER diagrams for raw and mart schemas |
| [docs/FLOWCHART.md](docs/FLOWCHART.md) | Mermaid flowcharts: ETL order, ML training, dashboard user flow |

---

> **Data privacy:** Raw dataset files (`data/`) and database credentials (`config/settings.env`) are excluded from version control via `.gitignore`.