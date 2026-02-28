# ER Diagram

This document shows the entity-relationship structure for the Olist project across two layers:

1. **Raw schema** – tables loaded 1-to-1 from the nine Olist CSV files.
2. **Mart schema** – analytics and ML tables derived by ETL scripts.

---

## Raw Schema ER Diagram

The nine source CSV files map to the following entities and relationships in the `raw` PostgreSQL schema.

```mermaid
erDiagram
    olist_customers_dataset {
        string customer_id PK
        string customer_unique_id
        string customer_zip_code_prefix
        string customer_city
        string customer_state
    }

    olist_orders_dataset {
        string    order_id PK
        string    customer_id FK
        string    order_status
        timestamp order_purchase_timestamp
        timestamp order_approved_at
        timestamp order_delivered_carrier_date
        timestamp order_delivered_customer_date
        timestamp order_estimated_delivery_date
    }

    olist_order_items_dataset {
        string order_id FK
        int    order_item_id
        string product_id FK
        string seller_id FK
        float  price
        float  freight_value
    }

    olist_order_payments_dataset {
        string order_id FK
        int    payment_sequential
        string payment_type
        int    payment_installments
        float  payment_value
    }

    olist_products_dataset {
        string product_id PK
        string product_category_name FK
        int    product_name_lenght
        int    product_photos_qty
        float  product_weight_g
    }

    olist_order_reviews_dataset {
        string    review_id PK
        string    order_id FK
        int       review_score
        timestamp review_creation_date
        timestamp review_answer_timestamp
    }

    olist_sellers_dataset {
        string seller_id PK
        string seller_zip_code_prefix
        string seller_city
        string seller_state
    }

    olist_geolocation_dataset {
        string geolocation_zip_code_prefix
        float  geolocation_lat
        float  geolocation_lng
        string geolocation_city
        string geolocation_state
    }

    product_category_name_translation {
        string product_category_name PK
        string product_category_name_english
    }

    olist_customers_dataset        ||--o{ olist_orders_dataset          : "places"
    olist_orders_dataset           ||--o{ olist_order_items_dataset     : "contains"
    olist_orders_dataset           ||--o{ olist_order_payments_dataset  : "paid via"
    olist_orders_dataset           ||--o{ olist_order_reviews_dataset   : "reviewed in"
    olist_order_items_dataset      }o--|| olist_products_dataset        : "refers to"
    olist_order_items_dataset      }o--|| olist_sellers_dataset         : "sold by"
    olist_products_dataset         }o--|| product_category_name_translation : "translates"
```

> **Note:** `olist_geolocation_dataset` is not directly joined in the ETL scripts; it is available for optional geo enrichment by matching on `zip_code_prefix`.

---

## Mart Schema ER Diagram

The mart tables are derived aggregates. Primary/foreign keys are logical (not enforced as DB constraints), and most tables are written with `pd.to_sql(if_exists="replace")`.

```mermaid
erDiagram
    customer_rfm_snapshot {
        date   snapshot_date
        string customer_unique_id PK
        int    recency_days
        int    frequency
        float  monetary
        ts     last_purchase_timestamp
        ts     first_purchase_timestamp
        int    tenure_days
    }

    customer_rfm_scored {
        date   snapshot_date
        string customer_unique_id PK
        int    recency_days
        int    frequency
        float  monetary
        int    r_score
        int    f_score
        int    m_score
        string rfm_score
    }

    customer_segments_snapshot {
        date   snapshot_date
        string customer_unique_id PK
        string segment_name
        int    r_score
        int    f_score
        int    m_score
        int    recency_days
        int    frequency
        float  monetary
    }

    customer_segment_geo_snapshot {
        date   snapshot_date
        string customer_unique_id PK
        string segment_name
        string customer_state
        string customer_city
        float  monetary
        int    frequency
        int    recency_days
    }

    cohort_retention {
        date cohort_month PK
        date order_month PK
        int  period_number
        int  customers
        int  cohort_size
        float retention_rate
    }

    delivery_performance_monthly {
        date  order_month PK
        int   orders
        float avg_delivery_days
        float avg_estimated_days
        int   late_orders
        float late_rate
    }

    category_sales_monthly {
        date   order_month PK
        string product_category_name PK
        int    orders
        int    items
        float  revenue
    }

    review_monthly {
        date  review_month PK
        int   reviews
        float avg_review_score
        float pct_1_star
        float pct_5_star
    }

    customer_churn_ml_dataset {
        date   snapshot_date PK
        string customer_unique_id PK
        int    recency_days
        int    frequency
        float  monetary
        float  avg_delivery_days
        float  late_rate
        float  avg_review_score
        string customer_state
        int    will_reorder_90d
    }

    customer_churn_features_snapshot {
        date   snapshot_date PK
        string customer_unique_id PK
        int    recency_days
        int    frequency
        float  monetary
        float  avg_delivery_days
        float  late_rate
        float  avg_review_score
        string customer_state
    }

    customer_churn_predictions_log {
        bigint id PK
        string run_id
        string source
        date   snapshot_date
        string model_name
        float  threshold
        string customer_unique_id
        float  recency_days
        float  frequency
        float  monetary
        float  avg_delivery_days
        float  late_rate
        float  avg_review_score
        string customer_state
        float  reorder_proba_90d
        string risk_bucket
        ts     created_at
    }

    customer_rfm_snapshot      ||--|| customer_rfm_scored              : "scored from"
    customer_rfm_scored        ||--|| customer_segments_snapshot       : "segmented into"
    customer_segments_snapshot ||--|| customer_segment_geo_snapshot    : "geo-enriched into"
    customer_churn_features_snapshot ||--o{ customer_churn_predictions_log : "scored into"
```

---

## Notes

- Composite primary keys are `(snapshot_date, customer_unique_id)` for RFM and ML tables; `(cohort_month, order_month)` for cohort retention; `(order_month, product_category_name)` for category sales.
- No foreign key constraints are enforced in the database; relationships are maintained at the application/ETL layer.
- For full column DDL, see the [Database documentation](DATABASE.md) and the setup SQL in [README.md](../README.md#8-setup--installation).
