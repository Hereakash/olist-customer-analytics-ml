# Project Flowchart

This document describes the end-to-end flow of the Olist Customer Analytics & ML project, from raw data ingestion through to the Streamlit dashboard.

---

## Full Project Flowchart

```mermaid
flowchart TD
    A["📂 Olist Dataset CSVs\n9 CSV files in data/"]

    subgraph ETL ["⚙️ ETL Layer"]
        B["etl/ingest_raw.py\nLoad CSVs → PostgreSQL raw schema"]
        C[(raw schema\nPostgreSQL)]

        D["etl/build_rfm.py"]
        E["etl/build_rfm_segments.py"]
        F["etl/build_customer_segment_geo.py"]
        G["etl/build_cohort_retention.py"]
        H["etl/build_delivery_performance.py"]
        I["etl/build_category_sales.py"]
        J["etl/build_review_monthly.py"]
        K["etl/build_customer_churn_ml_dataset.py\nlabelled, cutoff 2018-06-01"]
        L["etl/build_customer_churn_features_snapshot.py\nunlabelled, snapshot 2018-08-01"]
    end

    subgraph DB ["🗄️ mart schema (PostgreSQL)"]
        M[(customer_rfm_snapshot)]
        N[(customer_rfm_scored)]
        O[(customer_segments_snapshot)]
        P[(customer_segment_geo_snapshot)]
        Q[(cohort_retention)]
        R[(delivery_performance_monthly)]
        S[(category_sales_monthly)]
        T[(review_monthly)]
        U[(customer_churn_ml_dataset)]
        V[(customer_churn_features_snapshot)]
        W[(customer_churn_predictions_log)]
    end

    subgraph ML ["🤖 ML Training"]
        X["ml/train_churn_model.py\nLogisticRegression + RandomForest"]
        Y["ml/train_churn_torch_nn.py\nPyTorch MLP — 4 layers"]
        XA["models/churn_model.joblib\nmodels/churn_metrics.json"]
        YA["models/churn_torch_nn.pt\nmodels/churn_torch_bundle.joblib\nmodels/churn_torch_metrics.json\nreports/*.png"]
    end

    APP["🖥️ Streamlit Dashboard\ndashboard/streamlit_app.py"]

    A --> B --> C

    C --> D --> M --> E --> N & O --> F --> P
    C --> G --> Q
    C --> H --> R
    C --> I --> S
    C --> J --> T
    C --> K --> U --> X --> XA
                  U --> Y --> YA
    C --> L --> V

    P & Q & R & S & T & V & XA & YA --> APP
    APP -->|"Save predictions"| W
```

---

## Dashboard User Flow

```mermaid
flowchart TD
    Start([User opens Streamlit app]) --> SB[Sidebar: select filters\nstate / segment / category\nmodel / threshold / snapshot date]

    SB --> AN[Analytics sections]
    SB --> AI[AI / ML sections]

    subgraph AN [📊 Analytics Pages]
        AN1[KPI tiles\ntotal customers, avg monetary/frequency/recency]
        AN2[RFM Segments\nsegment distribution bar chart\ntop states, top-50 customers by spend]
        AN3[Cohort Retention\nmonthly heatmap, retention rate table]
        AN4[Delivery Performance\nactual vs estimated days, late-rate by month]
        AN5[Category Sales\ntop-15 categories, monthly revenue trend]
        AN6[Customer Satisfaction\navg review score trend, 1-star/5-star %, scatter]
    end

    subgraph AI [🤖 AI / ML Tabs]
        T1[Tab 1: Snapshot Scoring\nLoad mart.customer_churn_features_snapshot\nScore all customers\nRisk-band chart + histogram + insights]
        T2[Tab 2: Single Customer\nManual input form\nInstant probability + risk badge + suggestions]
        T3[Tab 3: Batch Upload\nDownload CSV template\nUpload CSV → score all rows\nResults + insights]
        T4[Tab 4: Prediction Log\nBrowse mart.customer_churn_predictions_log\nFilter by run_id / model / risk bucket]
    end

    T1 -->|User clicks Save| LOG[(mart.customer_churn_predictions_log)]
    T2 -->|User clicks Save| LOG
    T3 -->|User clicks Save| LOG
    T4 --> LOG
```

---

## ETL Execution Order

The ETL scripts must be run in the following order. All scripts are idempotent (safe to rerun).

```mermaid
flowchart LR
    S1["1. ingest_raw.py\nCSVs → raw schema"] -->
    S2["2. build_rfm.py\nRFM metrics"] -->
    S3["3. build_rfm_segments.py\nQuintile scores + segments"] -->
    S4["4. build_customer_segment_geo.py\nAdd state/city to segments"] -->
    S5["5. build_cohort_retention.py\nCohort × period retention"]

    S1 --> S6["6. build_delivery_performance.py\nDelivery KPIs by month"]
    S1 --> S7["7. build_category_sales.py\nRevenue by category + month"]
    S1 --> S8["8. build_review_monthly.py\nReview scores by month"]
    S1 --> S9["9. build_customer_churn_ml_dataset.py\nLabelled training set"]
    S1 --> S10["10. build_customer_churn_features_snapshot.py\nUnlabelled scoring set"]
```

> Steps 5–10 all depend only on the raw schema (step 1). Steps 3 and 4 depend on previous mart tables.

---

## ML Training Flow

```mermaid
flowchart TD
    DS[(mart.customer_churn_ml_dataset\nsnapshot_date = 2018-06-01)]

    DS --> SPLIT["Stratified train/test split\n75% train | 25% test"]

    SPLIT --> RF["ml/train_churn_model.py\n① LogisticRegression\n② RandomForest (n=400)"]
    SPLIT --> NN["ml/train_churn_torch_nn.py\nPyTorch MLP\n4-layer: 256→128→64→1\nBCEWithLogitsLoss + Adam"]

    RF -->|best PR-AUC| RFO["models/churn_model.joblib\nmodels/churn_metrics.json"]
    NN --> NNO["models/churn_torch_nn.pt\nmodels/churn_torch_bundle.joblib\nmodels/churn_torch_metrics.json\nreports/*.png"]

    RFO & NNO --> DASH["dashboard/streamlit_app.py\nModel selected at runtime"]

    FEAT[(mart.customer_churn_features_snapshot\nsnapshot_date = 2018-08-01)]
    FEAT --> DASH
    DASH --> PRED["Predictions\nreorder_proba_90d\nrisk_bucket: High/Medium/Low"]
    PRED -->|Save| LOG[(mart.customer_churn_predictions_log)]
```
