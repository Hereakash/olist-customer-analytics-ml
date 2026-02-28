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


def assign_segment(r, f, m):
    # Business-friendly segmentation rules
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    if r >= 4 and f >= 3:
        return "Loyal Customers"
    if r == 5 and f <= 2:
        return "New Customers"
    if r <= 2 and f >= 3:
        return "At Risk"
    if r == 1 and f == 1:
        return "Hibernating"
    if r <= 2 and m >= 4:
        return "Can't Lose Them"
    if r >= 3 and m >= 3:
        return "Potential Loyalists"
    return "Need Attention"


def main():
    if not all([DB_NAME, DB_USER, DB_PASSWORD]):
        raise ValueError(
            "Missing DB config. Check config/settings.env (DB_NAME, DB_USER, DB_PASSWORD)."
        )

    url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(url)

    # Show exactly which DB we are writing to (very important)
    with engine.connect() as conn:
        db = conn.execute(text("SELECT current_database()")).scalar_one()
        user = conn.execute(text("SELECT current_user")).scalar_one()
        print("CONNECTED TO:")
        print(" database:", db)
        print(" user    :", user)

        # sanity: see how many rows exist in source RFM
        rfm_rows = conn.execute(text("SELECT COUNT(*) FROM mart.customer_rfm_snapshot")).scalar_one()
        print("mart.customer_rfm_snapshot rows:", rfm_rows)

    # Load latest snapshot from mart.customer_rfm_snapshot
    rfm = pd.read_sql(
        """
        SELECT *
        FROM mart.customer_rfm_snapshot
        WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM mart.customer_rfm_snapshot)
        """,
        engine,
    )

    if rfm.empty:
        raise RuntimeError(
            "mart.customer_rfm_snapshot is empty. Run build_rfm.py again and confirm it inserted rows."
        )

    # R score: lower recency_days is better => higher score
    rfm["r_score"] = pd.qcut(rfm["recency_days"], 5, labels=[5, 4, 3, 2, 1]).astype(int)

    # F and M: higher is better
    # Use rank(method="first") to avoid qcut errors due to many ties
    rfm["frequency_rank"] = rfm["frequency"].rank(method="first")
    rfm["monetary_rank"] = rfm["monetary"].rank(method="first")

    rfm["f_score"] = pd.qcut(rfm["frequency_rank"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary_rank"], 5, labels=[1, 2, 3, 4, 5]).astype(int)

    rfm["rfm_score"] = (
        rfm["r_score"].astype(str) + rfm["f_score"].astype(str) + rfm["m_score"].astype(str)
    )

    rfm["segment_name"] = [
        assign_segment(r, f, m) for r, f, m in zip(rfm["r_score"], rfm["f_score"], rfm["m_score"])
    ]

    snapshot_date = rfm["snapshot_date"].iloc[0]

    scored = rfm[
        [
            "snapshot_date",
            "customer_unique_id",
            "recency_days",
            "frequency",
            "monetary",
            "r_score",
            "f_score",
            "m_score",
            "rfm_score",
        ]
    ].copy()

    segments = rfm[
        [
            "snapshot_date",
            "customer_unique_id",
            "segment_name",
            "r_score",
            "f_score",
            "m_score",
            "recency_days",
            "frequency",
            "monetary",
        ]
    ].copy()

    # Delete old snapshot and insert new (rerunnable)
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM mart.customer_rfm_scored WHERE snapshot_date = :d"),
            {"d": snapshot_date},
        )
        conn.execute(
            text("DELETE FROM mart.customer_segments_snapshot WHERE snapshot_date = :d"),
            {"d": snapshot_date},
        )

    scored.to_sql(
        "customer_rfm_scored",
        engine,
        schema="mart",
        if_exists="append",
        index=False,
        method="multi",
        chunksize=50_000,
    )
    segments.to_sql(
        "customer_segments_snapshot",
        engine,
        schema="mart",
        if_exists="append",
        index=False,
        method="multi",
        chunksize=50_000,
    )

    print(f"\nDONE: snapshot_date={snapshot_date}")
    print("Inserted rows into mart.customer_rfm_scored:", len(scored))
    print("Inserted rows into mart.customer_segments_snapshot:", len(segments))
    print("\nSegment counts:")
    print(segments["segment_name"].value_counts())


if __name__ == "__main__":
    main()