"""Debug and fix SQL-to-NoSQL join for supply chain data.

This script:
1) Cleans Asset_ID in both datasets (trim/uppercase/string).
2) Resolves dtype mismatches before joining.
3) Performs a configurable join (left or inner) so IoT/sentiment fields map correctly.
4) Prints debug stats, including successful match count on Asset_ID.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SQL_PATH = BASE_DIR / "dataco_postgresql_structured.csv"
NOSQL_PATH = BASE_DIR / "dataco_mongodb_contextual.csv"
OUTPUT_PATH = BASE_DIR / "integrated_data_warehouse_fixed.csv"


def _build_asset_id(df: pd.DataFrame, dataset_name: str, preferred_col: str | None = None) -> pd.DataFrame:
    """Ensure dataframe has an Asset_ID column used as the join key."""
    out = df.copy()

    if preferred_col and preferred_col in out.columns:
        source_col = preferred_col
    elif "Asset_ID" in out.columns:
        source_col = "Asset_ID"
    elif "asset_id" in out.columns:
        source_col = "asset_id"
    elif "Order Id" in out.columns:
        # Fallback when source does not store Asset_ID directly.
        source_col = "Order Id"
        print(f"[{dataset_name}] Asset_ID missing, using 'Order Id' as fallback key.")
    else:
        raise KeyError(
            f"[{dataset_name}] No Asset_ID-like column found. Required one of: "
            "Asset_ID, asset_id, Order Id"
        )

    out["Asset_ID"] = (
        out[source_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Remove empty/null-like keys that break joins.
    out = out[~out["Asset_ID"].isin(["", "NAN", "NONE", "NULL"])].copy()

    return out


def _prepare_for_join(sql_df: pd.DataFrame, nosql_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize key type and remove duplicate right-side keys to avoid row explosion."""
    # If SQL side has no Asset_ID but both have Order Id, use Order Id on both sides
    # as a normalized Asset_ID surrogate for deterministic matching.
    sql_pref = None
    nosql_pref = None
    if "Asset_ID" not in sql_df.columns and "Order Id" in sql_df.columns and "Order Id" in nosql_df.columns:
        sql_pref = "Order Id"
        nosql_pref = "Order Id"
        print("[JOIN KEY] Using 'Order Id' on both datasets to build normalized Asset_ID.")

    left = _build_asset_id(sql_df, "SQL", preferred_col=sql_pref)
    right = _build_asset_id(nosql_df, "NoSQL", preferred_col=nosql_pref)

    # Explicit dtype alignment for join key.
    left["Asset_ID"] = left["Asset_ID"].astype("string")
    right["Asset_ID"] = right["Asset_ID"].astype("string")

    # Keep one NoSQL row per Asset_ID to prevent repeated Cartesian-like expansion.
    before = len(right)
    right = right.sort_values(by=["Asset_ID"]).drop_duplicates(subset=["Asset_ID"], keep="first")
    removed = before - len(right)
    if removed > 0:
        print(f"[NoSQL] Removed {removed} duplicate Asset_ID rows before join.")

    return left, right


def run_join(join_type: str = "left") -> pd.DataFrame:
    """Run the fixed join and print debug metrics."""
    if join_type not in {"left", "inner"}:
        raise ValueError("join_type must be 'left' or 'inner'")

    print("Loading datasets...")
    sql_df = pd.read_csv(SQL_PATH, low_memory=False)
    nosql_df = pd.read_csv(NOSQL_PATH, low_memory=False)

    print(f"SQL rows: {len(sql_df):,}")
    print(f"NoSQL rows: {len(nosql_df):,}")

    left, right = _prepare_for_join(sql_df, nosql_df)

    print(f"SQL keys (cleaned): {left['Asset_ID'].nunique():,}")
    print(f"NoSQL keys (cleaned): {right['Asset_ID'].nunique():,}")

    merged = left.merge(
        right,
        on="Asset_ID",
        how=join_type,
        suffixes=("_sql", "_nosql"),
        indicator=True,
    )

    # Count successful key matches.
    successful_matches = int((merged["_merge"] == "both").sum())
    print(f"SUCCESSFUL MATCHES on Asset_ID: {successful_matches:,}")

    # Extra diagnostics.
    left_only = int((merged["_merge"] == "left_only").sum())
    right_only = int((merged["_merge"] == "right_only").sum())
    print(f"Rows only in SQL side: {left_only:,}")
    print(f"Rows only in NoSQL side (visible in inner join diagnostics): {right_only:,}")

    merged = merged.drop(columns=["_merge"])
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved fixed joined dataset: {OUTPUT_PATH}")

    return merged


if __name__ == "__main__":
    # Start with left join so SQL records are preserved while mapping NoSQL metrics.
    result = run_join(join_type="left")

    # Quick visibility into mapped operational columns.
    sample_cols = [
        col
        for col in [
            "Asset_ID",
            "Traffic_Status",
            "Logistics_Delay_Reason",
            "Temperature",
            "Humidity",
            "Doc_Category",
        ]
        if col in result.columns
    ]

    if sample_cols:
        print("\nSample mapped rows:")
        print(result[sample_cols].head(10).to_string(index=False))
