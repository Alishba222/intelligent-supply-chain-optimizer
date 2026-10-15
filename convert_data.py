import os
from pathlib import Path
import polars as pl

BASE_DIR = Path(__file__).resolve().parent
SOURCE_CSV = BASE_DIR / 'Intelligent Supply Chain & Inventory Optimizer' / 'integrated_data_warehouse.csv'
TARGET_PARQUET = BASE_DIR / 'Intelligent Supply Chain & Inventory Optimizer' / 'integrated_data_warehouse.parquet'


def main():
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"Source CSV not found: {SOURCE_CSV}")

    print(f"Loading CSV from: {SOURCE_CSV}")
    
    # EXPLICITLY FORCE ORDER IDS TO STRING TO HANDLE ALPHANUMERIC TOKENS SECURELY

    # Force all ID columns to string to avoid dtype errors on mixed values
    df = pl.read_csv(
        SOURCE_CSV,
        try_parse_dates=True,
        infer_schema_length=10000,
        schema_overrides={
            "Order Id_x": pl.String,
            "Order Id_y": pl.String,
            "Asset_ID": pl.String,
            "Asset Id": pl.String,
            "asset_id": pl.String,
        }
    )

    text_columns = [col for col, dtype in zip(df.columns, df.dtypes) if dtype == pl.Utf8]
    if text_columns:
        print(f"Dictionary-encoding text columns: {len(text_columns)} columns")
        df = df.with_columns([pl.col(col).cast(pl.Categorical) for col in text_columns])

    print(f"Writing optimized Parquet to: {TARGET_PARQUET}")
    df.write_parquet(TARGET_PARQUET, compression='snappy')
    print('Conversion complete.')


if __name__ == '__main__':
    main()