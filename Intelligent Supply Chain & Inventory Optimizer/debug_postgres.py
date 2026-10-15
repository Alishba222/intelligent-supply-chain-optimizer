import pandas as pd
from sqlalchemy import create_engine, inspect
import psycopg2

print("=== PostgreSQL DEBUG ===\n")

try:
    # Connect to PostgreSQL
    engine = create_engine('postgresql://postgres:turkkhan123!@localhost:5432/supplychain_db')
    
    # List all tables
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    print(f"Found {len(tables)} tables in database:")
    for table in tables:
        print(f"  - {table}")
    
    # Check each table for row count
    print("\n=== Table Row Counts ===")
    for table in tables:
        try:
            df = pd.read_sql(f"SELECT COUNT(*) FROM {table}", engine)
            count = df.iloc[0, 0]
            print(f"  {table}: {count} rows")
            
            # Show first few rows if has data
            if count > 0:
                df_sample = pd.read_sql(f"SELECT * FROM {table} LIMIT 3", engine)
                print(f"    Columns: {df_sample.columns.tolist()[:5]}...")
        except Exception as e:
            print(f"  {table}: ERROR - {e}")
    
    print("\n=== Attempting Direct Query ===")
    # Try the original query
    try:
        df = pd.read_sql("SELECT * FROM dataco_structured", engine)
        print(f"dataco_structured: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        print(f"dataco_structured query failed: {e}")
        
except Exception as e:
    print(f"Connection failed: {e}")
    print("PostgreSQL may not be running or credentials wrong")
