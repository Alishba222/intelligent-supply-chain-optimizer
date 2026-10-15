#paRT A - Data Extract (Postgres + Mongo) - USING LOCAL CSV FILES WITH CORRECT PATHS
import pandas as pd
import pymongo
from sqlalchemy import create_engine
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Script directory: {script_dir}")

# 1. Try to read from PostgreSQL, fallback to CSV
try:
    engine = create_engine('postgresql://postgres:alayafatima2025@localhost:5432/supplychain_db')
    postgres_df = pd.read_sql("SELECT * FROM dataco_structured", engine)
    print(f"PostgreSQL connection successful: {postgres_df.shape[0]} rows loaded")
    
    # If 0 rows, fall back to CSV
    if postgres_df.shape[0] == 0:
        raise Exception("PostgreSQL returned 0 rows - using CSV instead")
        
except Exception as e:
    print(f"PostgreSQL failed: {e}")
    print("Loading from CSV file instead...")
    try:
        csv_path = os.path.join(script_dir, 'dataco_postgresql_structured.csv')
        print(f"Trying: {csv_path}")
        postgres_df = pd.read_csv(csv_path, low_memory=False)
        print(f"CSV loaded successfully: {postgres_df.shape[0]} rows")
    except Exception as e2:
        print(f"CSV load failed: {e2}")
        postgres_df = pd.DataFrame()

# 2. Connect to MongoDB and fetch contextual logs
try:
    mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
    mongo_db = mongo_client["supply_chain_context"]
    mongo_collection = mongo_db["iot_logs"]
    
    # Convert MongoDB data into a Pandas table
    mongo_data = list(mongo_collection.find())
    mongo_df = pd.DataFrame(mongo_data)
    print(f"MongoDB connection successful: {mongo_df.shape[0]} rows loaded")
    
    # If 0 rows, fall back to CSV
    if mongo_df.shape[0] == 0:
        raise Exception("MongoDB returned 0 rows - using CSV instead")
        
except Exception as e:
    print(f"MongoDB failed: {e}")
    print("Loading from CSV file instead...")
    try:
        csv_path = os.path.join(script_dir, 'dataco_mongodb_contextual.csv')
        print(f"Trying: {csv_path}")
        mongo_df = pd.read_csv(csv_path, low_memory=False)
        print(f"CSV loaded successfully: {mongo_df.shape[0]} rows")
    except Exception as e2:
        print(f"CSV load failed: {e2}")
        mongo_df = pd.DataFrame()

#Part B - Transform (Continuous to Categorical Labels)
# Function to convert numbers into categories (Discretization)
def discretize_sensor_data(row):
    # If temperature is above 40, mark as high risk[cite: 1]
    if 'Temperature' in row and row['Temperature'] > 40:
        return 'HIGH_TEMP'
    return 'NORMAL_TEMP'

# Apply the labels to the MongoDB data
mongo_df['Temp_Status'] = mongo_df.apply(discretize_sensor_data, axis=1)

#part C - Merge (The Join Key)
# Merge the two tables using the common ID[cite: 1]
print("MongoDB Columns:", mongo_df.columns.tolist())
print("PostgreSQL Columns:", postgres_df.columns.tolist())

# Find matching columns between the two dataframes
postgres_cols = postgres_df.columns.tolist()
mongo_cols = mongo_df.columns.tolist()

print(f"\nPostgreSQL has {postgres_df.shape[0]} rows, {len(postgres_cols)} columns")
print(f"MongoDB has {mongo_df.shape[0]} rows, {len(mongo_cols)} columns")

# Build clean token key for deterministic SQL-to-NoSQL join.
def normalize_asset_key(series):
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
    )


if postgres_df.shape[0] > 0 and mongo_df.shape[0] > 0:
    # If PostgreSQL has no Asset_ID, use Order Id as a surrogate token key on both sides.
    if 'Asset_ID' in postgres_df.columns and 'Asset_ID' in mongo_df.columns:
        postgres_key_col = 'Asset_ID'
        mongo_key_col = 'Asset_ID'
    elif 'Order Id' in postgres_df.columns and 'Order Id' in mongo_df.columns:
        postgres_key_col = 'Order Id'
        mongo_key_col = 'Order Id'
    else:
        postgres_key_col = None
        mongo_key_col = None

    if postgres_key_col is None or mongo_key_col is None:
        print("ERROR: Could not find join key columns. Need Asset_ID or Order Id in both datasets.")
        integrated_data = pd.DataFrame()
    else:
        if postgres_key_col != 'Asset_ID' or mongo_key_col != 'Asset_ID':
            print(f"Using fallback key mapping: PostgreSQL='{postgres_key_col}', MongoDB='{mongo_key_col}'")

        postgres_df['Asset_ID'] = normalize_asset_key(postgres_df[postgres_key_col])
        mongo_df['Asset_ID'] = normalize_asset_key(mongo_df[mongo_key_col])

        # Remove empty/null-like keys.
        postgres_df = postgres_df[~postgres_df['Asset_ID'].isin(['', 'NAN', 'NONE', 'NULL'])].copy()
        mongo_df = mongo_df[~mongo_df['Asset_ID'].isin(['', 'NAN', 'NONE', 'NULL'])].copy()

        # Avoid row explosion by keeping one NoSQL row per key.
        mongo_before = len(mongo_df)
        mongo_df = mongo_df.drop_duplicates(subset=['Asset_ID'], keep='first').copy()
        print(f"MongoDB duplicate keys removed: {mongo_before - len(mongo_df)}")

        # Left join keeps SQL side complete while mapping NoSQL metrics.
        integrated_data = pd.merge(
            postgres_df,
            mongo_df,
            on='Asset_ID',
            how='left',
            suffixes=('_x', '_y'),
            indicator=True,
        )

        successful_matches = int((integrated_data['_merge'] == 'both').sum())
        print(f"SUCCESSFUL MATCHES on Asset_ID: {successful_matches}")
        integrated_data = integrated_data.drop(columns=['_merge'])
else:
    print(f"ERROR: PostgreSQL empty ({postgres_df.shape[0]} rows) or MongoDB empty ({mongo_df.shape[0]} rows)")
    integrated_data = pd.DataFrame()

print(f"Rows after merge: {integrated_data.shape[0]} (SQL rows after key cleaning: {postgres_df.shape[0]})")

# Fill missing values from unmatched MongoDB records
print("Imputing missing sensor values...")
for col in ['Temperature', 'Humidity']:
    if col in integrated_data.columns:
        median_val = integrated_data[col].median()
        integrated_data[col] = integrated_data[col].fillna(median_val)
        print(f"  - {col}: filled {integrated_data[col].isnull().sum()} nulls with median {median_val:.2f}")

# Re-apply temperature discretization after imputation
if 'Temperature' in integrated_data.columns:
    integrated_data['Temp_Status'] = integrated_data.apply(discretize_sensor_data, axis=1)
    print(f"Temperature re-discretized:")
    print(f"  - HIGH_TEMP: {(integrated_data['Temp_Status'] == 'HIGH_TEMP').sum()}")
    print(f"  - NORMAL_TEMP: {(integrated_data['Temp_Status'] == 'NORMAL_TEMP').sum()}\n")

# Remove duplicate or unnecessary columns (like MongoDB's internal _id)
if '_id' in integrated_data.columns:
    integrated_data = integrated_data.drop(columns=['_id'])

# Save the final table as a CSV file for the next step[cite: 1]
integrated_data.to_csv("integrated_data_warehouse.csv", index=False)
print("SUCCESS: integrated_data_warehouse.csv has been created!")