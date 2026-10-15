import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# 1. Load the data
file_path = r"N:\UNIVERSITY PROJECTS\Intelligent Supply Chain & Inventory Optimizer\Intelligent Supply Chain & Inventory Optimizer\integrated_data_warehouse.csv"
df = pd.read_csv(file_path, low_memory=False)

# 2. Add Target Labels (AI needs to see 'LATE' vs 'ON_TIME' instead of 1/0)
# Checking for both names just in case
risk_col = 'Late_delivery_risk_x' if 'Late_delivery_risk_x' in df.columns else 'Late_delivery_risk'

if risk_col in df.columns:
    df[risk_col] = df[risk_col].replace({1: 'LATE', 0: 'ON_TIME'})

# 3. Define the Strategic Mining Columns
mining_columns = [
    'Temp_Status', 
    'Doc_Category', 
    risk_col, 
    'Shipping Mode_x', 
    'Order Region_x'
]

# Filter available columns
available_cols = [col for col in mining_columns if col in df.columns]
print(f"Mining insights from: {available_cols}")

mining_df = df[available_cols]

# 4. Data Preprocessing
# Fill missing values and ensure everything is a string
mining_df_clean = mining_df.fillna('UNKNOWN').astype(str)

# Convert to transaction list
transactions = mining_df_clean.values.tolist()

# 5. Encoding
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
encoded_df = pd.DataFrame(te_ary, columns=te.columns_).astype(bool)

# 6. FP-Growth (Searching for patterns)
# Using min_support=0.05 (5% frequency) for more meaningful results
frequent_itemsets = fpgrowth(encoded_df, min_support=0.05, use_colnames=True)

# 7. Generate Association Rules
# We use a higher confidence threshold (0.3) to find stronger correlations
if not frequent_itemsets.empty:
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.1)
    
    # FILTER: Specifically look for rules that lead to 'LATE' status
    if not rules.empty:
        # We want patterns where 'LATE' is the result (consequent)
        interesting_rules = rules[rules['consequents'].astype(str).str.contains('LATE')]
        
        if interesting_rules.empty:
            interesting_rules = rules.sort_values(by='lift', ascending=False).head(10)

        print("\n--- SUCCESS: Supply Chain Intelligence Found! ---")
        print(interesting_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
        
        # Save results for evaluation.py
        interesting_rules.to_csv('mined_patterns_raw.csv', index=False)
    else:
        print("No strong rules found. Try lowering min_threshold.")
else:
    print("No frequent itemsets found. Try lowering min_support.")