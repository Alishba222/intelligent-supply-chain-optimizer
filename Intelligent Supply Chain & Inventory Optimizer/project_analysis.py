import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("INTELLIGENT SUPPLY CHAIN & INVENTORY OPTIMIZER - PROJECT ANALYSIS")
print("=" * 80)

# 1. DATA QUALITY ASSESSMENT
print("\n\n1. DATA QUALITY ASSESSMENT")
print("-" * 80)

try:
    pg_df = pd.read_csv('dataco_postgresql_structured.csv')
    print(f"\n✓ PostgreSQL Data:")
    print(f"  - Rows: {pg_df.shape[0]}")
    print(f"  - Columns: {pg_df.shape[1]}")
    print(f"  - Column names: {pg_df.columns.tolist()[:5]}... (showing first 5)")
    print(f"  - Data types:\n{pg_df.dtypes}")
    print(f"  - Missing values: {pg_df.isnull().sum().sum()} ({pg_df.isnull().sum().sum()/pg_df.size*100:.2f}%)")
    print(f"  - Duplicates: {pg_df.duplicated().sum()}")
except Exception as e:
    print(f"✗ Error reading PostgreSQL: {e}")

try:
    mongo_df = pd.read_csv('dataco_mongodb_contextual.csv')
    print(f"\n✓ MongoDB Data:")
    print(f"  - Rows: {mongo_df.shape[0]}")
    print(f"  - Columns: {mongo_df.shape[1]}")
    print(f"  - Column names: {mongo_df.columns.tolist()}")
    print(f"  - Missing values: {mongo_df.isnull().sum().sum()} ({mongo_df.isnull().sum().sum()/mongo_df.size*100:.2f}%)")
    print(f"  - Duplicates: {mongo_df.duplicated().sum()}")
except Exception as e:
    print(f"✗ Error reading MongoDB: {e}")

# 2. ETL PIPELINE VALIDATION
print("\n\n2. ETL PIPELINE VALIDATION")
print("-" * 80)

try:
    integrated = pd.read_csv('integrated_data_warehouse.csv')
    print(f"\n✓ Integrated Data Warehouse:")
    print(f"  - Rows: {integrated.shape[0]}")
    print(f"  - Columns: {integrated.shape[1]}")
    print(f"  - Missing values: {integrated.isnull().sum().sum()} ({integrated.isnull().sum().sum()/integrated.size*100:.2f}%)")
    print(f"  - Duplicates: {integrated.duplicated().sum()}")
    print(f"  - Data merge success: {'✓' if integrated.shape[0] > 0 else '✗'}")
    
    # Check for Temp_Status column (transformation applied)
    if 'Temp_Status' in integrated.columns:
        print(f"  - Temperature discretization: ✓")
        print(f"    * HIGH_TEMP: {(integrated['Temp_Status'] == 'HIGH_TEMP').sum()}")
        print(f"    * NORMAL_TEMP: {(integrated['Temp_Status'] == 'NORMAL_TEMP').sum()}")
    else:
        print(f"  - Temperature discretization: ✗ (Missing Temp_Status column)")
        
except Exception as e:
    print(f"✗ Error reading Integrated Data: {e}")

# 3. DATA MINING RESULTS
print("\n\n3. DATA MINING RESULTS")
print("-" * 80)

try:
    insights = pd.read_csv('final_delivery_insights.csv')
    print(f"\n✓ Final Delivery Insights:")
    print(f"  - Patterns Found: {insights.shape[0]}")
    print(f"  - Average Support: {insights['support'].mean():.4f}")
    print(f"  - Average Confidence: {insights['confidence'].mean():.4f}")
    print(f"  - Average Lift: {insights['lift'].mean():.4f}")
    print(f"  - Min/Max Support: {insights['support'].min():.4f} / {insights['support'].max():.4f}")
    print(f"  - Min/Max Confidence: {insights['confidence'].min():.4f} / {insights['confidence'].max():.4f}")
    print(f"  - Min/Max Lift: {insights['lift'].min():.4f} / {insights['lift'].max():.4f}")
    
    if insights.shape[0] > 0:
        print(f"\n  Top 3 Patterns by Confidence:")
        top_patterns = insights.nlargest(3, 'confidence')[['antecedents', 'consequents', 'confidence', 'support', 'lift']]
        for idx, row in top_patterns.iterrows():
            print(f"    • {row['antecedents']} → {row['consequents']}")
            print(f"      Confidence: {row['confidence']:.4f}, Support: {row['support']:.4f}, Lift: {row['lift']:.4f}")
except Exception as e:
    print(f"✗ Error reading Final Insights: {e}")

# 4. ACCURACY METRICS ANALYSIS
print("\n\n4. ACCURACY & VALIDATION")
print("-" * 80)

print(f"\n✓ Data Consistency Checks:")
print(f"  - ETL merge preservation: {('✓' if integrated.shape[0] >= min(pg_df.shape[0], mongo_df.shape[0]) else '✗')}")
print(f"  - Pattern discovery coverage: {('✓' if insights.shape[0] > 0 else '✗')}")
print(f"  - Thresholds met:")
print(f"    * Support ≥ 5%: {('✓' if (insights['support'] >= 0.05).all() else '✗')}")
print(f"    * Confidence ≥ 30%: {('✓' if (insights['confidence'] >= 0.30).all() else '✗')}")
print(f"    * Lift > 1.0: {('✓' if (insights['lift'] > 1.0).all() else '✗')}")

# 5. ISSUES & CONCERNS
print("\n\n5. ISSUES & CONCERNS DETECTED")
print("-" * 80)

issues = []

# Check data quality
if pg_df.isnull().sum().sum() / pg_df.size * 100 > 5:
    issues.append("⚠ PostgreSQL has >5% missing values - data quality concern")

if mongo_df.isnull().sum().sum() / mongo_df.size * 100 > 5:
    issues.append("⚠ MongoDB has >5% missing values - data quality concern")

# Check mining results
if insights['lift'].mean() < 1.1:
    issues.append("⚠ Average lift is low (~1.0) - weak pattern associations")

if insights['confidence'].mean() < 0.4:
    issues.append("⚠ Average confidence is low (<40%) - rules may not be reliable")

# Check data loss in merge
data_loss = ((min(pg_df.shape[0], mongo_df.shape[0]) - integrated.shape[0]) / min(pg_df.shape[0], mongo_df.shape[0]) * 100)
if data_loss > 10:
    issues.append(f"⚠ Data loss in ETL merge: {data_loss:.2f}% - check join logic")

# Check classifier coverage
if integrated['Doc_Category'].nunique() < 5:
    issues.append(f"⚠ Low document category diversity: only {integrated['Doc_Category'].nunique()} categories used")

if len(issues) == 0:
    print("✓ No critical issues detected!")
else:
    for issue in issues:
        print(issue)

# 6. RECOMMENDATIONS FOR ENHANCEMENT
print("\n\n6. ENHANCEMENT RECOMMENDATIONS")
print("-" * 80)

recommendations = [
    "1. DATA VALIDATION:",
    "   • Add null/missing value imputation strategy",
    "   • Implement outlier detection for sensor data (Temperature, Humidity)",
    "   • Add data type validation on import",
    "",
    "2. ETL IMPROVEMENTS:",
    "   • Add data lineage tracking for debugging",
    "   • Implement logging for merge operations",
    "   • Use inner/outer join strategies comparison",
    "",
    "3. MINING ENHANCEMENTS:",
    "   • Increase min_support threshold for stronger patterns",
    "   • Apply multiple algorithms (Apriori, Eclat) for comparison",
    "   • Add temporal analysis (time-series patterns)",
    "",
    "4. CLASSIFICATION IMPROVEMENT:",
    "   • Use ML models (NLP) instead of keyword matching",
    "   • Add confidence scores to classifications",
    "   • Train on labeled dataset",
    "",
    "5. ACCURACY TRACKING:",
    "   • Implement cross-validation for pattern rules",
    "   • Add confusion matrix for classification",
    "   • Track pattern precision/recall metrics",
    "",
    "6. VISUALIZATION & REPORTING:",
    "   • Add data quality dashboard",
    "   • Create pattern correlation heatmaps",
    "   • Generate PDF reports with insights",
]

for rec in recommendations:
    print(rec)

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
