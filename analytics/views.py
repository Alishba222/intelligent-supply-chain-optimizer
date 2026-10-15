import os
import csv
import json
import importlib.util
import subprocess
import threading
import sys
import time
import gc
import re
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
try:
    import polars as pl
except ImportError:
    pl = None
import plotly.express as px
import plotly.graph_objects as go
from django.shortcuts import render, redirect
from django.conf import settings
from django.http import JsonResponse
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.paginator import Paginator

# Machine Learning and Association Rule Frameworks
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Import Form and Chatbot Dependencies
from .forms import ChatbotForm, RealTimeAssetDataForm
from .models import RealTimeAssetData, ClassificationMetrics

# Inline check to gracefully load the custom chatbot module from the nested project folder.
chatbot_module_path = Path(settings.BASE_DIR) / 'Intelligent Supply Chain & Inventory Optimizer' / 'nlp_chatbot.py'
try:
    if chatbot_module_path.exists():
        chatbot_spec = importlib.util.spec_from_file_location('supply_chain_chatbot_impl', chatbot_module_path)
        if chatbot_spec and chatbot_spec.loader:
            chatbot_module = importlib.util.module_from_spec(chatbot_spec)
            chatbot_spec.loader.exec_module(chatbot_module)
            SupplyChainChatbot = chatbot_module.SupplyChainChatbot
        else:
            SupplyChainChatbot = None
    else:
        SupplyChainChatbot = None
except Exception:
    SupplyChainChatbot = None


def build_objectives_phases():
    return [
        {
            "phase": "Phase 1: ETL Pipeline Data Loading & Integration",
            "items": [
                {"id": 1, "objective": "Verify the successful loading of PostgreSQL dataset from CSV fallback after database connection failure.", "answer": "Output confirmed: 'CSV loaded successfully: 181,519 rows'. ETL pipeline auto-recovered from PostgreSQL connection error to supplychain_db."},
                {"id": 2, "objective": "Verify the successful loading of MongoDB dataset from CSV fallback after zero-row connection.", "answer": "Output confirmed: 'CSV loaded successfully: 66,752 rows'. MongoDB returned 0 rows via live connection; CSV fallback activated automatically."},
                {"id": 3, "objective": "Determine the total column count loaded for each dataset source in the ETL pipeline.", "answer": "PostgreSQL: 49 columns loaded; MongoDB: 30 columns loaded (including the engineered 'Temp_Status' column added during ETL)."},
                {"id": 4, "objective": "Analyse the sampling strategy applied to PostgreSQL data during dataset integration.", "answer": "ETL sampled every 181st PostgreSQL record, producing 1,003 representative rows for cross-dataset enrichment."},
                {"id": 5, "objective": "Evaluate the size of the integrated enriched data warehouse created by the ETL pipeline.", "answer": "Output confirmed: 'Created enriched dataset: 1,003,000 rows'. MongoDB's 66,752 records cross-joined with 1,003 PostgreSQL samples."},
                {"id": 6, "objective": "Assess the outcome of the final merge and confirm row preservation from the original PostgreSQL dataset.", "answer": "Output confirmed: 'Rows after merge: 1,003,000 (preserved from PostgreSQL: 181,519)'. All original records retained in the warehouse matrix."}
            ]
        },
        {
            "phase": "Phase 2: ETL Pipeline Data Quality & Imputation",
            "items": [
                {"id": 7, "objective": "Identify the columns that required missing value imputation during the ETL pipeline execution.", "answer": "Two IoT sensor columns required data imputation: Temperature (1,003,000 nulls filled) and Humidity (1,003,000 nulls filled) using median strategy."},
                {"id": 8, "objective": "Evaluate the imputation result for the Temperature column and identify the data quality issue.", "answer": "Output shows: 'Temperature: filled 1,003,000 nulls with median nan'. The median was NaN, indicating all Temperature values were missing in the enriched dataset."},
                {"id": 9, "objective": "Evaluate the imputation result for the Humidity column and identify the data quality issue.", "answer": "Output shows: 'Humidity: filled 1,003,000 nulls with median nan'. Same issue as Temperature; all 1,003,000 rows had null Humidity after enrichment."},
                {"id": 10, "objective": "Analyse the Temperature rediscretization output and determine the distribution of temperature categories.", "answer": "Output confirms: HIGH_TEMP: 0 records, NORMAL_TEMP: 1,003,000 records. Entire dataset classified as NORMAL_TEMP due to NaN imputation filling all values uniformly."},
                {"id": 11, "objective": "Assess the impact of the 'Temp_Status' engineered feature on the MongoDB dataset column count.", "answer": "MongoDB columns increased from 29 original columns to 30 post-ETL columns due to the addition of the engineered Temp_Status categorical feature."},
                {"id": 12, "objective": "Confirm successful creation of the integrated data warehouse file.", "answer": "Output confirmed: 'SUCCESS: integrated data warehouse.csv has been created!'. Final output file successfully written to disk after all ETL steps."}
            ]
        },
        {
            "phase": "Phase 3: Association Rule Mining Supply Chain Intelligence",
            "items": [
                {"id": 13, "objective": "Identify the features selected for association rule mining in the supply chain mining engine.", "answer": "Mining engine used 5 features: Temp_Status, Doc_Category, Late_delivery_risk_x, Shipping Mode_x, and Order Region_x."},
                {"id": 14, "objective": "Determine the strongest association rule discovered and interpret its business meaning.", "answer": "{Second Class} => {LATE} with Support=0.1446, Confidence=73.98%, Lift=1.377. Second Class shipping causes late delivery 74% of the time."},
                {"id": 15, "objective": "Analyse the support value of the top association rules and explain its significance.", "answer": "Support 0.1446 means 14.46% of all orders in the dataset follow the Second Class => LATE pattern, representing a statistically significant proportion."},
                {"id": 16, "objective": "Interpret the confidence value of 73.98% in the context of the top association rule.", "answer": "Confidence of 73.98% means whenever an order uses Second Class shipping, there is a 73.98% probability it will result in a late delivery."},
                {"id": 17, "objective": "Interpret the lift value of 1.377 found in the top association rules.", "answer": "Lift=1.377 (> 1.0) confirms a genuine positive association. Second Class shipping increases the likelihood of late delivery by 37.7% above random chance."},
                {"id": 18, "objective": "Identify all rule variants that produced identical support, confidence, and lift values.", "answer": "8 rules share identical metrics (support=0.1446, confidence=73.98%, lift=1.377). All are Second Class + UNKNOWN/NORMAL_TEMP combinations, confirming Temp_Status does not affect delay."},
                {"id": 19, "objective": "Analyse the association rule involving Central America region and interpret its implication.", "answer": "{Central America} => {LATE} with Support=0.0887, Confidence=59.73%, Lift=1.112. Central American orders have a 59.73% late delivery rate, 11.2% above baseline."},
                {"id": 20, "objective": "Compare the confidence of Second Class shipping vs Central America region for late delivery prediction.", "answer": "Second Class (73.98%) is significantly stronger than Central America (59.73%). Shipping Mode is a more reliable predictor of delay than Order Region."}
            ]
        },
        {
            "phase": "Phase 4: Machine Learning Delay Predictor Model",
            "items": [
                {"id": 21, "objective": "Identify the target variable auto-detected by the Delay Predictor model.", "answer": "Output confirmed: 'Auto-Detected Target: Late delivery risk x'. The ML model automatically identified the correct binary classification target."},
                {"id": 22, "objective": "Determine the class distribution of the target variable before balancing.", "answer": "Output: 539,000 Late orders vs 464,000 On-Time orders. Dataset was slightly imbalanced (53.7% Late, 46.3% On-Time) before balancing."},
                {"id": 23, "objective": "Evaluate the data balancing strategy applied before Random Forest training.", "answer": "Output: 'Training on Balanced Data: 928,000 total samples (50/50 split)'. Equal Late/On-Time samples used to eliminate class bias during training."},
                {"id": 24, "objective": "Determine the size of the training set used for Random Forest model training.", "answer": "Output: 'Training Random Forest on 742,400 samples'. 80% of the 928,000 balanced samples used for training (20% held for validation)."},
                {"id": 25, "objective": "Report the final accuracy achieved by the Random Forest Delay Predictor model.", "answer": "Output confirmed: 'Model Training Complete. Accuracy: 57.48%'. Model performs 7.48% above random chance (50%) on balanced binary classification."},
                {"id": 26, "objective": "Assess the scale of inference performed by the trained Delay Predictor model.", "answer": "Output confirmed: 'AI has predicted delays for the entire dataset!'. Model scored all 1,000,000+ orders and saved results to all_order_predictions_full.csv."}
            ]
        },
        {
            "phase": "Phase 5: Delay Prediction Order-Level Risk Scores",
            "items": [
                {"id": 27, "objective": "Identify the range of delay percentage risk scores produced by the model across sampled orders.", "answer": "Predicted delay percentages range from 0.0% (no risk) to 100.0% (certain delay). Model produces a full probability spectrum, not just binary labels."},
                {"id": 28, "objective": "Identify orders with extreme risk scores (0% and 100%) and interpret their meaning.", "answer": "Orders 653334 and 2140 scored 0.0% (no delay risk); Order 482271 scored 100.0% (certain delay). Model identified both safe and critical orders."},
                {"id": 29, "objective": "Analyse the clustering of predicted delay scores and determine the most common risk band.", "answer": "Majority of sampled orders cluster between 40%-56% risk, indicating most orders fall in the moderate uncertainty zone, consistent with 57.48% model accuracy."},
                {"id": 30, "objective": "Confirm the final output file generated by the Delay Predictor and its business utility.", "answer": "Output confirmed: 'Full report saved as: all_order_predictions_full.csv'. Every order in the 1M+ dataset has an individual AI-generated delay risk score for operational use."}
            ]
        }
    ]


# --- DATA HOOK: GENERATE HYBRID WAREHOUSE DATA (GRAPH ALIGNMENT MATCH) ---
def load_hybrid_warehouse_data():
    np.random.seed(42)
    records = 100
    regions = ['Karachi Central', 'Karachi East', 'Bin Qasim Town', 'Gulshan-e-Hadeed', 'Saddar Port']
    dates = pd.date_range(start="2026-01-01", periods=records, freq='h')
    
    data = {
        'Timestamp': dates,
        'order date (DateOrders)_x': dates,
        'Asset_ID': np.random.choice(['Truck_1', 'Truck_6', 'Truck_7', 'Truck_9', 'Truck_10'], records),
        'Region': np.random.choice(regions, records),
        'Order Region_x': np.random.choice(regions, records),
        'Shipment_Status': np.random.choice(['Delivered', 'In Transit', 'Delayed'], records, p=[0.6, 0.2, 0.2]),
        'Delivery Status_x': np.random.choice(['Delivered', 'In Transit', 'Delayed'], records, p=[0.6, 0.2, 0.2]),
        'Temperature': np.random.uniform(18.0, 42.0, records),
        'iot_temperature': np.random.uniform(18.0, 42.0, records),
        'Humidity': np.random.uniform(40.0, 95.0, records),
        'Traffic_Volume_Index': np.random.uniform(10, 100, records),
        'Traffic_Status': np.random.uniform(10, 100, records),
        'Waiting_Time_Hours': np.random.uniform(2.0, 15.0, records),
        'Days for shipping (real)_x': np.random.uniform(1.0, 5.0, records),
        'Document_Genre': np.random.choice(['Emotions', 'Adventure', 'Religion', 'Sports'], records),
        'Doc_Category': np.random.choice(['Emotions', 'Adventure', 'Religion', 'Sports'], records)
    }
    return pd.DataFrame(data)

PARQUET_FILE_PATH = os.path.join(settings.BASE_DIR, 'Intelligent Supply Chain & Inventory Optimizer', 'integrated_data_warehouse.parquet')
CSV_FILE_PATH = os.path.join(settings.BASE_DIR, 'Intelligent Supply Chain & Inventory Optimizer', 'integrated_data_warehouse.csv')

# --- GLOBAL DATA CACHE (THREAD-SAFE MEMORY MANAGEMENT) ---
_GLOBAL_DF_CACHE = {}
_DF_LOCK = threading.Lock()

def get_global_df():
    """Returns the globally cached Polars LazyFrame, using Parquet scan for zero-copy execution."""
    with _DF_LOCK:
        if 'df' not in _GLOBAL_DF_CACHE:
            if pl is None:
                try:
                    data_path = os.path.join(settings.BASE_DIR, 'Intelligent Supply Chain & Inventory Optimizer', 'integrated_data_warehouse.csv')
                    _GLOBAL_DF_CACHE['df'] = pd.read_csv(data_path, low_memory=False)
                except Exception as e:
                    print(f"Error loading CSV: {e}. Falling back to synthetic data.")
                    _GLOBAL_DF_CACHE['df'] = load_hybrid_warehouse_data()
            else:
                try:
                    if os.path.exists(PARQUET_FILE_PATH):
                        _GLOBAL_DF_CACHE['df'] = pl.scan_parquet(PARQUET_FILE_PATH)
                    elif os.path.exists(CSV_FILE_PATH):
                        _GLOBAL_DF_CACHE['df'] = pl.scan_csv(CSV_FILE_PATH, try_parse_dates=True)
                    else:
                        fallback_df = load_hybrid_warehouse_data()
                        _GLOBAL_DF_CACHE['df'] = pl.from_pandas(fallback_df).lazy()
                except Exception as e:
                    print(f"Error loading data with Polars: {e}. Falling back to synthetic data.")
                    fallback_df = load_hybrid_warehouse_data()
                    _GLOBAL_DF_CACHE['df'] = pl.from_pandas(fallback_df).lazy()
    return _GLOBAL_DF_CACHE['df']


def _build_warehouse_filter_context_polars(frame, asset_filter='', status_filter='', region_filter='', page_number=1, page_size=50):
    if pl is None or not isinstance(frame, pl.LazyFrame):
        return _build_warehouse_filter_context(frame, asset_filter, status_filter, region_filter, page_number, page_size)

    asset_filter = (asset_filter or '').strip()
    status_filter = (status_filter or '').strip()
    region_filter = (region_filter or '').strip()

    def _build_search_expr(column_names, search_value, exact=False):
        expr = None
        search_value = search_value.strip().lower()
        for col_name in column_names:
            if col_name in frame.columns:
                target = pl.col(col_name).cast(pl.Utf8).str.to_lowercase().fill_null('')
                condition = target == search_value if exact else target.str.contains(search_value)
                expr = condition if expr is None else expr | condition
        return expr

    filtered_lazy = frame
    if asset_filter:
        expr = _build_search_expr(['Asset_ID', 'Order Id_x', 'Order Id', 'asset_id'], asset_filter, exact=False)
        if expr is not None:
            filtered_lazy = filtered_lazy.filter(expr)
    if status_filter:
        expr = _build_search_expr(['Shipment_Status', 'Delivery Status_x', 'logistics_status'], status_filter, exact=True)
        if expr is not None:
            filtered_lazy = filtered_lazy.filter(expr)
    if region_filter:
        expr = _build_search_expr(['Region', 'Order Region_x', 'operational_region'], region_filter, exact=True)
        if expr is not None:
            filtered_lazy = filtered_lazy.filter(expr)

    collected = filtered_lazy.collect()
    return _build_warehouse_filter_context(collected.to_pandas(), asset_filter, status_filter, region_filter, page_number, page_size)


def _normalize_filter_value(value):
    if isinstance(value, (list, tuple, set)):
        flattened_values = []
        for item in value:
            normalized_item = _normalize_filter_value(item)
            if normalized_item not in (None, '', 'N/A'):
                flattened_values.append(str(normalized_item))
        return ', '.join(flattened_values) if flattened_values else None
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text_value = str(value).strip()
    return text_value if text_value else None


def _first_existing_series(frame, candidates, default_value=np.nan):
    for column_name in candidates:
        if column_name in frame.columns:
            return frame[column_name]
    return pd.Series([default_value] * len(frame), index=frame.index)


def _build_warehouse_filter_context(frame, asset_filter='', status_filter='', region_filter='', page_number=1, page_size=50):
    display_frame = frame.copy()

    if 'Asset_ID' not in display_frame.columns:
        display_frame['Asset_ID'] = _first_existing_series(display_frame, ['Asset_ID', 'Order Id_x', 'Order Id', 'asset_id'], default_value='Fleet_Unit')
    if 'Shipment_Status' not in display_frame.columns:
        display_frame['Shipment_Status'] = _first_existing_series(display_frame, ['Shipment_Status', 'Delivery Status_x', 'logistics_status'], default_value='In Transit')
    if 'Region' not in display_frame.columns:
        display_frame['Region'] = _first_existing_series(display_frame, ['Region', 'Order Region_x', 'operational_region'], default_value='Global Node')
    if 'Temperature' not in display_frame.columns:
        display_frame['Temperature'] = _first_existing_series(display_frame, ['Temperature', 'Temperature_y', 'iot_temperature'], default_value=np.nan)
    if 'Traffic_Volume_Index' not in display_frame.columns:
        display_frame['Traffic_Volume_Index'] = _first_existing_series(display_frame, ['Traffic_Status', 'Traffic_Status_y', 'Traffic_Volume_Index', 'traffic_volume_scale'], default_value=np.nan)
    if 'Waiting_Time_Hours' not in display_frame.columns:
        waiting_series = _first_existing_series(display_frame, ['Waiting_Time_Hours', 'Days for shipping (real)_x', 'Days for shipping (real)'], default_value=np.nan)
        display_frame['Waiting_Time_Hours'] = pd.to_numeric(waiting_series, errors='coerce') * (24 if 'Days for shipping (real)_x' in display_frame.columns or 'Days for shipping (real)' in display_frame.columns else 1)
    if 'Document_Genre' not in display_frame.columns:
        display_frame['Document_Genre'] = _first_existing_series(display_frame, ['Document_Genre', 'Doc_Category', 'Logistics_Delay_Reason', 'driver_sentiment'], default_value='N/A')

    display_frame['timestamp'] = pd.to_datetime(_first_existing_series(display_frame, ['Timestamp', 'timestamp', 'order date (DateOrders)_x']), errors='coerce')
    display_frame['asset_id'] = display_frame['Asset_ID'].map(_normalize_filter_value)
    display_frame['operational_region'] = display_frame['Region'].map(_normalize_filter_value)
    display_frame['logistics_status'] = display_frame['Shipment_Status'].map(_normalize_filter_value)
    display_frame['iot_temperature'] = pd.to_numeric(_first_existing_series(display_frame, ['Temperature', 'Temperature_y', 'iot_temperature']), errors='coerce')
    display_frame['traffic_volume_scale'] = pd.to_numeric(_first_existing_series(display_frame, ['Traffic_Status', 'Traffic_Status_y', 'Traffic_Volume_Index', 'traffic_volume_scale']), errors='coerce')
    display_frame['waiting_latency_hours'] = pd.to_numeric(display_frame['Waiting_Time_Hours'], errors='coerce')
    display_frame['nlp_classified_genre'] = display_frame['Document_Genre'].map(_normalize_filter_value)

    filtered_frame = display_frame.copy()
    asset_filter = (asset_filter or '').strip()
    status_filter = (status_filter or '').strip()
    region_filter = (region_filter or '').strip()

    if asset_filter:
        filtered_frame = filtered_frame[filtered_frame['asset_id'].astype(str).str.contains(asset_filter, case=False, na=False)]
    if status_filter:
        filtered_frame = filtered_frame[filtered_frame['logistics_status'].astype(str).str.lower() == status_filter.lower()]
    if region_filter:
        filtered_frame = filtered_frame[filtered_frame['operational_region'].astype(str).str.lower() == region_filter.lower()]

    def clean_display_value(value, placeholder='-'):
        if value is None:
            return placeholder
        try:
            if pd.isna(value):
                return placeholder
        except Exception:
            pass
        text_value = str(value).strip()
        if text_value.lower() in {'nan', 'none', 'null', 'n/a', ''}:
            return placeholder
        return text_value

    if 'timestamp' in filtered_frame.columns:
        filtered_frame['timestamp'] = pd.to_datetime(filtered_frame['timestamp'], errors='coerce')
    else:
        filtered_frame['timestamp'] = pd.NaT

    filtered_frame['timestamp_display'] = filtered_frame['timestamp'].apply(
        lambda value: value.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(value) else 'N/A'
    )
    filtered_frame['asset_id_display'] = filtered_frame['asset_id'].apply(lambda value: clean_display_value(value, 'N/A'))
    filtered_frame['operational_region_display'] = filtered_frame['operational_region'].apply(lambda value: clean_display_value(value, 'N/A'))
    filtered_frame['logistics_status_display'] = filtered_frame['logistics_status'].apply(lambda value: clean_display_value(value, 'N/A'))
    filtered_frame['iot_temperature_display'] = pd.to_numeric(filtered_frame['iot_temperature'], errors='coerce').apply(
        lambda value: f'{value:.1f}' if pd.notna(value) else 'N/A'
    )
    filtered_frame['traffic_volume_scale_display'] = pd.to_numeric(filtered_frame['traffic_volume_scale'], errors='coerce').apply(
        lambda value: f'{value:.0f}%' if pd.notna(value) else 'N/A'
    )
    filtered_frame['waiting_latency_hours_display'] = pd.to_numeric(filtered_frame['waiting_latency_hours'], errors='coerce').apply(
        lambda value: f'{value:.2f}' if pd.notna(value) else 'N/A'
    )
    filtered_frame['nlp_classified_genre_display'] = filtered_frame['nlp_classified_genre'].apply(lambda value: clean_display_value(value, '-'))

    page_records = filtered_frame.to_dict('records')

    total_records = len(filtered_frame)
    paginator = Paginator(page_records, page_size)
    page_obj = paginator.get_page(page_number)

    unique_assets = sorted(
        value for value in {
            _normalize_filter_value(value)
            for value in display_frame['asset_id'].dropna().tolist()
        }
        if value
    )
    unique_status = sorted(
        value for value in {
            _normalize_filter_value(value)
            for value in display_frame['logistics_status'].dropna().tolist()
        }
        if value
    )
    unique_regions = sorted(
        value for value in {
            _normalize_filter_value(value)
            for value in display_frame['operational_region'].dropna().tolist()
        }
        if value
    )

    return {
        'page_obj': page_obj,
        'total_records': total_records,
        'asset_filter': asset_filter,
        'status_filter': status_filter,
        'region_filter': region_filter,
        'unique_assets': unique_assets,
        'unique_status': unique_status,
        'unique_regions': unique_regions,
        'filtered_frame': filtered_frame,
    }

# Background non-blocking thread initialization
threading.Thread(target=get_global_df, daemon=True).start()


def compute_and_store_metrics(df, data_source='manual_test'):
    """Run the classification pipeline on `df`, store a ClassificationMetrics record, and return summary dict."""
    try:
        ml_df = df.sample(n=15000, random_state=42) if len(df) > 15000 else df.copy()
        target_col = 'Shipment_Status' if 'Shipment_Status' in ml_df.columns else ('Delivery Status_x' if 'Delivery Status_x' in ml_df.columns else None)
        if target_col is None:
            return {'ok': False, 'message': 'No target column found for classification.'}

        # --- DATA LEAKAGE PREVENTION IN BACKGROUND FUNCTION ---
        cols_to_drop = ['Timestamp', 'timestamp', 'order date (DateOrders)_x', 'shipping date (DateOrders)', 'Days for shipping (real)_x', 'Waiting_Time_Hours', 'Days for shipping (real)', 'late_delivery_risk', 'Late_delivery_risk']
        for d_col in cols_to_drop:
            if d_col in ml_df.columns and d_col != target_col:
                ml_df = ml_df.drop(d_col, axis=1)

        le = LabelEncoder()
        ml_df[target_col] = le.fit_transform(ml_df[target_col].astype(str))

        for col in ml_df.columns:
            if not pd.api.types.is_numeric_dtype(ml_df[col]):
                try:
                    ml_df[col] = LabelEncoder().fit_transform(ml_df[col].astype(str))
                except Exception:
                    ml_df[col] = 0

        ml_df = ml_df.fillna(0)

        X = ml_df.drop(columns=[target_col])
        y = ml_df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)
        clf = RandomForestClassifier(n_estimators=80, max_depth=8, min_samples_split=4, class_weight='balanced', random_state=12)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accuracy_val = accuracy_score(y_test, y_pred)
        precision_val = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall_val = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_val = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        all_classes = np.unique(y)
        cm = confusion_matrix(y_test, y_pred, labels=all_classes)
        labels = le.inverse_transform(all_classes)

        unique_vals, counts = np.unique(y, return_counts=True)
        class_counts = {str(int(u)): int(c) for u, c in zip(unique_vals, counts)}
        notes_text = f"Samples: {len(ml_df)}, Train: {len(X_train)}, Test: {len(X_test)}, ClassCounts: {class_counts}"

        try:
            ClassificationMetrics.objects.create(
                accuracy=float(accuracy_val),
                precision=float(precision_val),
                recall=float(recall_val),
                f1_score=float(f1_val),
                total_samples=len(y_test),
                data_source=data_source,
                confusion_matrix_json=cm.tolist(),
                class_labels=list(labels),
                notes=notes_text
            )
        except Exception as e:
            print('Error storing realtime metrics:', e)

        return {
            'ok': True,
            'accuracy': f"{accuracy_val*100:.2f}%",
            'precision': f"{precision_val*100:.2f}%",
            'recall': f"{recall_val*100:.2f}%",
            'f1_score': f"{f1_val*100:.2f}%",
            'total_samples': len(y_test),
            'data_source': data_source,
            'confusion_matrix': cm.tolist(),
            'class_labels': list(labels),
            'notes': notes_text,
        }
    except Exception as e:
        return {'ok': False, 'message': str(e)}


# --- CHATBOT API ENGINE ---
@csrf_exempt
def chatbot_api(request):
    try:
        if request.method != 'POST':
            return JsonResponse({'answer': 'Use POST method for chatbot requests.', 'ok': False}, status=200)
        question = request.POST.get('question', '')
        if not question and request.body:
            try:
                payload = json.loads(request.body.decode('utf-8'))
                question = payload.get('question', '')
            except Exception:
                pass
        question = (question or '').strip()
        if not question:
            return JsonResponse({'answer': 'Please type a question so I can help you.', 'ok': True}, status=200)
        if SupplyChainChatbot is None:
            return JsonResponse({'answer': 'Supply Chain Chatbot engine is currently offline or unlinked.', 'ok': False}, status=200)

        chatbot = SupplyChainChatbot(df=get_global_df())
        answer = chatbot.answer_query(question)
        return JsonResponse({'answer': answer, 'ok': True}, status=200)
    except Exception as e:
        return JsonResponse({'answer': f'Chatbot runtime error: {e}', 'ok': False}, status=200)


# --- HOME / SPLASH ENTRANCE GATEWAY ---
def home_view(request):
    try:
        return render(request, 'home.html')
    except Exception:
        return JsonResponse({'status': 'ok', 'message': 'Welcome to the Supply Chain Dashboard (home template missing).'})


# --- CHATBOT TRADITIONAL INTERFACE VIEW ---
def chatbot_view(request):
    answer = None
    if SupplyChainChatbot is not None:
        chatbot = SupplyChainChatbot()
        try: chatbot.load_data()
        except Exception: pass
    else:
        chatbot = None

    if request.method == 'POST':
        form = ChatbotForm(request.POST)
        if form.is_valid():
            question = form.cleaned_data['question']
            if chatbot: answer = chatbot.answer_query(question)
            else: answer = "Service unlinked."
    else:
        form = ChatbotForm()
    return render(request, 'chatbot.html', {'form': form, 'answer': answer})


# --- CENTRAL DASHBOARD VIEW ---
def dashboard_view(request):
    df_lazy = get_global_df()
    if pl is not None and isinstance(df_lazy, pl.LazyFrame):
        df = df_lazy.collect()
    elif isinstance(df_lazy, pd.DataFrame):
        df = df_lazy.copy()
    else:
        df = df_lazy
    selected_objective = request.POST.get('selected_objective', request.GET.get('selected_objective', request.POST.get('objective', request.GET.get('objective', 'classification'))))
    total_records = len(df)
    unique_assets = []
    unique_status = []
    unique_regions = []
    asset_filter = ''
    status_filter = ''
    region_filter = ''
    
    # =========================================================
    # CORE TRACKER SHIELD INITIALIZATION BLOCKS
    # =========================================================
    file_status = request.GET.get('file_status', request.POST.get('file_status', ''))
    run_pipeline = 'run_pipeline' in request.POST or 'run_pipeline' in request.GET
    pipeline_message = ''
    pipeline_status = ''
    rules_list = []
    chart_html = ''
    confusion_matrix_html = ''
    
    accuracy_val = 0.0
    precision_val = 0.0
    recall_val = 0.0
    f1_val = 0.0
    
    classification_metrics = {
        'accuracy': 'N/A',
        'precision': 'N/A',
        'recall': 'N/A',
        'f1_score': 'N/A',
        'total_samples': 0,
        'data_source': 'N/A'
    }

    try: k_clusters = int(request.POST.get('k_clusters', request.GET.get('k_clusters', 5)))
    except Exception: k_clusters = 5
    try: min_support = float(request.POST.get('min_support', request.GET.get('min_support', 0.01)))
    except Exception: min_support = 0.01
    try: min_confidence = float(request.POST.get('min_confidence', request.GET.get('min_confidence', 0.6)))
    except Exception: min_confidence = 0.6
    
    display_records = int(request.POST.get('display_records', request.GET.get('display_records', 50)))
    metrics_filter = request.POST.get('metrics_filter', request.GET.get('metrics_filter', 'latency_performance'))
    metrics_filter_aliases = {
        'latency': 'latency_performance',
        'status': 'fleet_status',
        'temperature': 'temperature_analysis',
        'region': 'regional_breakdown',
    }
    metrics_filter = metrics_filter_aliases.get(metrics_filter, metrics_filter)
    realtime_form = RealTimeAssetDataForm()
    realtime_page_obj = None
    objectives_phases = build_objectives_phases()
    verified_milestones = 30
    total_ingested_rows = '1,003,000 rows'
    max_mining_confidence = '73.98%'
    balanced_ml_samples = '928,000 rows'

    # =========================================================
    # 1. CORE PIPELINE AUTOMATED AUTOMATION SEQUENCING
    # =========================================================
    if run_pipeline:
        try:
            project_dir = os.path.join(settings.BASE_DIR, 'Intelligent Supply Chain & Inventory Optimizer')
            python_exec = sys.executable or "python"
            # Clear any cached dataframe before running the external pipeline
            with _DF_LOCK:
                if 'df' in _GLOBAL_DF_CACHE:
                    del _GLOBAL_DF_CACHE['df']

            subprocess.run([python_exec, "data_validator.py"], cwd=project_dir, check=True)
            subprocess.run([python_exec, "etl_pipeline.py"], cwd=project_dir, check=True)
            subprocess.run([python_exec, "mining_engine.py"], cwd=project_dir, check=True)
            subprocess.run([python_exec, "evaluation.py"], cwd=project_dir, check=True)
            subprocess.run([python_exec, "sentiment_analyzer.py"], cwd=project_dir, check=True)
            subprocess.run([python_exec, "delay_predictor.py"], cwd=project_dir, check=True)
            # Refresh the in-memory cache after the pipeline completes
            refreshed_df = get_global_df()
            if pl is not None and isinstance(refreshed_df, pl.LazyFrame):
                df = refreshed_df.collect()
            elif isinstance(refreshed_df, pd.DataFrame):
                df = refreshed_df.copy()
            else:
                df = refreshed_df
            pipeline_status = "COMPLETED"
            pipeline_message = "✅ Automated AI Pipeline Data Engine Refreshed Successfully!"
            messages.success(request, pipeline_message)
            
        except Exception as e:
            pipeline_status = "FAILED"
            pipeline_message = f"❌ Pipeline Execution Error: {e}"
            messages.error(request, pipeline_message)

    # =========================================================
    # 2. RUNTIME USER CSV INGESTION PIPELINES
    # =========================================================
    if request.method == 'POST' and request.FILES.get('dataset'):
        try:
            new_df = pd.read_csv(request.FILES['dataset'])
            with _DF_LOCK:
                _GLOBAL_DF_CACHE['df'] = new_df
            df = new_df
            file_status = "SUCCESS"
            messages.success(request, "Dataset ingested and compiled into data warehouse layers successfully!")
            threading.Thread(target=compute_and_store_metrics, args=(df, 'csv_upload'), daemon=True).start()
        except Exception as e:
            file_status = "FAILED"
            messages.error(request, f"File Integration Fault: {e}")

    # TIME SERIES INTERFACE
    if selected_objective == 'time_series':
        time_col = next((col for col in ['order date (DateOrders)_x', 'Timestamp'] if col in df.columns), df.columns[0])
        wait_col = next((col for col in ['Days for shipping (real)_x', 'Waiting_Time_Hours'] if col in df.columns), df.columns[-1])

        if pl is not None and isinstance(df, pl.DataFrame):
            ts_df = df.select([time_col, wait_col]).clone()
            ts_df = ts_df.with_columns(
                pl.col(time_col).cast(pl.String).str.to_datetime("%m/%d/%Y %H:%M", strict=False).alias(time_col)
            )
            ts_df = ts_df.drop_nulls(subset=[time_col, wait_col])
            if len(ts_df) > 0:
                q_low = ts_df.select(pl.col(wait_col).quantile(0.01)).item()
                q_hi = ts_df.select(pl.col(wait_col).quantile(0.99)).item()
                ts_df = ts_df.filter((pl.col(wait_col) >= q_low) & (pl.col(wait_col) <= q_hi)).sort(time_col)
                time_series_df = ts_df.to_pandas()
                time_series_df[time_col] = pd.to_datetime(time_series_df[time_col], errors='coerce')
                time_series_df = time_series_df.set_index(time_col).resample('ME')[[wait_col]].mean().reset_index().dropna()
                if not time_series_df.empty:
                    fig = px.line(time_series_df, x=time_col, y=wait_col, markers=True, title="Timeline Trend Matrix: Logistics Latency Cycles")
                    fig.update_traces(line_color='#0ea5e9')
                    fig.update_layout(plot_bgcolor="white", xaxis=dict(showgrid=True, gridcolor="#e2e8f0"), yaxis=dict(showgrid=True, gridcolor="#e2e8f0"))
                    kpi = f"<b>Peak Delay:</b> {time_series_df[wait_col].max():.2f} days | <b>Mean Delay:</b> {time_series_df[wait_col].mean():.2f} days"
                    chart_html = kpi + fig.to_html(full_html=False, include_plotlyjs='cdn')
                else:
                    chart_html = '<div class="alert alert-warning">Insufficient data frames post-resampling.</div>'
        else:
            ts_df = df[[time_col, wait_col]].copy()
            ts_df[time_col] = pd.to_datetime(ts_df[time_col], errors='coerce')
            ts_df = ts_df.dropna(subset=[time_col, wait_col])
            if len(ts_df) > 0:
                q_low = ts_df[wait_col].quantile(0.01)
                q_hi = ts_df[wait_col].quantile(0.99)
                ts_df = ts_df[(ts_df[wait_col] >= q_low) & (ts_df[wait_col] <= q_hi)].sort_values(time_col)
                time_series_df = ts_df.set_index(time_col).resample('ME')[[wait_col]].mean().reset_index().dropna()
                if not time_series_df.empty:
                    fig = px.line(time_series_df, x=time_col, y=wait_col, markers=True, title="Timeline Trend Matrix: Logistics Latency Cycles")
                    fig.update_traces(line_color='#0ea5e9')
                    fig.update_layout(plot_bgcolor="white", xaxis=dict(showgrid=True, gridcolor="#e2e8f0"), yaxis=dict(showgrid=True, gridcolor="#e2e8f0"))
                    kpi = f"<b>Peak Delay:</b> {time_series_df[wait_col].max():.2f} days | <b>Mean Delay:</b> {time_series_df[wait_col].mean():.2f} days"
                    chart_html = kpi + fig.to_html(full_html=False, include_plotlyjs='cdn')
                else:
                    chart_html = '<div class="alert alert-warning">Insufficient data frames post-resampling.</div>'

    elif selected_objective == 'clustering':
        try:
            # Convert Polars to Pandas for sklearn clustering
            df_pandas = df.to_pandas() if isinstance(df, pl.DataFrame) else df
            numeric_cols = df_pandas.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                feature_cols = numeric_cols[:3]
                x_col = feature_cols[0]
                y_col = feature_cols[1]
                size_col = feature_cols[2] if len(feature_cols) > 2 else None

                asset_col = next((c for c in ['Asset_ID', 'ID'] if c in df_pandas.columns), None)
                region_col = next((c for c in ['Region', 'Order Region_x', 'Operational_Region'] if c in df_pandas.columns), None)
                status_col = next((c for c in ['Shipment_Status', 'Delivery Status_x'] if c in df_pandas.columns), None)

                categorical_cols = [c for c in [asset_col, region_col, status_col] if c]
                cluster_cols = list(dict.fromkeys([x_col, y_col] + ([size_col] if size_col else []) + categorical_cols))
                ml_cluster_df = df_pandas[cluster_cols].copy()

                def normalize_cluster_value(value):
                    if isinstance(value, (list, tuple, set)):
                        flattened_values = []
                        for item in value:
                            if isinstance(item, (list, tuple, set)):
                                flattened_values.extend([str(sub_item) for sub_item in item if pd.notna(sub_item)])
                            elif pd.notna(item):
                                flattened_values.append(str(item))
                        return ", ".join(flattened_values) if flattened_values else np.nan
                    if isinstance(value, dict):
                        return json.dumps(value, sort_keys=True)
                    return value

                for col in cluster_cols:
                    ml_cluster_df[col] = ml_cluster_df[col].map(normalize_cluster_value)

                for col in [x_col, y_col, size_col]:
                    if col and col in ml_cluster_df.columns:
                        ml_cluster_df[col] = pd.to_numeric(ml_cluster_df[col], errors='coerce')

                ml_cluster_df = ml_cluster_df.dropna(subset=[x_col, y_col])

                if len(ml_cluster_df) >= 10:
                    numeric_feature_df = ml_cluster_df[[c for c in [x_col, y_col, size_col] if c]].copy().fillna(0)
                    encoded_feature_df = pd.DataFrame(index=ml_cluster_df.index)

                    if categorical_cols:
                        for col in categorical_cols:
                            categorical_series = ml_cluster_df[col].astype(str).fillna('UNKNOWN')
                            encoded_feature_df[f'{col}_encoded'] = pd.factorize(categorical_series)[0].astype(float)

                    feature_matrix = pd.concat([numeric_feature_df, encoded_feature_df], axis=1).fillna(0)

                    if feature_matrix.shape[1] >= 2:
                        X_scaled = StandardScaler().fit_transform(feature_matrix.astype(float).values)
                        kmeans = KMeans(n_clusters=min(k_clusters, len(ml_cluster_df), 10), random_state=42, n_init=10)
                        ml_cluster_df['Cluster_ID'] = kmeans.fit_predict(X_scaled).astype(str)

                        plot_df = ml_cluster_df.sample(n=min(1000, len(ml_cluster_df)), random_state=42) if len(ml_cluster_df) > 100 else ml_cluster_df.copy()
                        cluster_counts = ml_cluster_df['Cluster_ID'].value_counts().to_dict()
                        kpi = "<b>Node Distribution:</b> " + ", ".join([f'Cluster {k}: {v} records' for k, v in sorted(cluster_counts.items())])

                        hover_cols = [c for c in [asset_col, region_col, status_col] if c in plot_df.columns]
                        scatter_kwargs = dict(
                            x=x_col,
                            y=y_col,
                            color='Cluster_ID',
                            hover_data=hover_cols,
                            title=f"K-Means Clustering Analysis (K={k_clusters})",
                            color_discrete_sequence=px.colors.qualitative.Safe
                        )
                        if size_col and size_col in plot_df.columns:
                            scatter_kwargs['size'] = size_col

                        fig = px.scatter(plot_df, **scatter_kwargs)
                        fig.update_layout(plot_bgcolor="white", xaxis=dict(showgrid=True, gridcolor="#e2e8f0"), yaxis=dict(showgrid=True, gridcolor="#e2e8f0"), height=500)
                        chart_html = kpi + "<br><br>" + fig.to_html(full_html=False, include_plotlyjs='cdn')
                    else:
                        chart_html = '<div class="alert alert-warning">Insufficient encoded features for clustering. Need at least 2 usable fields.</div>'
                else:
                    chart_html = '<div class="alert alert-warning">Insufficient data for clustering. Need at least 10 valid rows.</div>'
            else:
                chart_html = '<div class="alert alert-danger">Insufficient numeric columns for clustering. Need at least 2 numeric fields.</div>'
        except Exception as e:
            chart_html = f'<div class="alert alert-danger">Clustering Error: {e}</div>'

    elif selected_objective == 'classification':
        # Ensure we can evaluate row count for Polars lazy frames
        if 'pl' in globals() and isinstance(df, pl.LazyFrame):
            df = df.collect()

        row_count = df.shape[0] if hasattr(df, 'shape') else len(df)
        if row_count > 15000:
            if isinstance(df, pd.DataFrame):
                ml_df = df.sample(n=15000, random_state=42)
            elif pl is not None and isinstance(df, pl.DataFrame):
                ml_df = df.sample(n=15000, seed=42)
            else:
                ml_df = df.sample(n=15000)
        else:
            ml_df = df.clone() if hasattr(df, 'clone') else df.copy()
        if isinstance(ml_df, pl.DataFrame):
            ml_df = ml_df.to_pandas()
        target_col = 'Shipment_Status' if 'Shipment_Status' in ml_df.columns else ('Delivery Status_x' if 'Delivery Status_x' in ml_df.columns else list(ml_df.columns)[0])
        
        cols_to_drop = ['Timestamp', 'timestamp', 'order date (DateOrders)_x', 'shipping date (DateOrders)', 'Days for shipping (real)_x', 'Waiting_Time_Hours', 'Days for shipping (real)', 'late_delivery_risk', 'Late_delivery_risk']
        existing_drops = [col for col in cols_to_drop if col in ml_df.columns and col != target_col]
        if existing_drops:
            ml_df = ml_df.drop(columns=existing_drops) if isinstance(ml_df, pd.DataFrame) else ml_df.drop(existing_drops)

        if 'pl' in globals() and isinstance(ml_df, pl.DataFrame):
            ml_df = ml_df.to_pandas()

        le = LabelEncoder()
        ml_df[target_col] = le.fit_transform(ml_df[target_col].astype(str))
        
        for col in ml_df.columns:
            if not pd.api.types.is_numeric_dtype(ml_df[col]):
                ml_df[col] = LabelEncoder().fit_transform(ml_df[col].astype(str))
                
        ml_df = ml_df.fillna(0)
        X = ml_df.drop(columns=[target_col])
        y = ml_df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)
        clf = RandomForestClassifier(n_estimators=80, max_depth=8, min_samples_split=4, class_weight='balanced', random_state=12)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        accuracy_val = accuracy_score(y_test, y_pred)
        precision_val = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall_val = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_val = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        all_classes = np.unique(y)
        cm = confusion_matrix(y_test, y_pred, labels=all_classes)
        labels = le.inverse_transform(all_classes)
        
        data_source = 'pipeline_run' if pipeline_status == "COMPLETED" else ('csv_upload' if file_status == "SUCCESS" else 'manual_test')

        try:
            ClassificationMetrics.objects.create(
                accuracy=float(accuracy_val), precision=float(precision_val),
                recall=float(recall_val), f1_score=float(f1_val),
                total_samples=len(y_test), data_source=data_source,
                confusion_matrix_json=cm.tolist(), class_labels=list(labels),
                notes="Dashboard inline execution complete."
            )
        except Exception:
            pass
        
        classification_metrics = {
            'accuracy': f"{accuracy_val*100:.2f}%",
            'precision': f"{precision_val*100:.2f}%",
            'recall': f"{recall_val*100:.2f}%",
            'f1_score': f"{f1_val*100:.2f}%",
            'total_samples': len(y_test),
            'data_source': data_source.replace('_', ' ').title()
        }
        
        feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': clf.feature_importances_}).sort_values(by='Importance', ascending=True).tail(15)
        fig = px.bar(feature_importances, x='Importance', y='Feature', orientation='h', title="Random Forest Information Gain Hierarchy",
                     color='Importance', color_continuous_scale=px.colors.sequential.Viridis)
        fig.update_layout(plot_bgcolor="white", height=400)
        chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        cm_fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', x=labels, y=labels, title="Confusion Matrix Heatmap")
        cm_fig.update_layout(plot_bgcolor="white", height=400, xaxis_title="Predicted", yaxis_title="Actual")
        confusion_matrix_html = cm_fig.to_html(full_html=False, include_plotlyjs='cdn')

    # INGEST REAL-TIME DATA LOGS
    elif selected_objective == 'add_realtime_data':
        chart_html = ""
        if request.method == 'POST' and 'submit_realtime_form' in request.POST:
            realtime_form = RealTimeAssetDataForm(request.POST)
            if realtime_form.is_valid():
                realtime_form.save()
                messages.success(request, f"✓ Real-time metadata tracking for {realtime_form.cleaned_data['asset_id']} stored successfully!")
                threading.Thread(target=compute_and_store_metrics, args=(get_global_df(), 'realtime_insert'), daemon=True).start()
                realtime_form = RealTimeAssetDataForm()
            else:
                for field, errors in realtime_form.errors.items():
                    for error in errors: messages.error(request, f"{field}: {error}")
        else:
            realtime_form = RealTimeAssetDataForm()

    elif selected_objective == 'objectives':
        chart_html = (
            '<div class="soft-panel p-4 mb-3">'
            '<h5 class="card-title fw-bold text-info mb-2"><i class="fa-solid fa-clipboard-check me-2"></i>Project Objectives & Results</h5>'
            '<p class="dashboard-subtitle mb-0">Consolidated objectives and analytical answers grouped by pipeline phase. Click a card to expand details.</p>'
            '</div>'
        )

    # =========================================================
    # REAL-TIME FLEET MONITOR WITH FORCE COLUMN INJECTIONS
    # =========================================================
    elif selected_objective == 'realtime_data':
        display_records = int(request.POST.get('display_records', request.GET.get('display_records', 50)))
        metrics_filter = request.POST.get('metrics_filter', request.GET.get('metrics_filter', 'latency_performance'))
        metrics_filter = metrics_filter_aliases.get(metrics_filter, metrics_filter)

        warehouse_filter_context = _build_warehouse_filter_context_polars(
            df_lazy,
            asset_filter=request.POST.get('asset_id', request.GET.get('asset_id', '')),
            status_filter=request.POST.get('status', request.GET.get('status', '')),
            region_filter=request.POST.get('region', request.GET.get('region', '')),
            page_number=request.POST.get('page', request.GET.get('page', 1)),
            page_size=50,
        )

        filtered_display_df = warehouse_filter_context['filtered_frame'].copy()
        
        # Sliced dataframe target copy array
        display_records_count = min(display_records, len(filtered_display_df)) if len(filtered_display_df) > 0 else 0
        display_df = filtered_display_df.tail(display_records_count).copy() if display_records_count else filtered_display_df.head(0).copy()

        # Overwrite missing key labels dynamically based on CSV variations
        if 'Asset_ID' not in display_df.columns:
            display_df['Asset_ID'] = 'Truck_' + display_df['Order Id_x'].astype(str) if 'Order Id_x' in display_df.columns else 'Fleet_Unit'
            
        if 'Waiting_Time_Hours' not in display_df.columns:
            display_df['Waiting_Time_Hours'] = display_df['Days for shipping (real)_x'] * 24 if 'Days for shipping (real)_x' in display_df.columns else 24.0
            
        if 'Shipment_Status' not in display_df.columns:
            display_df['Shipment_Status'] = display_df['Delivery Status_x'] if 'Delivery Status_x' in display_df.columns else 'In Transit'
            
        if 'Region' not in display_df.columns:
            display_df['Region'] = display_df['Order Region_x'] if 'Order Region_x' in display_df.columns else 'Global Node'

        if 'Temperature' not in display_df.columns:
            display_df['Temperature'] = display_df['Temperature'] if 'Temperature' in display_df.columns else (display_df['iot_temperature'] if 'iot_temperature' in display_df.columns else 25.0)

        asset_col = 'Asset_ID'
        wait_col = 'Waiting_Time_Hours'
        status_col = 'Shipment_Status'
        region_col = 'Region'
        temp_col = 'Temperature'
        
        def _clean_asset_label(raw_value):
            asset_label = str(raw_value).strip()
            if not asset_label:
                return 'Unknown'

            match = re.search(r'(?i)\b(trk[-_]\d+|truck[-_]\d+)\b', asset_label)
            if match:
                label = match.group(1).upper().replace('TRUCK_', 'TRK-').replace('TRUCK-', 'TRK-').replace('TRK_', 'TRK-')
                return label

            match = re.search(r'(\d{3,})', asset_label)
            if match:
                return f'TRK-{match.group(1)}'

            if len(asset_label) <= 12:
                return asset_label
            return asset_label[:12].rstrip(' -_')

        chart_source_df = display_df.copy() if len(display_df) > 0 else filtered_display_df.copy()

        def _infer_region_center(region_value):
            text = str(region_value or '').lower()
            if any(keyword in text for keyword in ['southeast asia', 'asia', 'singapore', 'bangkok', 'manila', 'jakarta']):
                return 1.3521, 103.8198
            if any(keyword in text for keyword in ['oceania', 'australia', 'sydney', 'melbourne', 'au']):
                return -33.8688, 151.2093
            if any(keyword in text for keyword in ['north america', 'usa', 'united states', 'canada', 'us ']):
                return 37.7749, -122.4194
            if any(keyword in text for keyword in ['europe', 'france', 'germany', 'uk', 'united kingdom', 'paris', 'london']):
                return 48.8566, 2.3522
            if any(keyword in text for keyword in ['south america', 'brazil', 'argentina', 'são paulo', 'buenos aires']):
                return -23.5505, -46.6333
            if any(keyword in text for keyword in ['africa', 'nigeria', 'kenya', 'egypt', 'south africa']):
                return -1.2921, 36.8219
            if any(keyword in text for keyword in ['middle east', 'dubai', 'uae', 'saudi', 'qatar']):
                return 25.2048, 55.2708
            return 20.0, 0.0

        def _marker_color(status_value):
            status_text = str(status_value or '').lower()
            if 'late delivery' in status_text or 'shipping canceled' in status_text or 'canceled' in status_text or 'delayed' in status_text:
                return 'red'
            if 'shipping on time' in status_text or 'advance shipping' in status_text or 'advance' in status_text or 'delivered' in status_text or 'on time' in status_text:
                return 'green'
            return 'blue'

        marker_rng = np.random.default_rng(42)
        map_marker_rows = []
        for _, row in display_df.iterrows():
            region_value = row.get('operational_region', row.get('Region', row.get('Order Region_x', 'Global Node')))
            status_value = row.get('logistics_status', row.get('Shipment_Status', 'Unknown'))
            asset_value = row.get('asset_id', row.get('Asset_ID', 'Unknown'))
            base_lat, base_lng = _infer_region_center(region_value)
            jitter_lat = float(base_lat + marker_rng.uniform(-0.25, 0.25))
            jitter_lng = float(base_lng + marker_rng.uniform(-0.25, 0.25))
            map_marker_rows.append({
                'asset_id': str(asset_value),
                'region': str(region_value),
                'status': str(status_value),
                'lat': jitter_lat,
                'lng': jitter_lng,
                'color': _marker_color(status_value),
            })

        if len(map_marker_rows) == 0:
            map_marker_rows.append({
                'asset_id': 'Demo Fleet',
                'region': 'Global Node',
                'status': 'No active fleet data',
                'lat': 20.0,
                'lng': 0.0,
                'color': 'blue',
            })
        map_markers_json = json.dumps(map_marker_rows)

        if metrics_filter == 'latency_performance':
            chart_source_df['asset_label'] = chart_source_df[asset_col].apply(_clean_asset_label)
            chart_source_df = chart_source_df.groupby('asset_label', as_index=False)[wait_col].mean().sort_values('asset_label')

            fig_realtime = go.Figure()
            fig_realtime.add_trace(go.Bar(
                x=chart_source_df['asset_label'],
                y=chart_source_df[wait_col],
                marker=dict(color=chart_source_df[wait_col], colorscale='RdYlGn_r'),
                name='Warehouse Data'
            ))
            try:
                rt_qs = RealTimeAssetData.objects.all().order_by('-timestamp')
                rt_df = pd.DataFrame(list(rt_qs.values('asset_id', 'waiting_latency_hours')))
                if not rt_df.empty:
                    rt_df['asset_label'] = rt_df['asset_id'].apply(_clean_asset_label)
                    fig_realtime.add_trace(go.Scatter(
                        x=rt_df['asset_label'], 
                        y=rt_df['waiting_latency_hours'], 
                        mode='markers', 
                        marker=dict(size=12, color='rgba(255,65,54,0.9)', line=dict(width=1, color='DarkRed')), 
                        name='Live Ingest Stream'
                    ))
            except Exception:
                pass
            fig_realtime.update_layout(
                title='Fleet Latency Performance Monitor',
                xaxis=dict(
                    tickmode='auto',
                    nticks=min(15, max(1, len(chart_source_df))),
                    tickangle=-45,
                    automargin=True,
                    tickfont=dict(size=11),
                    categoryorder='category ascending',
                ),
                bargap=0.2,
                bargroupgap=0.1,
            )
            
        elif metrics_filter == 'fleet_status':
            status_counts = chart_source_df[status_col].value_counts().reset_index()
            status_counts.columns = ['Status', 'Count']
            fig_realtime = px.pie(status_counts, values='Count', names='Status', hole=0.3, title="Fleet Status Distribution")
            
        elif metrics_filter == 'temperature_analysis':
            fig_realtime = go.Figure()
            fig_realtime.add_trace(go.Scatter(
                x=chart_source_df[asset_col].astype(str), 
                y=chart_source_df[temp_col], 
                mode='markers', 
                marker=dict(size=chart_source_df[wait_col].fillna(6), color=chart_source_df[temp_col], colorscale='RdYlBu_r', showscale=True), 
                name='Warehouse Data'
            ))
            try:
                rt_qs = RealTimeAssetData.objects.all().order_by('-timestamp')
                rt_df = pd.DataFrame(list(rt_qs.values('asset_id', 'iot_temperature', 'waiting_latency_hours')))
                if not rt_df.empty:
                    fig_realtime.add_trace(go.Scatter(
                        x=rt_df['asset_id'].astype(str), 
                        y=rt_df['iot_temperature'], 
                        mode='markers', 
                        marker=dict(size=rt_df['waiting_latency_hours'].fillna(6), color='rgba(255,65,54,0.9)', symbol='x'), 
                        name='Live Ingest Stream'
                    ))
            except Exception:
                pass
            fig_realtime.update_layout(title='Asset Temperature & Latency Analysis')
        else:
            region_data = chart_source_df.groupby(region_col, as_index=False)[wait_col].mean()
            fig_realtime = px.bar(region_data, x=region_col, y=wait_col, color=wait_col, title="Average Latency by Region", color_continuous_scale='Viridis')
        
        fig_realtime.update_layout(plot_bgcolor="white", paper_bgcolor="white", font_color="#0f172a", height=420)
        kpi_html = f'<div class="alert alert-info bg-info bg-opacity-10 border-0 text-info mb-3"><i class="fa-solid fa-chart-bar me-2"></i><strong>Monitoring:</strong> {display_records} records | Filter: {metrics_filter.title()}</div>'
        chart_html = kpi_html + fig_realtime.to_html(full_html=False, include_plotlyjs='cdn')
        realtime_page_obj = warehouse_filter_context['page_obj']
        total_records = warehouse_filter_context['total_records']
        unique_assets = warehouse_filter_context['unique_assets']
        unique_status = warehouse_filter_context['unique_status']
        unique_regions = warehouse_filter_context['unique_regions']
        asset_filter = warehouse_filter_context['asset_filter']
        status_filter = warehouse_filter_context['status_filter']
        region_filter = warehouse_filter_context['region_filter']

    # OPERATIONS ENVIRONMENT
    elif selected_objective == 'operations_env':
        df_for_ops = df.to_pandas() if isinstance(df, pl.DataFrame) else df

        status_col = 'Delivery Status_x' if 'Delivery Status_x' in df_for_ops.columns else ('Shipment_Status' if 'Shipment_Status' in df_for_ops.columns else df_for_ops.columns[0])
        status_counts = df_for_ops[status_col].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        fig1 = px.pie(status_counts, values='Count', names='Status', hole=0.4, title='Operational Distribution Breakdown')
        fig1.update_layout(plot_bgcolor="white", height=350)
        
        region_col = 'Order Region_x' if 'Order Region_x' in df_for_ops.columns else ('Region' if 'Region' in df_for_ops.columns else df_for_ops.columns[0])
        region_counts = df_for_ops[region_col].value_counts().reset_index()
        region_counts.columns = ['Region', 'Count']
        fig2 = px.bar(region_counts, x='Region', y='Count', title='Volume Traffic Metrics per Node', color='Count', color_continuous_scale='Teal')
        fig2.update_layout(plot_bgcolor="white", height=350)
        
        chart_html = f'<div class="row"><div class="col-md-6 mb-3">{fig1.to_html(full_html=False, include_plotlyjs="cdn")}</div><div class="col-md-6 mb-3">{fig2.to_html(full_html=False, include_plotlyjs="cdn")}</div></div>'

    # =========================================================
    # REPAIRED TEMPLATE PRESENTATION NORMALIZATION LAYER
    # =========================================================
    # Prefer rows that contain joined NoSQL enrichment so table values are not dominated by null-leading rows.
    if pl is not None and isinstance(df, pl.LazyFrame):
        display_source_df = df.collect().to_pandas()
    elif pl is not None and isinstance(df, pl.DataFrame):
        display_source_df = df.to_pandas()
    else:
        display_source_df = df.copy()

    enrichment_cols = ['Traffic_Status', 'Logistics_Delay_Reason', 'Doc_Category']
    available_enrichment_cols = [col for col in enrichment_cols if col in display_source_df.columns]

    if available_enrichment_cols:
        enriched_mask = display_source_df[available_enrichment_cols].notna().any(axis=1)
        display_source_df = display_source_df.assign(_has_enrichment=enriched_mask)
        display_source_df = display_source_df.sort_values(by='_has_enrichment', ascending=False)

    display_df = display_source_df.head(display_records).copy()
    if '_has_enrichment' in display_df.columns:
        display_df = display_df.drop(columns=['_has_enrichment'])
    
    # 1. Map Timestamp Columns Dynamically
    if 'order date (DateOrders)_x' in display_df.columns:
        display_df['Timestamp'] = display_df['order date (DateOrders)_x']
    elif 'Timestamp' in display_df.columns:
        display_df['Timestamp'] = display_df['Timestamp']
    else:
        display_df['Timestamp'] = display_df.index
        
    # 2. Map Asset Columns Dynamically
    if 'Asset_ID' in display_df.columns:
        display_df['Asset_ID'] = display_df['Asset_ID']
    elif 'Order Id_x' in display_df.columns:
        display_df['Asset_ID'] = 'Truck_' + display_df['Order Id_x'].astype(str)
    else:
        display_df['Asset_ID'] = 'Fleet_Unit'

    # 3. Map Region Columns Dynamically
    if 'Order Region_x' in display_df.columns:
        display_df['Region'] = display_df['Order Region_x']
    elif 'Region' in display_df.columns:
        display_df['Region'] = display_df['Region']
    else:
        display_df['Region'] = 'Southeast Asia'

    # 4. Map Shipment/Logistics Status Columns Dynamically
    if 'Delivery Status_x' in display_df.columns:
        display_df['Shipment_Status'] = display_df['Delivery Status_x']
    elif 'Shipment_Status' in display_df.columns:
        display_df['Shipment_Status'] = display_df['Shipment_Status']
    else:
        display_df['Shipment_Status'] = 'In Transit'

    # 5. Map Latency/Waiting Time Columns Dynamically (Convert Days to Hours safely)
    if 'Days for shipping (real)_x' in display_df.columns:
        display_df['Waiting_Time_Hours'] = pd.to_numeric(display_df['Days for shipping (real)_x'], errors='coerce').fillna(3) * 24
    elif 'Waiting_Time_Hours' in display_df.columns:
        display_df['Waiting_Time_Hours'] = display_df['Waiting_Time_Hours']
    else:
        display_df['Waiting_Time_Hours'] = 72.0

    # 6. Map joined NoSQL analytics fields explicitly for dashboard rendering.
    def pick_first_existing(columns, default_value='N/A'):
        for col in columns:
            if col in display_df.columns:
                return display_df[col]
        return pd.Series([default_value] * len(display_df), index=display_df.index)

    # IoT Temperature: prefer NoSQL/joined variants first.
    display_df['Temperature'] = pick_first_existing([
        'Temperature_y',
        'Temperature',
        'iot_temperature',
    ], default_value='N/A')

    # Traffic Volume Scale: explicitly bind to Traffic_Status from joined NoSQL data.
    if 'Traffic_Status' in display_df.columns:
        display_df['Traffic_Volume_Index'] = display_df['Traffic_Status']
    elif 'Traffic_Status_y' in display_df.columns:
        display_df['Traffic_Volume_Index'] = display_df['Traffic_Status_y']
    else:
        display_df['Traffic_Volume_Index'] = pick_first_existing([
            'Traffic_Volume_Index_y',
            'Traffic_Volume_Index',
            'traffic_volume_scale',
        ], default_value='N/A')

    # Driver Sentiment: use Logistics_Delay_Reason / Doc_Category as primary source text.
    raw_driver_sentiment = pick_first_existing([
        'Logistics_Delay_Reason',
        'Logistics_Delay_Reason_y',
        'Doc_Category',
        'Doc_Category_y',
        'Driver_Sentiment',
        'driver_sentiment',
        'stress_level',
        'NLP_Sentiment',
        'Document_Genre',
    ], default_value='N/A')

    def standardize_driver_sentiment(series):
        def _label(value):
            text = str(value).strip().lower()
            if text in {'', 'n/a', 'nan', 'none', 'null'}:
                return 'N/A'

            # Normalize pre-labeled values first.
            if text in {'high', 'critical', 'severe', 'negative'}:
                return 'HIGH'
            if text in {'medium', 'moderate', 'neutral'}:
                return 'MEDIUM'
            if text in {'low', 'positive', 'safe', 'normal'}:
                return 'LOW'

            high_keywords = [
                'accident', 'crash', 'collision', 'panic', 'furious',
                'exhausted', 'fatigue', 'high_temp', 'extreme', 'angry',
            ]
            medium_keywords = [
                'traffic', 'delay', 'stuck', 'congestion', 'rain', 'storm',
                'worried', 'anxious', 'frustrated',
            ]
            low_keywords = [
                'smooth', 'safe', 'good', 'great', 'on time', 'delivered',
                'efficient', 'happy',
            ]

            if any(keyword in text for keyword in high_keywords):
                return 'HIGH'
            if any(keyword in text for keyword in medium_keywords):
                return 'MEDIUM'
            if any(keyword in text for keyword in low_keywords):
                return 'LOW'

            return 'MEDIUM'

        return series.apply(_label)

    display_df['Document_Genre'] = standardize_driver_sentiment(raw_driver_sentiment)

    # Re-fill missing values with uniform placeholders safely after structural transformations
    # Safely handle categorical columns by adding the placeholder category when possible,
    # otherwise convert the column to object (string) and proceed to fillna.
    categorical_cols = [c for c in display_df.columns if isinstance(display_df[c].dtype, pd.CategoricalDtype)]
    for cat_col in categorical_cols:
        try:
            display_df[cat_col] = display_df[cat_col].cat.add_categories(['N/A'])
        except Exception:
            display_df[cat_col] = display_df[cat_col].astype(object)
    display_df = display_df.fillna('N/A')

    context = {
        'selected_objective': selected_objective,
        'objectives_phases': objectives_phases,
        'verified_milestones': verified_milestones,
        'total_ingested_rows': total_ingested_rows,
        'max_mining_confidence': max_mining_confidence,
        'balanced_ml_samples': balanced_ml_samples,
        'k_clusters': k_clusters,
        'min_support': min_support,
        'min_confidence': min_confidence,
        'display_records': display_records,
        'metrics_filter': metrics_filter,
        'chart_html': chart_html,
        'rules_list': rules_list,
        'accuracy': accuracy_val,
        'precision': precision_val,
        'recall': recall_val,
        'f1_score': f1_val,
        'confusion_matrix_html': confusion_matrix_html,
        'classification_metrics': classification_metrics,
        'pipeline_message': pipeline_message,
        'file_status': file_status,
        'pipeline_status': pipeline_status,
        'warehouse_data': display_df.head(50).to_dict('records'),
        'map_markers_json': map_markers_json if selected_objective == 'realtime_data' else json.dumps([]),
        'realtime_form': realtime_form,
        'realtime_page_obj': realtime_page_obj,
        'page_obj': realtime_page_obj,
        'total_records': total_records,
        'asset_filter': asset_filter,
        'status_filter': status_filter,
        'region_filter': region_filter,
        'unique_assets': unique_assets,
        'unique_status': unique_status,
        'unique_regions': unique_regions,
    }
    return render(request, 'dashboard.html', context)


def objectives_view(request):
    """Render the objectives report in the premium dashboard shell."""
    objectives_phases = build_objectives_phases()
    context = {
        'selected_objective': 'objectives',
        'objectives_phases': objectives_phases,
        'verified_milestones': 30,
        'total_ingested_rows': '1,003,000 rows',
        'max_mining_confidence': '73.98%',
        'balanced_ml_samples': '928,000 rows',
        'chart_html': '',
        'file_status': '',
        'pipeline_status': '',
        'realtime_form': RealTimeAssetDataForm(),
        'k_clusters': 5,
        'display_records': 50,
        'metrics_filter': 'latency_performance',
        'warehouse_data': [],
        'classification_metrics': {'accuracy': 'N/A', 'precision': 'N/A', 'recall': 'N/A', 'f1_score': 'N/A', 'total_samples': 0, 'data_source': 'N/A'},
        'map_markers_json': '[]'
    }
    return render(request, 'dashboard.html', context)


@csrf_exempt
def latest_classification_metrics(request):
    try:
        latest = ClassificationMetrics.objects.order_by('-timestamp').first()
        if not latest: return JsonResponse({'ok': False, 'message': 'No metrics available.'}, status=200)
        return JsonResponse({
            'ok': True, 'timestamp': latest.timestamp.isoformat(), 'accuracy': f"{latest.accuracy*100:.2f}%",
            'precision': f"{latest.precision*100:.2f}%", 'recall': f"{latest.recall*100:.2f}%", 'f1_score': f"{latest.f1_score*100:.2f}%",
            'total_samples': latest.total_samples, 'data_source': latest.data_source, 'confusion_matrix': latest.confusion_matrix_json,
            'class_labels': latest.class_labels, 'notes': latest.notes,
        }, status=200)
    except Exception as e:
        return JsonResponse({'ok': False, 'message': f'Error: {e}'}, status=200)


# --- STAND-ALONE REALTIME DATA INTERFACES ---
@require_http_methods(["GET", "POST"])
def realtime_data_view(request):
    all_data = RealTimeAssetData.objects.all().order_by('-timestamp')
    asset_filter = request.GET.get('asset_id', '')
    status_filter = request.GET.get('status', '')
    region_filter = request.GET.get('region', '')
    
    if asset_filter: all_data = all_data.filter(asset_id__icontains=asset_filter)
    if status_filter: all_data = all_data.filter(logistics_status=status_filter)
    if region_filter: all_data = all_data.filter(operational_region=region_filter)
    
    paginator = Paginator(all_data, 50)
    page_obj = paginator.get_page(request.GET.get('page', 1))
    
    assets = RealTimeAssetData.objects.values_list('asset_id', flat=True).distinct()
    statuses = RealTimeAssetData.objects.values_list('logistics_status', flat=True).distinct()
    regions = RealTimeAssetData.objects.values_list('operational_region', flat=True).distinct()
    
    context = {
        'page_obj': page_obj, 'assets': sorted(set(assets)), 'statuses': sorted(set(statuses)), 'regions': sorted(set(regions)),
        'total_records': all_data.count(), 'asset_filter': asset_filter, 'status_filter': status_filter, 'region_filter': region_filter,
    }
    return render(request, 'realtime_data.html', context)


@require_http_methods(["GET", "POST"])
def add_realtime_data(request):
    if request.method == 'POST':
        form = RealTimeAssetDataForm(request.POST)
        if form.is_valid():
            instance = form.save()

            try:
                try:
                    import polars as pl
                    from datetime import datetime

                    # 1. Locate Parquet file path
                    PARQUET_FILE_PATH = os.path.join(
                        settings.BASE_DIR,
                        'Intelligent Supply Chain & Inventory Optimizer',
                        'integrated_data_warehouse.parquet'
                    )

                    # 2. Read the Parquet file fully to avoid Windows memory-mapped file locks.
                    if os.path.exists(PARQUET_FILE_PATH):
                        existing_df = pl.read_parquet(PARQUET_FILE_PATH, use_pyarrow=True, low_memory=True)
                        columns = existing_df.columns
                    else:
                        messages.warning(request, "Parquet file not found. Entry not saved to warehouse.")
                        existing_df = None
                        columns = []

                    # Map form fields to Parquet columns
                    now = datetime.now()
                    new_row = {}
                    for col in columns:
                        if col in ['Timestamp', 'order date (DateOrders)_x', 'order date (DateOrders)_y']:
                            new_row[col] = now
                        elif col == 'Asset_ID':
                            new_row[col] = instance.asset_id
                        elif col in ['Order Region_x', 'Region', 'operational_region']:
                            new_row[col] = instance.operational_region
                        elif col in ['Delivery Status_x', 'Shipment_Status', 'logistics_status']:
                            new_row[col] = instance.logistics_status
                        elif col in ['Temperature', 'iot_temperature']:
                            new_row[col] = instance.iot_temperature
                        elif col in ['Traffic_Status', 'traffic_volume_scale']:
                            new_row[col] = instance.traffic_volume_scale
                        elif col in ['Waiting_Time_Hours', 'waiting_latency_hours']:
                            new_row[col] = instance.waiting_latency_hours
                        elif col in ['Doc_Category', 'nlp_classified_genre']:
                            new_row[col] = instance.nlp_classified_genre
                        else:
                            new_row[col] = None

                    new_row_df = pl.DataFrame([new_row])

                    # Align new row datatypes dynamically to match Categorical/String expectations
                    if existing_df is not None:
                        aligned_exprs = []
                        for col_name in existing_df.columns:
                            target_dtype = existing_df.schema[col_name]
                            if col_name in new_row_df.columns:
                                try:
                                    # If incoming column is temporal but target expects categorical/string, format it
                                    if getattr(new_row_df.schema[col_name], 'is_temporal', lambda: False)() and (
                                        target_dtype == pl.Categorical or target_dtype == pl.String
                                    ):
                                        expr = pl.col(col_name).dt.strftime('%m/%d/%Y %H:%M').cast(target_dtype)
                                    else:
                                        expr = pl.col(col_name).cast(target_dtype)
                                except Exception:
                                    expr = pl.col(col_name).cast(target_dtype)
                            else:
                                expr = pl.lit(None).cast(target_dtype)
                            aligned_exprs.append(expr)

                        new_row_df = new_row_df.select(aligned_exprs)

                    # 3. Concat and write back to Parquet with a short retry loop
                    if existing_df is None:
                        updated_df = new_row_df
                    else:
                        updated_df = pl.concat([existing_df, new_row_df])

                    write_attempts = 3
                    for attempt in range(write_attempts):
                        try:
                            updated_df.write_parquet(PARQUET_FILE_PATH, compression="snappy")
                            break
                        except Exception as write_error:
                            if attempt < write_attempts - 1:
                                time.sleep(0.2)
                            else:
                                raise write_error

                    # 4. Clear global cache for instant dashboard update
                    with _DF_LOCK:
                        _GLOBAL_DF_CACHE.pop('df', None)

                    messages.success(request, f"✓ Real-time data for {form.cleaned_data['asset_id']} added to Parquet warehouse!")
                except ImportError:
                    messages.warning(request, "Polars is not installed. Entry not saved to Parquet warehouse.")
            except Exception as e:
                messages.warning(request, f"Saved to DB but failed to append Parquet: {e}")

            threading.Thread(target=compute_and_store_metrics, args=(get_global_df(), 'realtime_insert'), daemon=True).start()
            return redirect('realtime_data')
        else:
            for field, errors in form.errors.items():
                for error in errors: messages.error(request, f"{field}: {error}")
    else:
        form = RealTimeAssetDataForm()
    return render(request, 'add_realtime_data.html', {'form': form, 'page_title': 'Add Real-Time Asset Data'})


@csrf_exempt
@require_http_methods(["POST"])
def add_realtime_data_api(request):
    try:
        data = json.loads(request.body)
        realtime_obj = RealTimeAssetData.objects.create(
            asset_id=data.get('asset_id', 'Unknown'), operational_region=data.get('operational_region', 'Southeast Asia'),
            logistics_status=data.get('logistics_status', 'In Transit'), iot_temperature=data.get('iot_temperature'),
            traffic_volume_scale=data.get('traffic_volume_scale'), waiting_latency_hours=data.get('waiting_latency_hours'),
            nlp_classified_genre=data.get('nlp_classified_genre', '')
        )
        threading.Thread(target=compute_and_store_metrics, args=(get_global_df(), 'realtime_insert'), daemon=True).start()
        return JsonResponse({'status': 'success', 'message': f'Data for {realtime_obj.asset_id} added successfully', 'id': realtime_obj.id, 'timestamp': realtime_obj.timestamp.isoformat()})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)


@require_http_methods(["GET"])
def realtime_data_api(request):
    asset_id = request.GET.get('asset_id')
    status = request.GET.get('status')
    limit = int(request.GET.get('limit', 100))
    queryset = RealTimeAssetData.objects.all().order_by('-timestamp')
    if asset_id: queryset = queryset.filter(asset_id__iexact=asset_id)
    if status: queryset = queryset.filter(logistics_status=status)
    
    data = [{
        'id': obj.id, 'timestamp': obj.timestamp.isoformat(), 'asset_id': obj.asset_id, 'operational_region': obj.operational_region,
        'logistics_status': obj.logistics_status, 'iot_temperature': obj.iot_temperature, 'traffic_volume_scale': obj.traffic_volume_scale,
        'waiting_latency_hours': obj.waiting_latency_hours, 'nlp_classified_genre': obj.nlp_classified_genre
    } for obj in queryset[:limit]]
    return JsonResponse({'status': 'success', 'count': len(data), 'data': data})



