# FEATURE 1: AI-POWERED DELAY PREDICTOR
# Machine Learning Model to Predict Order Delays

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

class DelayPredictor:
    """AI Model to Predict if an Order will be Delayed"""
    
    def __init__(self, model_path='delay_predictor_model.pkl'):
        self.model = None
        self.encoders = {}
        self.model_path = model_path
        self.accuracy = 0
    
    def prepare_data(self, df):
        """Prepare training dataframe and feature list.

        - Identifies or derives a target column (binary: delayed=1 / on-time=0)
        - Drops obvious non-feature columns (timestamps, IDs)
        - Encodes non-numeric features using LabelEncoder and stores encoders in `self.encoders`
        - Fills numeric NaNs with medians

        Returns: (df_prepared, feature_list)
        """
        df_proc = df.copy()

        # --- Determine target ---
        target_candidates = [
            'Late_delivery_risk', 'late_delivery_risk', 'Late_delivery_risk_x', 'late_delivery_risk_x',
            'delay', 'delayed', 'delivery_status', 'Delivery Status_x', 'Shipment_Status', 'target'
        ]
        target_col = None
        for c in target_candidates:
            if c in df_proc.columns:
                target_col = c
                break

        if target_col is None:
            # fallback: derive from waiting/shipping duration if available
            wait_cols = ['Waiting_Time_Hours', 'Days for shipping (real)_x', 'Days for shipping (real)']
            wait_col = next((c for c in wait_cols if c in df_proc.columns), None)
            if wait_col is None:
                raise ValueError('No target column found and no waiting/time column to derive target from')
            # convert days->hours when necessary
            series = pd.to_numeric(df_proc[wait_col], errors='coerce')
            if 'Days for shipping' in wait_col:
                series = series * 24
            median_hours = series.median()
            df_proc['target'] = (series > median_hours).fillna(0).astype(int)
        else:
            s = df_proc[target_col]
            if pd.api.types.is_numeric_dtype(s):
                df_proc['target'] = pd.to_numeric(s, errors='coerce').fillna(0).astype(int)
            else:
                s_str = s.astype(str).str.lower()
                df_proc['target'] = s_str.apply(lambda x: 1 if any(k in x for k in ['delay', 'delayed', 'late', 'cancel']) else 0)

        # --- Drop columns that should not be used as features ---
        drop_candidates = ['Timestamp', 'timestamp', 'order date (DateOrders)_x', 'shipping date (DateOrders)',
                           'Order Id', 'order_id', 'OrderId', 'Days for shipping (real)_x', 'Days for shipping (real)',
                           'Waiting_Time_Hours', 'late_delivery_risk', 'Late_delivery_risk', target_col]
        drop_cols = [c for c in drop_candidates if c in df_proc.columns and c != 'target']

        feature_cols = [c for c in df_proc.columns if c not in drop_cols and c != 'target']

        # --- Clean and encode features ---
        final_features = []
        for col in feature_cols:
            # skip if identical to target
            if col == 'target':
                continue
            if pd.api.types.is_numeric_dtype(df_proc[col]):
                df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce')
                median_val = df_proc[col].median()
                df_proc[col] = df_proc[col].fillna(median_val if not pd.isna(median_val) else 0)
                final_features.append(col)
            else:
                # categorical/text -> label encode
                df_proc[col] = df_proc[col].fillna('missing').astype(str)
                le = LabelEncoder()
                try:
                    df_proc[col] = le.fit_transform(df_proc[col])
                    self.encoders[col] = le
                    final_features.append(col)
                except Exception:
                    # if encoding fails, drop the column
                    continue

        if len(final_features) == 0:
            raise ValueError('No usable features were identified for training')

        # keep only target + features (order doesn't matter)
        df_result = df_proc[final_features + ['target']].copy()
        return df_result, final_features
    
    def train(self, df):
        """Train the Random Forest model with Robust Class Balancing"""
        print("\n=== DELAY PREDICTOR - TRAINING ===\n")
        
        df_prepared, features = self.prepare_data(df)
        self.features = features
        
        # 1. Separating Late and On-Time orders
        late_orders = df_prepared[df_prepared['target'] == 1]
        ontime_orders = df_prepared[df_prepared['target'] == 0]
        
        print(f"Detected: {len(late_orders)} Late, {len(ontime_orders)} On-Time")

        # 2. SMART BALANCING LOGIC (The Fix)
        # Hum dono groups mein se jo chota hai, uske size ke barabar sampling karenge
        min_size = min(len(late_orders), len(ontime_orders))

        if min_size > 0:
            # Dono classes se barabar samples uthayein (Undersampling)
            late_sampled = late_orders.sample(n=min_size, random_state=42)
            ontime_sampled = ontime_orders.sample(n=min_size, random_state=42)
            df_balanced = pd.concat([late_sampled, ontime_sampled])
            print(f"✅ Training on Balanced Data: {len(df_balanced)} total samples (50/50 split)")
        else:
            # Fallback agar aik group bilkul hi khali ho
            df_balanced = df_prepared
            print("⚠️ Using raw data (no balancing possible)")
        
        X = df_balanced[features]
        y = df_balanced['target']
        
        # 3. Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"Training Random Forest on {len(X_train)} samples...")
        self.model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
        self.model.fit(X_train, y_train)
        
        # 4. Accuracy Check
        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        print(f"✅ Model Training Complete. Accuracy: {self.accuracy:.2%}")
        
        # Save model
        with open(self.model_path, 'wb') as f:
            pickle.dump({'model': self.model, 'features': self.features, 'accuracy': self.accuracy}, f)
        
        return True
    def predict_batch(self, df):
        """Predict delays for multiple orders"""
        if self.model is None:
            print("ERROR: Model not trained")
            return df
        
        df_copy = df.copy()
        
        # Prepare features (using the same logic as training)
        X = df_copy[self.features].copy()
        for col in self.features:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        # Predict Probabilities
        probabilities = self.model.predict_proba(X)[:, 1]
        
        # Adding result columns
        df_copy['delay_probability'] = probabilities
        df_copy['delay_prediction'] = (probabilities > 0.5).astype(int)
        df_copy['delay_percentage'] = (probabilities * 100).round(1)
        
        return df_copy
    
def load_model(self):
        """Load pre-trained model"""
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.features = data['features']
                self.accuracy = data['accuracy']
            print(f"Model loaded (Accuracy: {self.accuracy:.2%})")
            return True
        except:
            print("No pre-trained model found")
            return False
    
def predict(self, order_data):
        """Predict delay for a single order
        
        Args:
            order_data: dict with features
            
        Returns:
            delay_probability (0.0-1.0): Probability of delay
        """
        if self.model is None:
            print("ERROR: Model not trained. Call train() first.")
            return None
        
        # Prepare features
        feature_vector = []
        for feat in self.features:
            val = order_data.get(feat, 0)
            feature_vector.append(float(val))
        
        # Predict
        probability = self.model.predict_proba([feature_vector])[0][1]
        
        return probability
    
def predict_batch(self, df):
        """Predict delays for multiple orders"""
        if self.model is None:
            print("ERROR: Model not trained")
            return None
        
        df_copy = df.copy()
        
        # Prepare features
        X = df_copy[self.features].copy()
        for col in self.features:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        # Predict
        probabilities = self.model.predict_proba(X)[:, 1]
        df_copy['delay_probability'] = probabilities
        df_copy['delay_prediction'] = (probabilities > 0.5).astype(int)
        df_copy['delay_percentage'] = (probabilities * 100).round(1)
        
        return df_copy[[col for col in df_copy.columns if col in 
                       ['Order Id', 'delay_probability', 'delay_prediction', 'delay_percentage']] + 
                      [col for col in df_copy.columns if col not in 
                       ['Order Id', 'delay_probability', 'delay_prediction', 'delay_percentage']]]


# Example Usage
if __name__ == "__main__":
    # Load data
    try:
        df = pd.read_csv('integrated_data_warehouse.csv')
        print(f"Data loaded: {df.shape[0]} rows")
    except:
        try:
            df = pd.read_csv('dataco_postgresql_structured.csv')
            print(f"Using PostgreSQL CSV: {df.shape[0]} rows")
        except:
            print("ERROR: No data found")
            exit()
    
    # Train model
    predictor = DelayPredictor()
    predictor.train(df)
    
    # Make predictions
    if df.shape[0] > 0:
        print("\n=== SAMPLE PREDICTIONS ===")
        sample_rows = min(5, df.shape[0])
        predictions = predictor.predict_batch(df.head(sample_rows))
        
        for idx, row in predictions.head(5).iterrows():
            order_id = row.get('Order Id', f'Order_{idx}')
            delay_pct = row.get('delay_percentage', 0)
            print(f"\n{order_id}:")
            print(f"  Delay Probability: {delay_pct:.1f}%")
            if delay_pct > 70:
                print(f"  Status: HIGH RISK - Urgent intervention needed")
            elif delay_pct > 50:
                print(f"  Status: MEDIUM RISK - Monitor closely")
            else:
                print(f"  Status: LOW RISK - On track")
