import os
from pathlib import Path

import pandas as pd
import re
from typing import Tuple

class DriverSentimentAnalyzer:
    """Analyze driver emotions and stress levels using Hybrid Logic (Text + Data)"""
    
    # Emotion keywords
    STRESS_KEYWORDS = {
        'fatigue': ['tired', 'exhausted', 'fatigue', 'sleepy', 'worn out', 'drained'],
        'frustration': ['frustrated', 'angry', 'annoyed', 'irritated', 'fed up', 'furious'],
        'fear': ['afraid', 'scared', 'nervous', 'worried', 'anxious', 'panic'],
        'accident_risk': ['accident', 'crash', 'collision', 'hit', 'damage', 'broken'],
        'traffic': ['traffic', 'jam', 'congestion', 'blocked', 'stuck', 'delay'],
        'weather': ['rain', 'storm', 'snow', 'fog', 'weather', 'slippery', 'wet'],
    }
    
    POSITIVE_KEYWORDS = {
        'good': ['good', 'great', 'excellent', 'perfect', 'wonderful', 'amazing'],
        'safe': ['safe', 'smooth', 'easy', 'fast', 'quick', 'efficient'],
        'happy': ['happy', 'pleased', 'satisfied', 'content', 'glad', 'cheerful'],
    }
    
    def __init__(self):
        self.analyzed_logs = []

    def calculate_hybrid_stress(self, row, text_score):
        """NEW: Data-Driven Stress Logic (Proactive Analysis)"""
        extra_score = 0
        risk_factors = []
        
        # 1. Agar order LATE hai (High Pressure)
        if row.get('Late_delivery_risk_x') == 1:
            extra_score += 40
            risk_factors.append("DELIVERY DELAY PRESSURE")
            
        # 2. Agar temperature HIGH hai (Tough conditions)
        if row.get('Temp_Status') == 'HIGH_TEMP':
            extra_score += 30
            risk_factors.append("EXTREME WEATHER/TEMP")
            
        # 3. Agar shipping mode 'First Class' hai (Time pressure)
        if 'First Class' in str(row.get('Shipping Mode_x', '')):
            extra_score += 20
            risk_factors.append("URGENT PRIORITY STRESS")
            
        final_score = min(100, text_score + extra_score)
        return final_score, risk_factors

    def analyze_text_with_data(self, row, log_column='Logistics_Delay_Reason') -> dict:
        """Analyze sentiment combining text keywords and operational data"""
        text = str(row.get(log_column, ""))
        text_lower = text.lower()
        emotions_detected = []
        risk_factors = []
        text_stress_score = 0
        
        # Keyword-based analysis
        if text and text != "nan":
            for emotion_type, keywords in self.STRESS_KEYWORDS.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        emotions_detected.append(emotion_type)
                        text_stress_score += 15
                        if emotion_type in ['accident_risk', 'fatigue']:
                            risk_factors.append(f"⚠️ {emotion_type.upper()}")

            for sentiment_type, keywords in self.POSITIVE_KEYWORDS.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        text_stress_score = max(0, text_stress_score - 10)
                        emotions_detected.append(sentiment_type)

        # Apply Hybrid Logic
        final_score, extra_risks = self.calculate_hybrid_stress(row, text_stress_score)
        risk_factors.extend(extra_risks)

        # Determine stress level
        if final_score >= 80: stress_level = 'CRITICAL'
        elif final_score >= 60: stress_level = 'HIGH'
        elif final_score >= 30: stress_level = 'MEDIUM'
        else: stress_level = 'LOW'
        
        # Recommendations
        if stress_level == 'CRITICAL':
            recommendation = 'IMMEDIATE: Contact driver, offer rest break'
        elif stress_level == 'HIGH':
            recommendation = 'Monitor driver, suggest safety precautions'
        elif stress_level == 'MEDIUM':
            recommendation = 'Gentle reminder about safety protocols'
        else:
            recommendation = 'Driver is doing well'
        
        return {
            'stress_level': stress_level,
            'stress_score': final_score,
            'emotions': list(set(emotions_detected)) if emotions_detected else ["Neutral"],
            'risk_factors': list(set(risk_factors)),
            'recommendation': recommendation,
            'log_text': text
        }

    def analyze_batch(self, df, log_column='Logistics_Delay_Reason'):
        """Updated batch analysis to use row data"""
        results = []
        print(f"Analyzing {len(df)} logs with Hybrid Stress Logic...")
        
        for idx, row in df.iterrows():
            analysis = self.analyze_text_with_data(row, log_column)
            analysis['row_index'] = idx
            results.append(analysis)
        
        self.analyzed_logs = results
        return pd.DataFrame(results)

    # ... (Baaki ke functions: get_summary_report, get_critical_drivers, get_top_risk_factors wese hi rahenge)
    def get_critical_drivers(self):
        return [log for log in self.analyzed_logs if log['stress_level'] == 'CRITICAL']

    def get_top_risk_factors(self, top_n=5):
        risk_count = {}
        for log in self.analyzed_logs:
            for risk in log['risk_factors']:
                risk_count[risk] = risk_count.get(risk, 0) + 1
        return sorted(risk_count.items(), key=lambda x: x, reverse=True)[:top_n]

    def get_summary_report(self):
        if not self.analyzed_logs: return "No logs analyzed yet"
        total = len(self.analyzed_logs)
        critical = len([l for l in self.analyzed_logs if l['stress_level'] == 'CRITICAL'])
        high = len([l for l in self.analyzed_logs if l['stress_level'] == 'HIGH'])
        medium = len([l for l in self.analyzed_logs if l['stress_level'] == 'MEDIUM'])
        low = len([l for l in self.analyzed_logs if l['stress_level'] == 'LOW'])
        avg_stress = sum(l['stress_score'] for l in self.analyzed_logs) / total

        return f"""
DRIVER SENTIMENT ANALYSIS REPORT

STRESS DISTRIBUTION:
    CRITICAL: {critical:3d} drivers ({critical/total*100:.1f}%)
    HIGH:     {high:3d} drivers ({high/total*100:.1f}%)
    MEDIUM:   {medium:3d} drivers ({medium/total*100:.1f}%)
    LOW:      {low:3d} drivers ({low/total*100:.1f}%)

METRICS:
    Average Stress Score: {avg_stress:.1f}/100
    Total Logs Analyzed:  {total}
"""


def _load_pipeline_data():
    script_dir = Path(__file__).resolve().parent
    candidate_paths = [
        script_dir / 'integrated_data_warehouse.csv',
        script_dir / 'dataco_postgresql_structured.csv',
    ]

    for data_path in candidate_paths:
        if data_path.exists():
            print(f"Loading data from: {data_path}")
            return pd.read_csv(data_path)

    raise FileNotFoundError('No pipeline data file found for sentiment analysis')


def _choose_log_column(df):
    preferred_columns = [
        'Logistics_Delay_Reason',
        'Shipment_Status',
        'Delivery Status_x',
        'delivery_status',
        'status',
    ]
    return next((col for col in preferred_columns if col in df.columns), df.columns[0])


if __name__ == '__main__':
    try:
        df = _load_pipeline_data()
        analyzer = DriverSentimentAnalyzer()
        log_column = _choose_log_column(df)
        print(f'Using log column: {log_column}')

        results_df = analyzer.analyze_batch(df, log_column=log_column)
        output_path = Path(__file__).resolve().parent / 'driver_sentiment_analysis.csv'
        results_df.to_csv(output_path, index=False)

        print(analyzer.get_summary_report())
        print(f'✅ Sentiment analysis saved to {output_path.name}')
    except Exception as exc:
        print(f'❌ Sentiment analyzer failed: {exc}')