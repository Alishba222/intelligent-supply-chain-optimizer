# Data Validation Framework for ETL Pipeline
import pandas as pd
import numpy as np

class DataValidator:
    """Comprehensive data validation for supply chain data"""
    
    def __init__(self):
        self.validation_report = {}
    
    def validate_postgres_data(self, df):
        """Validate PostgreSQL source data"""
        report = {
            'rows': df.shape[0],
            'columns': df.shape[1],
            'missing_count': df.isnull().sum().sum(),
            'missing_percent': df.isnull().sum().sum() / df.size * 100,
            'duplicates': df.duplicated().sum(),
            'numeric_cols': df.select_dtypes(include=[np.number]).shape[1],
        }
        
        # Check for critical columns
        expected_cols = ['order_id', 'Late_delivery_risk', 'Logistics_Delay_Reason']
        missing_cols = [col for col in expected_cols if col not in df.columns]
        report['missing_columns'] = missing_cols
        
        # Check for negative values in numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        negative_counts = (numeric_df < 0).sum()
        report['negative_values'] = negative_counts[negative_counts > 0].to_dict()
        
        self.validation_report['postgres'] = report
        return report
    
    def validate_mongodb_data(self, df):
        """Validate MongoDB source data"""
        report = {
            'rows': df.shape[0],
            'columns': df.shape[1],
            'missing_count': df.isnull().sum().sum(),
            'missing_percent': df.isnull().sum().sum() / df.size * 100,
            'duplicates': df.duplicated().sum(),
        }
        
        # Check for sensor data
        sensor_cols = ['Temperature', 'Humidity', 'Traffic_Status']
        available_sensors = [col for col in sensor_cols if col in df.columns]
        missing_sensors = [col for col in sensor_cols if col not in df.columns]
        
        report['available_sensors'] = available_sensors
        report['missing_sensors'] = missing_sensors
        
        # Check sensor data ranges
        if 'Temperature' in df.columns:
            temp_stats = {
                'min': df['Temperature'].min(),
                'max': df['Temperature'].max(),
                'mean': df['Temperature'].mean(),
                'out_of_range': ((df['Temperature'] < -50) | (df['Temperature'] > 60)).sum()
            }
            report['temperature_stats'] = temp_stats
        
        if 'Humidity' in df.columns:
            humidity_stats = {
                'min': df['Humidity'].min(),
                'max': df['Humidity'].max(),
                'mean': df['Humidity'].mean(),
                'out_of_range': ((df['Humidity'] < 0) | (df['Humidity'] > 100)).sum()
            }
            report['humidity_stats'] = humidity_stats
        
        self.validation_report['mongodb'] = report
        return report
    
    def validate_merge(self, pg_df, mongo_df, integrated_df, join_key_pg, join_key_mongo):
        """Validate ETL merge operation"""
        report = {
            'postgres_rows_before': pg_df.shape[0],
            'mongodb_rows_before': mongo_df.shape[0],
            'integrated_rows_after': integrated_df.shape[0],
            'data_retention_rate': integrated_df.shape[0] / pg_df.shape[0] * 100,
            'data_loss_rate': (1 - integrated_df.shape[0] / pg_df.shape[0]) * 100,
        }
        
        # Check join key quality
        pg_unique_keys = pg_df[join_key_pg].nunique()
        mongo_unique_keys = mongo_df[join_key_mongo].nunique()
        pg_null_keys = pg_df[join_key_pg].isnull().sum()
        mongo_null_keys = mongo_df[join_key_mongo].isnull().sum()
        
        report['join_key_analysis'] = {
            f'{join_key_pg}_unique': pg_unique_keys,
            f'{join_key_mongo}_unique': mongo_unique_keys,
            f'{join_key_pg}_nulls': pg_null_keys,
            f'{join_key_mongo}_nulls': mongo_null_keys,
        }
        
        # Check for common keys
        common_keys = set(pg_df[join_key_pg].dropna()) & set(mongo_df[join_key_mongo].dropna())
        report['common_keys'] = len(common_keys)
        report['join_efficiency'] = len(common_keys) / min(pg_unique_keys, mongo_unique_keys) * 100
        
        self.validation_report['merge'] = report
        return report
    
    def generate_report(self):
        """Generate comprehensive validation report"""
        print("\n" + "="*80)
        print("DATA VALIDATION REPORT")
        print("="*80)
        
        for section, report in self.validation_report.items():
            print(f"\n{section.upper()}:")
            for key, value in report.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    - {k}: {v}")
                else:
                    print(f"  {key}: {value}")
        
        print("\n" + "="*80)
    
    def get_summary(self):
        """Get validation summary with issues"""
        issues = []
        
        pg_report = self.validation_report.get('postgres', {})
        if pg_report.get('missing_percent', 0) > 5:
            issues.append(f"WARNING: PostgreSQL has {pg_report['missing_percent']:.2f}% missing values")
        
        mongo_report = self.validation_report.get('mongodb', {})
        if mongo_report.get('missing_percent', 0) > 15:
            issues.append(f"WARNING: MongoDB has {mongo_report['missing_percent']:.2f}% missing values")
        
        merge_report = self.validation_report.get('merge', {})
        if merge_report.get('data_loss_rate', 0) > 50:
            issues.append(f"CRITICAL: {merge_report['data_loss_rate']:.2f}% data loss in merge")
        
        if merge_report.get('join_efficiency', 0) < 10:
            issues.append(f"WARNING: Low join efficiency ({merge_report['join_efficiency']:.2f}%)")
        
        return issues
