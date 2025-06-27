"""
Notebook-friendly script for testing gold table processing.
Copy and paste these functions into your Jupyter notebook.
"""

import os
import sys
import pyspark
import pandas as pd
from datetime import datetime

# Add the scripts/utils directory to the path
sys.path.append('scripts/utils')

# Import the gold table processing function
from data_processing_gold_table import process_gold_table

def check_silver_data(snapshot_date="2023-01-01"):
    """
    Check silver data availability for a specific date.
    
    Args:
        snapshot_date (str): Date in 'YYYY-MM-DD' format
    
    Returns:
        dict: Status of each silver table
    """
    silver_db = "scripts/datamart/silver"
    tables = ['attributes', 'clickstream', 'financials', 'loan_type', 'lms']
    
    results = {}
    
    for table in tables:
        partition_name = f'silver_{table}_mthly_{snapshot_date.replace("-", "_")}.parquet'
        filepath = os.path.join(silver_db, table, partition_name)
        
        if os.path.exists(filepath):
            try:
                spark = pyspark.sql.SparkSession.builder \
                    .appName("DataCheck") \
                    .master("local[*]") \
                    .getOrCreate()
                
                df = spark.read.parquet(filepath)
                row_count = df.count()
                spark.stop()
                
                results[table] = {
                    'status': '✅ Available',
                    'rows': row_count,
                    'filepath': filepath
                }
            except Exception as e:
                results[table] = {
                    'status': f'❌ Error: {str(e)}',
                    'rows': 0,
                    'filepath': filepath
                }
        else:
            results[table] = {
                'status': '❌ Not found',
                'rows': 0,
                'filepath': filepath
            }
    
    return results

def run_gold_processing(snapshot_date="2023-01-01"):
    """
    Run gold table processing for a specific date.
    
    Args:
        snapshot_date (str): Date in 'YYYY-MM-DD' format
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"🚀 Processing gold table for {snapshot_date}")
    
    # Define paths
    silver_db = "scripts/datamart/silver"
    gold_db = "scripts/datamart/gold"
    
    # Create directories if they don't exist
    os.makedirs(os.path.join(gold_db, 'feature_store'), exist_ok=True)
    os.makedirs(os.path.join(gold_db, 'label_store'), exist_ok=True)
    
    try:
        # Run processing
        process_gold_table(silver_db, gold_db, snapshot_date)
        
        # Check output files
        feature_file = f"gold_feature_store_{snapshot_date.replace('-', '_')}.parquet"
        label_file = f"gold_label_store_{snapshot_date.replace('-', '_')}.parquet"
        
        feature_path = os.path.join(gold_db, 'feature_store', feature_file)
        label_path = os.path.join(gold_db, 'label_store', label_file)
        
        print(f"✅ Processing completed!")
        print(f"📊 Feature store: {feature_path}")
        print(f"🏷️  Label store: {label_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_gold_data(snapshot_date="2023-01-01"):
    """
    Load and display gold table data for inspection.
    
    Args:
        snapshot_date (str): Date in 'YYYY-MM-DD' format
    
    Returns:
        tuple: (feature_df, label_df) or (None, None) if error
    """
    gold_db = "scripts/datamart/gold"
    
    feature_file = f"gold_feature_store_{snapshot_date.replace('-', '_')}.parquet"
    label_file = f"gold_label_store_{snapshot_date.replace('-', '_')}.parquet"
    
    feature_path = os.path.join(gold_db, 'feature_store', feature_file)
    label_path = os.path.join(gold_db, 'label_store', label_file)
    
    try:
        # Load feature store
        if os.path.exists(feature_path):
            feature_df = pd.read_parquet(feature_path)
            print(f"✅ Feature store loaded: {len(feature_df)} rows, {len(feature_df.columns)} columns")
        else:
            print(f"❌ Feature store not found: {feature_path}")
            feature_df = None
        
        # Load label store
        if os.path.exists(label_path):
            label_df = pd.read_parquet(label_path)
            print(f"✅ Label store loaded: {len(label_df)} rows, {len(label_df.columns)} columns")
        else:
            print(f"❌ Label store not found: {label_path}")
            label_df = None
        
        return feature_df, label_df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None

# Example usage functions for notebook
def quick_test():
    """
    Quick test function - check data and run processing for 2023-01-01
    """
    print("🔍 Checking silver data availability...")
    silver_status = check_silver_data("2023-01-01")
    
    print("\n📋 Silver Data Status:")
    for table, info in silver_status.items():
        print(f"  {table:12}: {info['status']} ({info['rows']} rows)")
    
    print("\n🚀 Running gold table processing...")
    success = run_gold_processing("2023-01-01")
    
    if success:
        print("\n📊 Loading results...")
        feature_df, label_df = load_gold_data("2023-01-01")
        
        if feature_df is not None:
            print(f"\nFeature store preview:")
            print(feature_df.head())
            print(f"\nFeature store info:")
            print(feature_df.info())
        
        if label_df is not None:
            print(f"\nLabel store preview:")
            print(label_df.head())
            print(f"\nLabel distribution:")
            print(label_df['label'].value_counts())
    
    return success

def test_multiple_months(months=["2023-01-01", "2023-02-01", "2023-03-01"]):
    """
    Test processing for multiple months.
    
    Args:
        months (list): List of dates to test
    """
    results = {}
    
    for month in months:
        print(f"\n{'='*50}")
        print(f"Testing month: {month}")
        print(f"{'='*50}")
        
        # Check silver data
        silver_status = check_silver_data(month)
        all_available = all(info['rows'] > 0 for info in silver_status.values())
        
        if all_available:
            success = run_gold_processing(month)
            results[month] = success
        else:
            print(f"❌ Skipping {month} - silver data not available")
            results[month] = False
    
    print(f"\n{'='*50}")
    print("SUMMARY:")
    for month, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {month}: {status}")
    
    return results 