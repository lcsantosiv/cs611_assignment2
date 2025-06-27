#!/usr/bin/env python3
"""
Test script to run one month of gold table processing.
Use this script in your notebook to test the data processing pipeline.
"""

import os
import sys
import pyspark
from datetime import datetime

# Add the scripts/utils directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts', 'utils'))

# Import the gold table processing function
from data_processing_gold_table import process_gold_table

def run_gold_table_test(snapshot_date_str="2023-01-01"):
    """
    Run gold table processing for a specific month.
    
    Args:
        snapshot_date_str (str): Date string in format 'YYYY-MM-DD'
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"🚀 Starting gold table processing for {snapshot_date_str}")
    
    # Define paths
    silver_db = "scripts/datamart/silver"
    gold_db = "scripts/datamart/gold"
    
    # Check if directories exist
    if not os.path.exists(silver_db):
        print(f"❌ Silver database directory not found: {silver_db}")
        return False
    
    if not os.path.exists(gold_db):
        print(f"📁 Creating gold database directory: {gold_db}")
        os.makedirs(gold_db, exist_ok=True)
    
    # Create subdirectories if they don't exist
    feature_store_dir = os.path.join(gold_db, 'feature_store')
    label_store_dir = os.path.join(gold_db, 'label_store')
    
    os.makedirs(feature_store_dir, exist_ok=True)
    os.makedirs(label_store_dir, exist_ok=True)
    
    try:
        # Run the gold table processing
        print(f"📊 Processing gold table for {snapshot_date_str}")
        process_gold_table(silver_db, gold_db, snapshot_date_str)
        
        print(f"✅ Gold table processing completed successfully for {snapshot_date_str}")
        
        # Check if files were created
        feature_file = f"gold_feature_store_{snapshot_date_str.replace('-', '_')}.parquet"
        label_file = f"gold_label_store_{snapshot_date_str.replace('-', '_')}.parquet"
        
        feature_path = os.path.join(feature_store_dir, feature_file)
        label_path = os.path.join(label_store_dir, label_file)
        
        if os.path.exists(feature_path):
            print(f"✅ Feature store file created: {feature_path}")
        else:
            print(f"❌ Feature store file not found: {feature_path}")
        
        if os.path.exists(label_path):
            print(f"✅ Label store file created: {label_path}")
        else:
            print(f"❌ Label store file not found: {label_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during gold table processing: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_silver_data_availability(snapshot_date_str="2023-01-01"):
    """
    Check if silver data is available for the specified date.
    
    Args:
        snapshot_date_str (str): Date string in format 'YYYY-MM-DD'
    
    Returns:
        dict: Dictionary with availability status for each table
    """
    silver_db = "scripts/datamart/silver"
    tables = ['attributes', 'clickstream', 'financials', 'loan_type', 'lms']
    
    availability = {}
    
    for table in tables:
        partition_name = f'silver_{table}_mthly_{snapshot_date_str.replace("-", "_")}.parquet'
        filepath = os.path.join(silver_db, table, partition_name)
        
        if os.path.exists(filepath):
            # Check if file has data (not empty)
            try:
                spark = pyspark.sql.SparkSession.builder \
                    .appName("DataCheck") \
                    .master("local[*]") \
                    .getOrCreate()
                
                df = spark.read.parquet(filepath)
                row_count = df.count()
                spark.stop()
                
                availability[table] = {
                    'exists': True,
                    'filepath': filepath,
                    'row_count': row_count,
                    'status': '✅ Available' if row_count > 0 else '❌ Empty file'
                }
            except Exception as e:
                availability[table] = {
                    'exists': True,
                    'filepath': filepath,
                    'row_count': 'Error',
                    'status': f'❌ Error reading: {e}'
                }
        else:
            availability[table] = {
                'exists': False,
                'filepath': filepath,
                'row_count': 0,
                'status': '❌ Not found'
            }
    
    return availability

def main():
    """
    Main function to run the test.
    """
    # Test date - you can change this
    test_date = "2023-01-01"
    
    print("🔍 Checking silver data availability...")
    availability = check_silver_data_availability(test_date)
    
    print("\n📋 Silver Data Availability Report:")
    print("=" * 50)
    for table, info in availability.items():
        print(f"{table:12} | {info['status']}")
        if info['exists'] and info['row_count'] != 'Error':
            print(f"{'':12} |   Rows: {info['row_count']}")
    
    print("\n" + "=" * 50)
    
    # Check if all required tables are available
    all_available = all(info['exists'] and info['row_count'] > 0 for info in availability.values())
    
    if all_available:
        print("✅ All silver tables are available. Proceeding with gold table processing...")
        success = run_gold_table_test(test_date)
        
        if success:
            print("\n🎉 Test completed successfully!")
        else:
            print("\n💥 Test failed!")
    else:
        print("❌ Some silver tables are missing or empty. Cannot proceed with gold table processing.")
        print("Please ensure all silver tables are available before running gold table processing.")

if __name__ == "__main__":
    main() 