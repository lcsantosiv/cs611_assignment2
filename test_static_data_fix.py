#!/usr/bin/env python3
"""
Test script to verify static data fix for attributes and financials.
This script tests the updated logic where:
- attributes and financials are loaded once as static data
- clickstream and lms remain monthly data
"""

import os
import sys
import pyspark
import pyspark.sql.functions as F
from datetime import datetime
from pyspark.sql.functions import col

# Add the utils path
sys.path.append('/opt/airflow/scripts/utils')

def test_static_data_logic():
    """
    Test the static data logic by simulating the pipeline
    """
    print("=== Testing Static Data Logic ===")
    
    # Initialize Spark
    spark = pyspark.sql.SparkSession.builder \
        .appName("test_static_data") \
        .master("local[*]") \
        .getOrCreate()
    
    try:
        # Test 1: Check if static data files are created correctly
        print("\n1. Testing static data file creation...")
        
        # Simulate bronze layer processing for static data
        from data_processing_bronze_table import process_bronze_attributes_table, process_bronze_financials_table
        
        # Test attributes (static)
        print("Processing bronze attributes (static)...")
        process_bronze_attributes_table("2023-01-01", "/opt/airflow/scripts/datamart/bronze/attributes/")
        
        # Test financials (static)
        print("Processing bronze financials (static)...")
        process_bronze_financials_table("2023-01-01", "/opt/airflow/scripts/datamart/bronze/financials/")
        
        # Check if static files were created
        static_files = [
            "/opt/airflow/scripts/datamart/bronze/attributes/bronze_attr_static.csv",
            "/opt/airflow/scripts/datamart/bronze/financials/bronze_fin_static.csv"
        ]
        
        for file_path in static_files:
            if os.path.exists(file_path):
                print(f"✓ Static file created: {file_path}")
                # Check file size
                file_size = os.path.getsize(file_path)
                print(f"  File size: {file_size} bytes")
            else:
                print(f"✗ Static file missing: {file_path}")
        
        # Test 2: Check if monthly data files are created correctly
        print("\n2. Testing monthly data file creation...")
        
        from data_processing_bronze_table import process_bronze_clickstream_table, process_bronze_loan_table
        
        # Test clickstream (monthly)
        print("Processing bronze clickstream (monthly)...")
        process_bronze_clickstream_table("2023-01-01", "/opt/airflow/scripts/datamart/bronze/clickstream/")
        
        # Test lms (monthly)
        print("Processing bronze lms (monthly)...")
        process_bronze_loan_table("2023-01-01", "/opt/airflow/scripts/datamart/bronze/lms/")
        
        # Check if monthly files were created
        monthly_files = [
            "/opt/airflow/scripts/datamart/bronze/clickstream/bronze_clks_mthly_2023_01_01.csv",
            "/opt/airflow/scripts/datamart/bronze/lms/bronze_loan_daily_2023_01_01.csv"
        ]
        
        for file_path in monthly_files:
            if os.path.exists(file_path):
                print(f"✓ Monthly file created: {file_path}")
                # Check file size
                file_size = os.path.getsize(file_path)
                print(f"  File size: {file_size} bytes")
            else:
                print(f"✗ Monthly file missing: {file_path}")
        
        # Test 3: Test silver layer processing
        print("\n3. Testing silver layer processing...")
        
        from data_processing_silver_table import process_silver_table
        
        # Test static data in silver layer
        print("Processing silver attributes (static)...")
        process_silver_table('attributes', '/opt/airflow/scripts/datamart/bronze/', '/opt/airflow/scripts/datamart/silver/', '2023-01-01')
        
        print("Processing silver financials (static)...")
        process_silver_table('financials', '/opt/airflow/scripts/datamart/bronze/', '/opt/airflow/scripts/datamart/silver/', '2023-01-01')
        
        # Test monthly data in silver layer
        print("Processing silver clickstream (monthly)...")
        process_silver_table('clickstream', '/opt/airflow/scripts/datamart/bronze/', '/opt/airflow/scripts/datamart/silver/', '2023-01-01')
        
        print("Processing silver lms (monthly)...")
        process_silver_table('lms', '/opt/airflow/scripts/datamart/bronze/', '/opt/airflow/scripts/datamart/silver/', '2023-01-01')
        
        # Check if silver files were created
        silver_files = [
            "/opt/airflow/scripts/datamart/silver/attributes/silver_attributes_static.parquet",
            "/opt/airflow/scripts/datamart/silver/financials/silver_financials_static.parquet",
            "/opt/airflow/scripts/datamart/silver/clickstream/silver_clickstream_mthly_2023_01_01.parquet",
            "/opt/airflow/scripts/datamart/silver/lms/silver_lms_mthly_2023_01_01.parquet"
        ]
        
        for file_path in silver_files:
            if os.path.exists(file_path):
                print(f"✓ Silver file created: {file_path}")
            else:
                print(f"✗ Silver file missing: {file_path}")
        
        # Test 4: Test gold layer processing
        print("\n4. Testing gold layer processing...")
        
        from data_processing_gold_table import process_gold_table
        
        print("Processing gold tables...")
        process_gold_table('/opt/airflow/scripts/datamart/silver/', '/opt/airflow/scripts/datamart/gold/', '2023-01-01')
        
        # Check if gold files were created
        gold_files = [
            "/opt/airflow/scripts/datamart/gold/feature_store/gold_feature_store_2023_01_01.parquet",
            "/opt/airflow/scripts/datamart/gold/label_store/gold_label_store_2023_01_01.parquet"
        ]
        
        for file_path in gold_files:
            if os.path.exists(file_path):
                print(f"✓ Gold file created: {file_path}")
                # Read and check data
                df = spark.read.parquet(file_path)
                print(f"  Row count: {df.count()}")
                print(f"  Columns: {len(df.columns)}")
            else:
                print(f"✗ Gold file missing: {file_path}")
        
        # Test 5: Verify data consistency
        print("\n5. Testing data consistency...")
        
        # Read gold feature store and check if all data sources are joined correctly
        feature_store_path = "/opt/airflow/scripts/datamart/gold/feature_store/gold_feature_store_2023_01_01.parquet"
        if os.path.exists(feature_store_path):
            df_features = spark.read.parquet(feature_store_path)
            print(f"Feature store row count: {df_features.count()}")
            
            # Check for null values in key columns
            key_columns = ['customer_id', 'snapshot_date']
            for col_name in key_columns:
                null_count = df_features.filter(col(col_name).isNull()).count()
                print(f"Null values in {col_name}: {null_count}")
            
            # Check if we have data from all sources
            source_indicators = {
                'attributes': ['age', 'occupation'],
                'financials': ['annual_income', 'credit_history_age_month'],
                'clickstream': ['avg_fe_1', 'avg_fe_2'],
                'lms': ['mob', 'dpd']
            }
            
            for source, columns in source_indicators.items():
                available_cols = [col for col in columns if col in df_features.columns]
                if available_cols:
                    non_null_count = df_features.select(available_cols).na.drop().count()
                    print(f"{source} data available: {non_null_count} rows with non-null values")
                else:
                    print(f"{source} data: No columns found")
        
        print("\n=== Test completed ===")
        
    finally:
        spark.stop()

if __name__ == "__main__":
    test_static_data_logic() 