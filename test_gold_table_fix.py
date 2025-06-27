#!/usr/bin/env python3
"""
Test script to verify the gold table processing fix.
"""

import os
import sys
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Add the scripts/utils directory to the path
sys.path.append('scripts/utils')

def test_gold_table_fix(snapshot_date_str="2023-07-01"):
    """
    Test the fixed gold table processing.
    """
    print(f"🧪 Testing fixed gold table processing for {snapshot_date_str}")
    print("=" * 60)
    
    # Define paths
    silver_db = "scripts/datamart/silver"
    gold_db = "scripts/datamart/gold"
    
    try:
        # Import the fixed function
        from data_processing_gold_table import process_gold_table
        
        # Run the processing (this will create and stop its own Spark session)
        print("🚀 Running gold table processing...")
        process_gold_table(silver_db, gold_db, snapshot_date_str)
        
        # Create a NEW Spark session for reading results
        print("📊 Creating new Spark session to read results...")
        spark = SparkSession.builder \
            .appName("GoldTableResultsReader") \
            .master("local[*]") \
            .getOrCreate()
        
        try:
            # Check if files were created
            feature_file = f"gold_feature_store_{snapshot_date_str.replace('-', '_')}.parquet"
            label_file = f"gold_label_store_{snapshot_date_str.replace('-', '_')}.parquet"
            
            feature_path = os.path.join(gold_db, 'feature_store', feature_file)
            label_path = os.path.join(gold_db, 'label_store', label_file)
            
            print(f"\n📊 Checking output files:")
            
            if os.path.exists(feature_path):
                # Load and check feature store
                feature_df = spark.read.parquet(feature_path)
                print(f"✅ Feature store: {feature_df.count()} rows, {len(feature_df.columns)} columns")
                
                # Check for nulls in key columns
                print(f"\n🔍 Checking for nulls in key columns:")
                key_columns = ['age', 'annual_income', 'num_bank_accounts', 'auto_loan', 'avg_fe_1']
                
                for col_name in key_columns:
                    if col_name in feature_df.columns:
                        null_count = feature_df.filter(col(col_name).isNull()).count()
                        total_count = feature_df.count()
                        print(f"   {col_name}: {null_count} nulls out of {total_count} rows ({null_count/total_count*100:.1f}%)")
                    else:
                        print(f"   {col_name}: Column not found")
                
                # Show sample data
                print(f"\n📋 Sample feature store data:")
                feature_df.show(3)
                
            else:
                print(f"❌ Feature store file not found: {feature_path}")
            
            if os.path.exists(label_path):
                # Load and check label store
                label_df = spark.read.parquet(label_path)
                print(f"✅ Label store: {label_df.count()} rows, {len(label_df.columns)} columns")
                
                # Show label distribution
                print(f"\n🏷️  Label distribution:")
                label_df.groupBy("label").count().show()
                
            else:
                print(f"❌ Label store file not found: {label_path}")
            
            print(f"\n🎉 Test completed!")
            
        finally:
            # Stop the Spark session we created for reading
            spark.stop()
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test with the date that was having issues
    test_gold_table_fix("2023-07-01") 