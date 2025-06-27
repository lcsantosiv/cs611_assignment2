#!/usr/bin/env python3
"""
Debug script to check why gold table joins are resulting in null values.
"""

import os
import sys
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count

# Add the scripts/utils directory to the path
sys.path.append('scripts/utils')

def debug_gold_table_joins(snapshot_date_str="2023-01-01"):
    """
    Debug the join logic in gold table processing.
    """
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("GoldTableDebug") \
        .master("local[*]") \
        .getOrCreate()
    
    # Define paths
    silver_db = "scripts/datamart/silver"
    gold_db = "scripts/datamart/gold"
    
    print(f"🔍 Debugging gold table joins for {snapshot_date_str}")
    print("=" * 60)
    
    # Read all silver tables
    tables = ['attributes', 'clickstream', 'financials', 'loan_type', 'lms']
    silver_data = {}
    
    for table in tables:
        partition_name = f'silver_{table}_mthly_{snapshot_date_str.replace("-", "_")}.parquet'
        filepath = os.path.join(silver_db, table, partition_name)
        
        if os.path.exists(filepath):
            df = spark.read.parquet(filepath)
            silver_data[table] = df
            print(f"✅ {table}: {df.count()} rows")
        else:
            print(f"❌ {table}: File not found - {filepath}")
            silver_data[table] = None
    
    print("\n" + "=" * 60)
    print("ANALYZING JOIN KEYS")
    print("=" * 60)
    
    # Check customer_id and snapshot_date in each table
    for table, df in silver_data.items():
        if df is not None:
            print(f"\n📊 {table.upper()} TABLE:")
            print(f"   Total rows: {df.count()}")
            
            # Check for nulls in join keys
            null_customer = df.filter(col("customer_id").isNull()).count()
            null_date = df.filter(col("snapshot_date").isNull()).count()
            print(f"   Null customer_id: {null_customer}")
            print(f"   Null snapshot_date: {null_date}")
            
            # Show sample data
            print(f"   Sample data:")
            df.select("customer_id", "snapshot_date").show(5)
            
            # Check unique combinations
            unique_combinations = df.select("customer_id", "snapshot_date").distinct().count()
            print(f"   Unique (customer_id, snapshot_date) combinations: {unique_combinations}")
    
    print("\n" + "=" * 60)
    print("TESTING JOIN LOGIC")
    print("=" * 60)
    
    # Test the current join logic
    if all(df is not None for df in silver_data.values()):
        df_attributes = silver_data['attributes']
        df_financials = silver_data['financials']
        df_loan_type = silver_data['loan_type']
        df_clickstream = silver_data['clickstream']
        df_lms = silver_data['lms']
        
        # Current logic: Use df_lms filtered to mob == 6 as anchor
        df_lms_mob6 = df_lms.filter(col("mob") == 6)
        print(f"\n🔗 LMS mob=6 anchor: {df_lms_mob6.count()} rows")
        
        # Test joins step by step
        print(f"\n📈 Testing joins step by step:")
        
        # Step 1: Join with attributes
        step1 = df_lms_mob6.select("customer_id", "snapshot_date") \
            .join(df_attributes, ["customer_id", "snapshot_date"], "left")
        print(f"   Step 1 (LMS + attributes): {step1.count()} rows")
        
        # Check for nulls in attributes columns
        attributes_cols = [c for c in df_attributes.columns if c not in ["customer_id", "snapshot_date"]]
        for col_name in attributes_cols[:3]:  # Check first 3 columns
            if col_name in step1.columns:
                null_count = step1.filter(col(col_name).isNull()).count()
                print(f"     {col_name}: {null_count} nulls out of {step1.count()} rows")
        
        # Step 2: Join with financials
        step2 = step1.join(df_financials, ["customer_id", "snapshot_date"], "left")
        print(f"   Step 2 (+ financials): {step2.count()} rows")
        
        # Step 3: Join with loan_type
        step3 = step2.join(df_loan_type, ["customer_id", "snapshot_date"], "left")
        print(f"   Step 3 (+ loan_type): {step3.count()} rows")
        
        # Step 4: Join with clickstream
        df_clickstream_agg = df_clickstream.groupBy("customer_id", "snapshot_date").agg(
            *[F.avg(f'fe_{i}').alias(f"avg_fe_{i}") for i in range(1, 21)]
        )
        step4 = step3.join(df_clickstream_agg, ["customer_id", "snapshot_date"], "left")
        print(f"   Step 4 (+ clickstream): {step4.count()} rows")
        
        print(f"\n🎯 Final result: {step4.count()} rows")
        
        # Show sample of final result
        print(f"\n📋 Sample of final joined data:")
        step4.show(3)
        
        # Check which tables are contributing data
        print(f"\n🔍 Data contribution analysis:")
        for table in ['attributes', 'financials', 'loan_type']:
            if table == 'attributes':
                df_test = df_attributes
            elif table == 'financials':
                df_test = df_financials
            elif table == 'loan_type':
                df_test = df_loan_type
            
            # Check intersection with LMS mob=6
            lms_keys = df_lms_mob6.select("customer_id", "snapshot_date").distinct()
            table_keys = df_test.select("customer_id", "snapshot_date").distinct()
            
            intersection = lms_keys.intersect(table_keys)
            print(f"   {table}: {intersection.count()} matching keys out of {lms_keys.count()} LMS keys")
    
    spark.stop()

def test_alternative_join_logic(snapshot_date_str="2023-01-01"):
    """
    Test alternative join logic using label store as anchor.
    """
    spark = SparkSession.builder \
        .appName("AlternativeJoinTest") \
        .master("local[*]") \
        .getOrCreate()
    
    silver_db = "scripts/datamart/silver"
    
    # Read tables
    tables = ['attributes', 'clickstream', 'financials', 'loan_type', 'lms']
    silver_data = {}
    
    for table in tables:
        partition_name = f'silver_{table}_mthly_{snapshot_date_str.replace("-", "_")}.parquet'
        filepath = os.path.join(silver_db, table, partition_name)
        if os.path.exists(filepath):
            silver_data[table] = spark.read.parquet(filepath)
    
    if all(table in silver_data for table in tables):
        # Build label store first
        from data_processing_gold_table import build_label_store
        df_label = build_label_store(6, 30, silver_data['lms'])
        
        print(f"\n🏷️  Label store: {df_label.count()} rows")
        print("Sample label store:")
        df_label.show(3)
        
        # Use label store as anchor for joins
        print(f"\n🔄 Testing alternative join logic (using label store as anchor):")
        
        df_joined = df_label.select("customer_id", "snapshot_date") \
            .join(silver_data['attributes'], ["customer_id", "snapshot_date"], "left") \
            .join(silver_data['financials'], ["customer_id", "snapshot_date"], "left") \
            .join(silver_data['loan_type'], ["customer_id", "snapshot_date"], "left")
        
        print(f"   Joined with attributes, financials, loan_type: {df_joined.count()} rows")
        
        # Check for nulls
        for table in ['attributes', 'financials', 'loan_type']:
            if table == 'attributes':
                test_cols = ['age', 'annual_income', 'monthly_inhand_salary']
            elif table == 'financials':
                test_cols = ['num_bank_accounts', 'num_credit_card', 'interest_rate']
            elif table == 'loan_type':
                test_cols = ['auto_loan', 'payday_loan', 'student_loan']
            
            for col_name in test_cols:
                if col_name in df_joined.columns:
                    null_count = df_joined.filter(col(col_name).isNull()).count()
                    print(f"     {col_name}: {null_count} nulls out of {df_joined.count()} rows")
        
        # Show sample
        print(f"\n📋 Sample of alternative join result:")
        df_joined.show(3)
    
    spark.stop()

if __name__ == "__main__":
    # Test with a specific date
    test_date = "2023-01-01"
    
    print("🔍 DEBUGGING GOLD TABLE JOINS")
    print("=" * 60)
    
    debug_gold_table_joins(test_date)
    
    print("\n" + "=" * 60)
    print("TESTING ALTERNATIVE JOIN LOGIC")
    print("=" * 60)
    
    test_alternative_join_logic(test_date) 