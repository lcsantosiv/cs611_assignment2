#!/usr/bin/env python3
"""
Debug Script: Data Consistency Check Across Months

This script checks record counts across months in the data preprocessing pipeline
to ensure data consistency and identify any missing or duplicate records.

Usage in Jupyter:
    %run debug_data_consistency.py
    or
    exec(open('debug_data_consistency.py').read())
"""

import os
import pandas as pd
from pyspark.sql import SparkSession
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import glob
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Set your gold store base path here (relative to notebook or absolute)
GOLD_BASE_PATH = "scripts/datamart/gold"
FEATURE_STORE_PATH = os.path.join(GOLD_BASE_PATH, "feature_store")
LABEL_STORE_PATH = os.path.join(GOLD_BASE_PATH, "label_store")

spark = SparkSession.builder \
    .appName("GoldStoreCheck") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .getOrCreate()

def get_monthly_files(directory):
    files = glob.glob(os.path.join(directory, "*.parquet"))
    month_map = {}
    for f in files:
        fname = os.path.basename(f)
        # Try to extract YYYY_MM from filename
        date_part = None
        # gold_feature_store_2024_05_01.parquet or 2023_01_01.parquet
        for pat in [r"(\d{4}_\d{2})_\d{2}", r"(\d{4}_\d{2})"]:
            import re
            m = re.search(pat, fname)
            if m:
                date_part = m.group(1)
                break
        if date_part:
            month_map[date_part] = f
    return month_map

def count_parquet_rows(path):
    try:
        df = spark.read.parquet(path)
        return df.count()
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return 0

def check_gold_stores():
    print("Checking gold feature_store and label_store...")
    feature_files = get_monthly_files(FEATURE_STORE_PATH)
    label_files = get_monthly_files(LABEL_STORE_PATH)
    all_months = sorted(set(feature_files.keys()) | set(label_files.keys()))
    print(f"\n{'Month':<10} {'Feature Rows':>15} {'Label Rows':>15} {'Match?':>8}")
    print("-"*50)
    mismatch = False
    for month in all_months:
        f_path = feature_files.get(month)
        l_path = label_files.get(month)
        f_count = count_parquet_rows(f_path) if f_path else 0
        l_count = count_parquet_rows(l_path) if l_path else 0
        match = f_count == l_count and f_count > 0
        print(f"{month:<10} {f_count:>15,} {l_count:>15,} {str(match):>8}")
        if not match:
            mismatch = True
    if not mismatch:
        print("\n✅ All months have matching, nonzero row counts in feature_store and label_store!")
    else:
        print("\n❌ Some months are missing or have mismatched row counts!")

# Run instantly in Jupyter
check_gold_stores()

class DataConsistencyChecker:
    def __init__(self, base_path: str = "/opt/airflow/scripts/datamart"):
        """
        Initialize the data consistency checker.
        
        Args:
            base_path: Base path to the datamart directory
        """
        self.base_path = base_path
        self.spark = SparkSession.builder \
            .appName("DataConsistencyCheck") \
            .config("spark.driver.memory", "2g") \
            .config("spark.executor.memory", "2g") \
            .getOrCreate()
        
        # Define paths
        self.bronze_paths = {
            'lms': f"{base_path}/bronze/lms",
            'attributes': f"{base_path}/bronze/attributes", 
            'financials': f"{base_path}/bronze/financials",
            'clickstream': f"{base_path}/bronze/clickstream"
        }
        
        self.silver_paths = {
            'lms': f"{base_path}/silver/lms",
            'attributes': f"{base_path}/silver/attributes",
            'financials': f"{base_path}/silver/financials", 
            'clickstream': f"{base_path}/silver/clickstream"
        }
        
        self.gold_paths = {
            'feature_store': f"{base_path}/gold/feature_store",
            'label_store': f"{base_path}/gold/label_store"
        }
    
    def get_monthly_files(self, directory: str, pattern: str = "*.csv") -> Dict[str, List[str]]:
        """
        Get all files organized by month for a given directory.
        
        Args:
            directory: Directory to search
            pattern: File pattern to match
            
        Returns:
            Dictionary with month as key and list of files as value
        """
        monthly_files = {}
        
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} does not exist")
            return monthly_files
        
        # Get all files matching pattern
        files = glob.glob(os.path.join(directory, pattern))
        
        for file_path in files:
            # Extract month from filename
            filename = os.path.basename(file_path)
            
            # Try different patterns for month extraction
            month = None
            
            # Pattern 1: bronze_attr_mthly_2023_01_01.csv
            if 'mthly_' in filename:
                parts = filename.split('mthly_')
                if len(parts) > 1:
                    date_part = parts[1].split('.')[0]  # Remove extension
                    try:
                        date_obj = datetime.strptime(date_part, '%Y_%m_%d')
                        month = date_obj.strftime('%Y_%m')
                    except:
                        pass
            
            # Pattern 2: silver_attributes_mthly_2023_01_01.parquet
            elif 'mthly_' in filename and '.parquet' in filename:
                parts = filename.split('mthly_')
                if len(parts) > 1:
                    date_part = parts[1].split('.')[0]  # Remove extension
                    try:
                        date_obj = datetime.strptime(date_part, '%Y_%m_%d')
                        month = date_obj.strftime('%Y_%m')
                    except:
                        pass
            
            # Pattern 3: gold_feature_store_2024_05_01.parquet
            elif 'gold_feature_store_' in filename:
                date_part = filename.split('gold_feature_store_')[1].split('.')[0]
                try:
                    date_obj = datetime.strptime(date_part, '%Y_%m_%d')
                    month = date_obj.strftime('%Y_%m')
                except:
                    pass
            
            # Pattern 4: gold_label_store_2024_05_01.parquet
            elif 'gold_label_store_' in filename:
                date_part = filename.split('gold_label_store_')[1].split('.')[0]
                try:
                    date_obj = datetime.strptime(date_part, '%Y_%m_%d')
                    month = date_obj.strftime('%Y_%m')
                except:
                    pass
            
            # Pattern 5: 2023_01_01.parquet (older format)
            elif filename.endswith('.parquet') and len(filename.split('_')) >= 3:
                try:
                    date_part = filename.split('.')[0]
                    date_obj = datetime.strptime(date_part, '%Y_%m_%d')
                    month = date_obj.strftime('%Y_%m')
                except:
                    pass
            
            if month:
                if month not in monthly_files:
                    monthly_files[month] = []
                monthly_files[month].append(file_path)
            else:
                print(f"Warning: Could not extract month from filename: {filename}")
        
        return monthly_files
    
    def count_records_csv(self, file_path: str) -> int:
        """Count records in a CSV file."""
        try:
            df = pd.read_csv(file_path)
            return len(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return 0
    
    def count_records_parquet(self, file_path: str) -> int:
        """Count records in a parquet file/directory."""
        try:
            df = self.spark.read.parquet(file_path)
            return df.count()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return 0
    
    def check_bronze_layer(self) -> Dict[str, Dict[str, int]]:
        """Check record counts in bronze layer."""
        print("=== BRONZE LAYER CHECK ===")
        bronze_counts = {}
        
        for table_name, directory in self.bronze_paths.items():
            print(f"\nChecking {table_name}...")
            monthly_files = self.get_monthly_files(directory, "*.csv")
            table_counts = {}
            
            for month, files in monthly_files.items():
                total_count = 0
                for file_path in files:
                    count = self.count_records_csv(file_path)
                    total_count += count
                    print(f"  {os.path.basename(file_path)}: {count:,} records")
                
                table_counts[month] = total_count
                print(f"  Month {month} total: {total_count:,} records")
            
            bronze_counts[table_name] = table_counts
        
        return bronze_counts
    
    def check_silver_layer(self) -> Dict[str, Dict[str, int]]:
        """Check record counts in silver layer."""
        print("\n=== SILVER LAYER CHECK ===")
        silver_counts = {}
        
        for table_name, directory in self.silver_paths.items():
            print(f"\nChecking {table_name}...")
            monthly_files = self.get_monthly_files(directory, "*.parquet")
            table_counts = {}
            
            for month, files in monthly_files.items():
                total_count = 0
                for file_path in files:
                    count = self.count_records_parquet(file_path)
                    total_count += count
                    print(f"  {os.path.basename(file_path)}: {count:,} records")
                
                table_counts[month] = total_count
                print(f"  Month {month} total: {total_count:,} records")
            
            silver_counts[table_name] = table_counts
        
        return silver_counts
    
    def check_gold_layer(self) -> Dict[str, Dict[str, int]]:
        """Check record counts in gold layer."""
        print("\n=== GOLD LAYER CHECK ===")
        gold_counts = {}
        
        for store_name, directory in self.gold_paths.items():
            print(f"\nChecking {store_name}...")
            monthly_files = self.get_monthly_files(directory, "*.parquet")
            store_counts = {}
            
            for month, files in monthly_files.items():
                total_count = 0
                for file_path in files:
                    count = self.count_records_parquet(file_path)
                    total_count += count
                    print(f"  {os.path.basename(file_path)}: {count:,} records")
                
                store_counts[month] = total_count
                print(f"  Month {month} total: {total_count:,} records")
            
            gold_counts[store_name] = store_counts
        
        return gold_counts
    
    def analyze_consistency(self, bronze_counts: Dict, silver_counts: Dict, gold_counts: Dict):
        """Analyze data consistency across layers."""
        print("\n=== DATA CONSISTENCY ANALYSIS ===")
        
        # Get all months
        all_months = set()
        for table_counts in bronze_counts.values():
            all_months.update(table_counts.keys())
        for table_counts in silver_counts.values():
            all_months.update(table_counts.keys())
        for store_counts in gold_counts.values():
            all_months.update(store_counts.keys())
        
        all_months = sorted(all_months)
        
        # Create summary DataFrame
        summary_data = []
        
        for month in all_months:
            row = {'month': month}
            
            # Bronze layer counts
            for table_name, table_counts in bronze_counts.items():
                row[f'bronze_{table_name}'] = table_counts.get(month, 0)
            
            # Silver layer counts
            for table_name, table_counts in silver_counts.items():
                row[f'silver_{table_name}'] = table_counts.get(month, 0)
            
            # Gold layer counts
            for store_name, store_counts in gold_counts.items():
                row[f'gold_{store_name}'] = store_counts.get(month, 0)
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Display summary
        print("\nMonthly Record Count Summary:")
        print(summary_df.to_string(index=False))
        
        # Check for missing months
        print("\n=== MISSING MONTHS ANALYSIS ===")
        for table_name, table_counts in bronze_counts.items():
            missing_months = [m for m in all_months if m not in table_counts]
            if missing_months:
                print(f"Bronze {table_name} missing months: {missing_months}")
        
        for table_name, table_counts in silver_counts.items():
            missing_months = [m for m in all_months if m not in table_counts]
            if missing_months:
                print(f"Silver {table_name} missing months: {missing_months}")
        
        for store_name, store_counts in gold_counts.items():
            missing_months = [m for m in all_months if m not in store_counts]
            if missing_months:
                print(f"Gold {store_name} missing months: {missing_months}")
        
        # Check for zero counts
        print("\n=== ZERO COUNT ANALYSIS ===")
        for table_name, table_counts in bronze_counts.items():
            zero_months = [m for m, count in table_counts.items() if count == 0]
            if zero_months:
                print(f"Bronze {table_name} zero count months: {zero_months}")
        
        for table_name, table_counts in silver_counts.items():
            zero_months = [m for m, count in table_counts.items() if count == 0]
            if zero_months:
                print(f"Silver {table_name} zero count months: {zero_months}")
        
        for store_name, store_counts in gold_counts.items():
            zero_months = [m for m, count in store_counts.items() if count == 0]
            if zero_months:
                print(f"Gold {store_name} zero count months: {zero_months}")
        
        return summary_df
    
    def plot_monthly_trends(self, summary_df: pd.DataFrame):
        """Plot monthly trends for visual analysis."""
        print("\n=== GENERATING PLOTS ===")
        
        # Set up the plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Monthly Record Counts by Layer', fontsize=16)
        
        # Bronze layer plot
        bronze_cols = [col for col in summary_df.columns if col.startswith('bronze_')]
        if bronze_cols:
            ax1 = axes[0, 0]
            for col in bronze_cols:
                ax1.plot(summary_df['month'], summary_df[col], marker='o', label=col.replace('bronze_', ''))
            ax1.set_title('Bronze Layer')
            ax1.set_ylabel('Record Count')
            ax1.legend()
            ax1.tick_params(axis='x', rotation=45)
        
        # Silver layer plot
        silver_cols = [col for col in summary_df.columns if col.startswith('silver_')]
        if silver_cols:
            ax2 = axes[0, 1]
            for col in silver_cols:
                ax2.plot(summary_df['month'], summary_df[col], marker='s', label=col.replace('silver_', ''))
            ax2.set_title('Silver Layer')
            ax2.set_ylabel('Record Count')
            ax2.legend()
            ax2.tick_params(axis='x', rotation=45)
        
        # Gold layer plot
        gold_cols = [col for col in summary_df.columns if col.startswith('gold_')]
        if gold_cols:
            ax3 = axes[1, 0]
            for col in gold_cols:
                ax3.plot(summary_df['month'], summary_df[col], marker='^', label=col.replace('gold_', ''))
            ax3.set_title('Gold Layer')
            ax3.set_ylabel('Record Count')
            ax3.legend()
            ax3.tick_params(axis='x', rotation=45)
        
        # Total records plot
        ax4 = axes[1, 1]
        total_records = summary_df[[col for col in summary_df.columns if col != 'month']].sum(axis=1)
        ax4.plot(summary_df['month'], total_records, marker='o', color='red', linewidth=2)
        ax4.set_title('Total Records Across All Layers')
        ax4.set_ylabel('Total Record Count')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        print("Plots generated successfully!")
    
    def run_full_check(self):
        """Run the complete data consistency check."""
        print("Starting Data Consistency Check...")
        print(f"Base path: {self.base_path}")
        
        # Check all layers
        bronze_counts = self.check_bronze_layer()
        silver_counts = self.check_silver_layer()
        gold_counts = self.check_gold_layer()
        
        # Analyze consistency
        summary_df = self.analyze_consistency(bronze_counts, silver_counts, gold_counts)
        
        # Generate plots
        self.plot_monthly_trends(summary_df)
        
        print("\n=== DATA CONSISTENCY CHECK COMPLETE ===")
        
        return {
            'bronze_counts': bronze_counts,
            'silver_counts': silver_counts,
            'gold_counts': gold_counts,
            'summary_df': summary_df
        }

# Create global instance for easy use in Jupyter
checker = DataConsistencyChecker()

def run_consistency_check():
    """Run the complete data consistency check."""
    return checker.run_full_check()

# Example usage in Jupyter:
# results = run_consistency_check()
# summary_df = results['summary_df']
# summary_df.head() 