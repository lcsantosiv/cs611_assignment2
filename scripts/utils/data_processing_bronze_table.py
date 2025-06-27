import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_bronze_loan_table(snapshot_date_str, bronze_lms_directory):
    # prepare arguments
    spark = pyspark.sql.SparkSession.builder \
    .appName("bronze_loan_table") \
    .master("local[*]") \
    .getOrCreate()

    try:
        snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
        
        # connect to source back end - IRL connect to back end source system
        # csv_file_path = "../data/lms_loan_daily.csv"
        csv_file_path = "/opt/airflow/scripts/data/lms_loan_daily.csv"

        # CREATE THE DIRECTORY IF IT DOESN'T EXIST
        if not os.path.exists(bronze_lms_directory):
            os.makedirs(bronze_lms_directory, exist_ok=True)
            print(f"Created directory: {bronze_lms_directory}")

        # load data - IRL ingest from back end source system
        df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
        print(snapshot_date_str + 'row count:', df.count())

        # save bronze table to datamart - IRL connect to database to write
        partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
        filepath = bronze_lms_directory + partition_name
        df.toPandas().to_csv(filepath, index=False)
        print('saved to:', filepath)

    finally:
        spark.stop()

def process_bronze_clickstream_table(snapshot_date_str, bronze_clks_directory):
    # prepare arguments
    spark = pyspark.sql.SparkSession.builder \
    .appName("bronze_clickstream_table") \
    .master("local[*]") \
    .getOrCreate()

    try:
        snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
        
        # connect to source back end - IRL connect to back end source system
        csv_file_path = "/opt/airflow/scripts/data/feature_clickstream.csv" 
        # CREATE THE DIRECTORY IF IT DOESN'T EXIST
        if not os.path.exists(bronze_clks_directory):
            os.makedirs(bronze_clks_directory, exist_ok=True)
            print(f"Created directory: {bronze_clks_directory}")

        # load data - IRL ingest from back end source system
        df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
        print(snapshot_date_str + 'row count:', df.count())

        # save bronze table to datamart - IRL connect to database to write
        partition_name = "bronze_clks_mthly_" + snapshot_date_str.replace('-','_') + '.csv'
        filepath = bronze_clks_directory + partition_name
        df.toPandas().to_csv(filepath, index=False)
        print('saved to:', filepath)

    finally:
        spark.stop()

def process_bronze_attributes_table(snapshot_date_str, bronze_attr_directory):
    # prepare arguments
    spark = pyspark.sql.SparkSession.builder \
    .appName("bronze_attributes_table") \
    .master("local[*]") \
    .getOrCreate()

    try:
        # connect to source back end - IRL connect to back end source system
        csv_file_path = "/opt/airflow/scripts/data/features_attributes.csv"

        # CREATE THE DIRECTORY IF IT DOESN'T EXIST
        if not os.path.exists(bronze_attr_directory):
            os.makedirs(bronze_attr_directory, exist_ok=True)
            print(f"Created directory: {bronze_attr_directory}")
        
        # Check if static file already exists (only load once)
        static_file = "bronze_attr_static.csv"
        static_filepath = bronze_attr_directory + static_file
        
        if os.path.exists(static_filepath):
            print(f"Static attributes file already exists: {static_filepath}")
            return
        
        # load data - IRL ingest from back end source system (load entire file, no filtering)
        df = spark.read.csv(csv_file_path, header=True, inferSchema=True)
        print(f"Static attributes row count: {df.count()}")

        # save bronze table to datamart - IRL connect to database to write (static file)
        filepath = static_filepath
        df.toPandas().to_csv(filepath, index=False)
        print('saved to:', filepath)

    finally:
        spark.stop()

def process_bronze_financials_table(snapshot_date_str, bronze_fin_directory):
    spark = pyspark.sql.SparkSession.builder \
    .appName("bronze_financials_table") \
    .master("local[*]") \
    .getOrCreate()

    try:
        # connect to source back end - IRL connect to back end source system
        csv_file_path = "/opt/airflow/scripts/data/features_financials.csv"

        # CREATE THE DIRECTORY IF IT DOESN'T EXIST
        if not os.path.exists(bronze_fin_directory):
            os.makedirs(bronze_fin_directory, exist_ok=True)
            print(f"Created directory: {bronze_fin_directory}")

        # Check if static file already exists (only load once)
        static_file = "bronze_fin_static.csv"
        static_filepath = bronze_fin_directory + static_file
        
        if os.path.exists(static_filepath):
            print(f"Static financials file already exists: {static_filepath}")
            return

        # load data - IRL ingest from back end source system (load entire file, no filtering)
        df = spark.read.csv(csv_file_path, header=True, inferSchema=True)
        print(f"Static financials row count: {df.count()}")

        # save bronze table to datamart - IRL connect to database to write (static file)
        filepath = static_filepath
        df.toPandas().to_csv(filepath, index=False)
        print('saved to:', filepath)

    finally:
        spark.stop()
