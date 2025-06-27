from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta
import os
import sys
from pyspark.sql import SparkSession

sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts/utils'))
from data_processing_bronze_table import process_bronze_loan_table, process_bronze_clickstream_table, process_bronze_attributes_table, process_bronze_financials_table
from silver_processing_retrofit import process_silver_table as process_silver_table_retrofit
from gold_processing_retrofit import process_gold_table as process_gold_table_retrofit
from dag_functions import check_static_data_loaded

# === Spark Session Wrapper Functions ===
def run_silver_table_lms(table_name, snapshot_date, bronze_dir, silver_dir):
    spark = SparkSession.builder.appName("SilverTableLMS").getOrCreate()
    try:
        process_silver_table_retrofit(table_name, snapshot_date, bronze_dir, silver_dir, spark)
        return None
    finally:
        spark.stop()

def run_silver_table_clickstream(table_name, snapshot_date, bronze_dir, silver_dir):
    spark = SparkSession.builder.appName("SilverTableClickstream").getOrCreate()
    try:
        process_silver_table_retrofit(table_name, snapshot_date, bronze_dir, silver_dir, spark)
        return None
    finally:
        spark.stop()

def run_silver_table_attributes(table_name, snapshot_date, bronze_dir, silver_dir):
    spark = SparkSession.builder.appName("SilverTableAttributes").getOrCreate()
    try:
        process_silver_table_retrofit(table_name, snapshot_date, bronze_dir, silver_dir, spark)
        return None
    finally:
        spark.stop()

def run_silver_table_financials(table_name, snapshot_date, bronze_dir, silver_dir):
    spark = SparkSession.builder.appName("SilverTableFinancials").getOrCreate()
    try:
        process_silver_table_retrofit(table_name, snapshot_date, bronze_dir, silver_dir, spark)
        return None
    finally:
        spark.stop()

def run_gold_table(snapshot_date, silver_dir, gold_dir, dpd, mob):
    spark = SparkSession.builder.appName("GoldTable").getOrCreate()
    try:
        process_gold_table_retrofit(snapshot_date, silver_dir, gold_dir, spark, dpd, mob)
        return None
    finally:
        spark.stop()

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': True,
    'start_date': datetime(2023, 1, 1),
    'end_date': datetime(2024, 12, 31),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'data_preprocessing_pipeline',
    default_args=default_args,
    description='Data preprocessing pipeline for credit scoring',
    schedule_interval='0 6 1 * *',
    catchup=True,
    tags=['data-preprocessing', 'credit-scoring']
) as dag:
    wait_for_previous_run = DummyOperator(task_id='wait_for_previous_run')
    start = DummyOperator(task_id='start_pipeline')
    start_preprocessing = DummyOperator(task_id='start_preprocessing')

    check_static_data = BranchPythonOperator(
        task_id='check_static_data',
        python_callable=check_static_data_loaded
    )

    dep_check_source_data_bronze_1 = DummyOperator(task_id='dep_check_source_lms')
    dep_check_source_data_bronze_2 = DummyOperator(task_id='dep_check_source_attributes')
    dep_check_source_data_bronze_3 = DummyOperator(task_id='dep_check_source_financials')
    dep_check_source_data_bronze_4 = DummyOperator(task_id='dep_check_source_clickstream')

    # Bronze Layer Processing
    bronze_table_1 = PythonOperator(
        task_id='run_bronze_table_lms',
        python_callable=process_bronze_loan_table,
        op_args=['{{ ds }}', '/opt/airflow/scripts/datamart/bronze/lms/']
    )
    bronze_table_4 = PythonOperator(
        task_id='run_bronze_table_clickstream',
        python_callable=process_bronze_clickstream_table,
        op_args=['{{ ds }}', '/opt/airflow/scripts/datamart/bronze/clickstream/']
    )
    bronze_table_2 = PythonOperator(
        task_id='run_bronze_table_attributes',
        python_callable=process_bronze_attributes_table,
        op_args=['{{ ds }}', '/opt/airflow/scripts/datamart/bronze/attributes/']
    )
    bronze_table_3 = PythonOperator(
        task_id='run_bronze_table_financials',
        python_callable=process_bronze_financials_table,
        op_args=['{{ ds }}', '/opt/airflow/scripts/datamart/bronze/financials/']
    )
    skip_static_data = DummyOperator(task_id='skip_static_data')

    static_data_ready = DummyOperator(
        task_id='static_data_ready',
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS
    )
    monthly_data_ready = DummyOperator(
        task_id='monthly_data_ready',
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS
    )

    # Silver Layer Processing
    silver_table_1 = PythonOperator(
        task_id = 'run_silver_table_lms',
        python_callable = run_silver_table_lms,
        op_args = ['lms', '{{ ds }}', '/opt/airflow/scripts/datamart/bronze/', '/opt/airflow/scripts/datamart/silver/'],
    )
    silver_table_4 = PythonOperator(
        task_id = 'run_silver_table_clickstream',
        python_callable = run_silver_table_clickstream,
        op_args = ['clickstream', '{{ ds }}', '/opt/airflow/scripts/datamart/bronze/', '/opt/airflow/scripts/datamart/silver/'],
    )
    silver_table_2 = PythonOperator(
        task_id = 'run_silver_table_attributes',
        python_callable = run_silver_table_attributes,
        op_args = ['attributes', '{{ ds }}', '/opt/airflow/scripts/datamart/bronze/', '/opt/airflow/scripts/datamart/silver/'],
    )
    silver_table_3 = PythonOperator(
        task_id = 'run_silver_table_financials',
        python_callable = run_silver_table_financials,
        op_args = ['financials', '{{ ds }}', '/opt/airflow/scripts/datamart/bronze/', '/opt/airflow/scripts/datamart/silver/'],
    )

    # Gold Layer Processing
    gold_feature_and_label_store = PythonOperator(
        task_id = 'run_gold_table',
        python_callable = run_gold_table,
        op_args = ['{{ ds }}', '/opt/airflow/scripts/datamart/silver/', '/opt/airflow/scripts/datamart/gold/', 30, 6],
    )

    gold_table_completed = DummyOperator(task_id="gold_table_completed")
    preprocessing_complete = DummyOperator(task_id="preprocessing_complete")
    end = DummyOperator(task_id='end_pipeline')

    # Dependencies
    wait_for_previous_run >> start >> start_preprocessing >> check_static_data
    check_static_data >> bronze_table_2
    check_static_data >> bronze_table_3
    bronze_table_2 >> silver_table_2
    bronze_table_3 >> silver_table_3
    [silver_table_2, silver_table_3] >> static_data_ready
    check_static_data >> skip_static_data
    skip_static_data >> static_data_ready
    dep_check_source_data_bronze_1 >> bronze_table_1
    dep_check_source_data_bronze_4 >> bronze_table_4
    bronze_table_1 >> silver_table_1
    bronze_table_4 >> silver_table_4
    [silver_table_1, silver_table_4] >> monthly_data_ready
    [static_data_ready, monthly_data_ready] >> gold_feature_and_label_store
    gold_feature_and_label_store >> gold_table_completed >> preprocessing_complete >> end 