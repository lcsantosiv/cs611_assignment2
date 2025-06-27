from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.state import State
from datetime import datetime, timedelta
import os
import json
import sys
import pandas as pd
from pyspark.sql import SparkSession
from typing import List
from airflow.exceptions import AirflowSkipException
from dateutil.relativedelta import relativedelta
from airflow.utils.trigger_rule import TriggerRule

sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts/utils'))
from data_processing_bronze_table import process_bronze_loan_table, process_bronze_clickstream_table, process_bronze_attributes_table, process_bronze_financials_table
from data_processing_silver_table import process_silver_table
from data_processing_gold_table import process_gold_table
from dag_functions import (
    decide_pipeline_path,
    check_retraining_trigger,
    check_static_data_loaded,
    select_best_model_initial,
    register_model_initial,
    evaluate_production_model,
    train_lightgbm_monthly,
    train_catboost_monthly,
    select_best_model_monthly,
    register_model_monthly,
    run_model_inference,
    train_lightgbm_initial,
    train_catboost_initial
)
from silver_processing_retrofit import process_silver_table as process_silver_table_retrofit
from gold_processing_retrofit import process_gold_table as process_gold_table_retrofit

# === Spark Session Wrapper Functions ===
def run_bronze_table_lms(snapshot_date, output_dir):
    """Wrapper function to create Spark session and process bronze LMS table"""
    spark = SparkSession.builder.appName("BronzeTableLMS").getOrCreate()
    try:
        process_bronze_loan_table(snapshot_date, output_dir)
        return None
    finally:
        spark.stop()

def run_bronze_table_clickstream(snapshot_date, output_dir):
    """Wrapper function to create Spark session and process bronze clickstream table"""
    spark = SparkSession.builder.appName("BronzeTableClickstream").getOrCreate()
    try:
        process_bronze_clickstream_table(snapshot_date, output_dir)
        return None
    finally:
        spark.stop()

def run_bronze_table_attributes(snapshot_date, output_dir):
    """Wrapper function to create Spark session and process bronze attributes table"""
    spark = SparkSession.builder.appName("BronzeTableAttributes").getOrCreate()
    try:
        process_bronze_attributes_table(snapshot_date, output_dir)
        return None
    finally:
        spark.stop()

def run_bronze_table_financials(snapshot_date, output_dir):
    """Wrapper function to create Spark session and process bronze financials table"""
    spark = SparkSession.builder.appName("BronzeTableFinancials").getOrCreate()
    try:
        process_bronze_financials_table(snapshot_date, output_dir)
        return None
    finally:
        spark.stop()

def run_silver_table_lms(table_name, snapshot_date, bronze_dir, silver_dir):
    """Wrapper function to create Spark session and process silver LMS table"""
    spark = SparkSession.builder.appName("SilverTableLMS").getOrCreate()
    try:
        process_silver_table_retrofit(table_name, snapshot_date, bronze_dir, silver_dir, spark)
        return None
    finally:
        spark.stop()

def run_silver_table_clickstream(table_name, snapshot_date, bronze_dir, silver_dir):
    """Wrapper function to create Spark session and process silver clickstream table"""
    spark = SparkSession.builder.appName("SilverTableClickstream").getOrCreate()
    try:
        process_silver_table_retrofit(table_name, snapshot_date, bronze_dir, silver_dir, spark)
        return None
    finally:
        spark.stop()

def run_silver_table_attributes(table_name, snapshot_date, bronze_dir, silver_dir):
    """Wrapper function to create Spark session and process silver attributes table"""
    spark = SparkSession.builder.appName("SilverTableAttributes").getOrCreate()
    try:
        process_silver_table_retrofit(table_name, snapshot_date, bronze_dir, silver_dir, spark)
        return None
    finally:
        spark.stop()

def run_silver_table_financials(table_name, snapshot_date, bronze_dir, silver_dir):
    """Wrapper function to create Spark session and process silver financials table"""
    spark = SparkSession.builder.appName("SilverTableFinancials").getOrCreate()
    try:
        process_silver_table_retrofit(table_name, snapshot_date, bronze_dir, silver_dir, spark)
        return None
    finally:
        spark.stop()

def run_gold_table(snapshot_date, silver_dir, gold_dir, dpd, mob):
    """Wrapper function to create Spark session and process gold table"""
    spark = SparkSession.builder.appName("GoldTable").getOrCreate()
    try:
        process_gold_table_retrofit(snapshot_date, silver_dir, gold_dir, spark, dpd, mob)
        return None
    finally:
        spark.stop()

class SafeExternalTaskSensor(ExternalTaskSensor):
    def poke(self, context):
        prev_execution_date = context['execution_date'] - relativedelta(months=1)
        dag = context['dag']
        dag_start_date = getattr(dag, 'start_date', None) or (dag.default_args.get('start_date') if hasattr(dag, 'default_args') else None)
        if dag_start_date is None:
            # If we can't determine the start date, always succeed
            return True
        if prev_execution_date < dag_start_date:
            # Mark as success explicitly
            context['ti'].set_state(State.SUCCESS)
            return True
        return super().poke(context)

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': True,  # Critical: Each run waits for previous month's run to complete
    'start_date': datetime(2023, 1, 1),
    'end_date': datetime(2024, 12, 31),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    'data_ml_pipeline',
    default_args=default_args,
    description='End-to-end ML lifecycle pipeline for credit scoring',
    schedule_interval='0 6 1 * *',  # Monthly on the 1st at 6 AM
    catchup=True,
    tags=['data-preprocessing', 'credit-scoring', 'ml-lifecycle']
) as dag:
    """
    ML Lifecycle Pipeline with Monthly Dependencies
    
    CRITICAL: This DAG uses an ExternalTaskSensor to ensure proper monthly dependencies.
    - Each month's run waits for the previous month's 'preprocessing_complete' task to complete.
    - This prevents race conditions and ensures data consistency.
    - Currently in preprocessing-only mode for testing data pipeline.
    - EXTERNAL TASK SENSOR TEMPORARILY DISABLED for independent preprocessing testing.
    """

    wait_for_previous_run = SafeExternalTaskSensor(
        task_id='wait_for_previous_run',
        external_dag_id='data_ml_pipeline',
        external_task_id='end_pipeline',  # Wait for end_pipeline completion
        execution_delta=relativedelta(months=1),  # Wait for previous month
        allowed_states=['success'],
        mode='poke',
        timeout=60 * 60 * 3,
        poke_interval=5,  # Check every 5 seconds for faster backfill
    )

    # === Start of Pipeline ===
    start = DummyOperator(task_id='start_pipeline')
    start_preprocessing = DummyOperator(task_id='start_preprocessing')

    # --- Static Data Loading Check ---
    check_static_data = BranchPythonOperator(
        task_id='check_static_data',
        python_callable=check_static_data_loaded
    )

    # --- Data Source Dependencies ---
    dep_check_source_data_bronze_1 = FileSensor(
        task_id='dep_check_source_lms',
        filepath='/opt/airflow/scripts/data/lms_loan_daily.csv',
        poke_interval=10,
        timeout=600,
        mode='poke'
    )
    dep_check_source_data_bronze_2 = FileSensor(
        task_id='dep_check_source_attributes',
        filepath='/opt/airflow/scripts/data/features_attributes.csv',
        poke_interval=10,
        timeout=600,
        mode='poke'
    )
    dep_check_source_data_bronze_3 = FileSensor(
        task_id='dep_check_source_financials',
        filepath='/opt/airflow/scripts/data/features_financials.csv',
        poke_interval=10,
        timeout=600,
        mode='poke'
    )
    dep_check_source_data_bronze_4 = FileSensor(
        task_id='dep_check_source_clickstream',
        filepath='/opt/airflow/scripts/data/feature_clickstream.csv',
        poke_interval=10,
        timeout=600,
        mode='poke'
    )

    # --- Bronze Layer Processing ---
    # Monthly data (always process)
    bronze_table_1 = PythonOperator(
        task_id='run_bronze_table_lms',
        python_callable=run_bronze_table_lms,
        op_args=['{{ ds }}', '/opt/airflow/scripts/datamart/bronze/lms/']
    )
    bronze_table_4 = PythonOperator(
        task_id='run_bronze_table_clickstream',
        python_callable=run_bronze_table_clickstream,
        op_args=['{{ ds }}', '/opt/airflow/scripts/datamart/bronze/clickstream/']
    )

    # Static data (only process on 2023-01-01)
    bronze_table_2 = PythonOperator(
        task_id='run_bronze_table_attributes',
        python_callable=run_bronze_table_attributes,
        op_args=['{{ ds }}', '/opt/airflow/scripts/datamart/bronze/attributes/']
    )
    bronze_table_3 = PythonOperator(
        task_id='run_bronze_table_financials',
        python_callable=run_bronze_table_financials,
        op_args=['{{ ds }}', '/opt/airflow/scripts/datamart/bronze/financials/']
    )

    # Skip static data loading for non-initial months
    skip_static_data = DummyOperator(task_id='skip_static_data')

    # Dummy operators to ensure gold table dependencies are met
    static_data_ready = DummyOperator(
        task_id='static_data_ready',
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS
    )
    monthly_data_ready = DummyOperator(
        task_id='monthly_data_ready',
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS
    )

    # --- Silver Layer Processing ---
    # Monthly data (always process)
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

    # Static data (only process on 2023-01-01)
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

    # --- Gold Layer Processing ---
    gold_feature_and_label_store = PythonOperator(
        task_id = 'run_gold_table',
        python_callable = run_gold_table,
        op_args = ['{{ ds }}', '/opt/airflow/scripts/datamart/silver/', '/opt/airflow/scripts/datamart/gold/', 30, 6],
    )

    gold_table_completed = DummyOperator(task_id="gold_table_completed")
    preprocessing_complete = DummyOperator(task_id="preprocessing_complete")

    # === Dependencies ===
    wait_for_previous_run >> start >> start_preprocessing >> check_static_data
    start_preprocessing >> dep_check_source_data_bronze_1
    start_preprocessing >> dep_check_source_data_bronze_4
    
    # Static data branch
    check_static_data >> dep_check_source_data_bronze_2
    check_static_data >> dep_check_source_data_bronze_3
    check_static_data >> skip_static_data
    dep_check_source_data_bronze_2 >> bronze_table_2
    dep_check_source_data_bronze_3 >> bronze_table_3
    bronze_table_2 >> silver_table_2
    bronze_table_3 >> silver_table_3
    [silver_table_2, silver_table_3] >> static_data_ready
    skip_static_data >> static_data_ready
    
    # Monthly data (always process)
    dep_check_source_data_bronze_1 >> bronze_table_1
    dep_check_source_data_bronze_4 >> bronze_table_4
    bronze_table_1 >> silver_table_1
    bronze_table_4 >> silver_table_4
    [silver_table_1, silver_table_4] >> monthly_data_ready
    
    # Gold layer depends on both static and monthly data being ready
    [static_data_ready, monthly_data_ready] >> gold_feature_and_label_store

    end = DummyOperator(task_id='end_pipeline', trigger_rule='one_success')
    
    gold_feature_and_label_store >> gold_table_completed >> preprocessing_complete

    # === ML PIPELINE - UNCOMMENTED ===
    # Gate 1: Decide on the main pipeline path

    # Uncomment ML pipeline tasks and dependencies

    decide_pipeline_path_task = BranchPythonOperator(
        task_id='decide_pipeline_path',
        python_callable=decide_pipeline_path,
    )

    # Path 1: Skip (for historical runs before 2023)
    skip_run = DummyOperator(task_id='skip_run')

    # Path 2: One-Time Initial Training Flow
    run_initial_training_flow = DummyOperator(task_id='run_initial_training_flow')
    train_lightgbm_initial = PythonOperator(
        task_id='train_lightgbm_initial',
        python_callable=train_lightgbm_initial,
    )
    train_catboost_initial = PythonOperator(
        task_id='train_catboost_initial',
        python_callable=train_catboost_initial,
    )
    select_best_model_initial_task = PythonOperator(
        task_id='select_best_model_initial',
        python_callable=select_best_model_initial,
    )
    register_model_initial_task = PythonOperator(
        task_id='register_model_initial',
        python_callable=register_model_initial,
    )

    # Path 3: Standard Monthly Lifecycle Flow
    run_monthly_lifecycle_flow = DummyOperator(task_id='run_monthly_lifecycle_flow')
    evaluate_production_model_task = PythonOperator(
        task_id='evaluate_production_model',
        python_callable=evaluate_production_model,
    )
    check_retraining_trigger_task = BranchPythonOperator(
        task_id='check_retraining_trigger',
        python_callable=check_retraining_trigger,
    )
    skip_retraining = DummyOperator(task_id='skip_retraining')
    trigger_retraining = DummyOperator(task_id='trigger_retraining')
    train_lightgbm_monthly_task = PythonOperator(
        task_id='train_lightgbm_monthly',
        python_callable=train_lightgbm_monthly,
    )
    train_catboost_monthly_task = PythonOperator(
        task_id='train_catboost_monthly',
        python_callable=train_catboost_monthly,
    )
    select_best_model_monthly_task = PythonOperator(
        task_id='select_best_model_monthly',
        python_callable=select_best_model_monthly,
    )
    register_model_monthly_task = PythonOperator(
        task_id='register_model_monthly',
        python_callable=register_model_monthly,
    )
    run_model_inference_task = PythonOperator(
        task_id='run_model_inference',
        python_callable=run_model_inference,
        trigger_rule='one_success',
    )

    # === Define ML DAG Dependencies ===
    preprocessing_complete >> decide_pipeline_path_task

    decide_pipeline_path_task >> skip_run >> end

    # Path 2: Initial Training Logic (staggered)
    decide_pipeline_path_task >> run_initial_training_flow
    run_initial_training_flow >> train_lightgbm_initial >> train_catboost_initial >> select_best_model_initial_task >> register_model_initial_task >> end

    # Path 3: Monthly Lifecycle Logic (staggered)
    decide_pipeline_path_task >> run_monthly_lifecycle_flow
    run_monthly_lifecycle_flow >> evaluate_production_model_task >> check_retraining_trigger_task

    # Branching after evaluation
    check_retraining_trigger_task >> skip_retraining >> run_model_inference_task
    check_retraining_trigger_task >> trigger_retraining

    # Retraining sub-path (staggered)
    trigger_retraining >> train_lightgbm_monthly_task >> train_catboost_monthly_task >> select_best_model_monthly_task >> register_model_monthly_task
    register_model_monthly_task >> run_model_inference_task

    # Inference is the final step for the monthly path
    run_model_inference_task >> end
