"""
DAG Functions for ML Lifecycle Pipeline

This module contains all the Python functions used by the ML lifecycle pipeline DAG.
Separating these functions from the main DAG file improves maintainability and readability.
"""
import sys
import os
import json
import re
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import mlflow
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
sys.path.append('/opt/airflow/scripts/utils')
from model_inference_utils import (
    load_model_from_mlflow,
    load_weekly_data,
    prepare_features,
    evaluate_weekly_performance,
    save_metrics_to_postgres,
    calculate_comprehensive_metrics,
    save_model_metrics_to_postgres,
    run_model_inference_with_saving,
    load_deployed_model_from_registry,
    list_models_in_registry
)

PG_CONFIG = {
    'host': 'postgres',
    'port': 5432,
    'database': 'airflow',
    'user': 'airflow',
    'password': 'airflow'
}

FEATURE_STORE_PATH = "/opt/airflow/scripts/datamart/gold/feature_store"
LABEL_STORE_PATH = "/opt/airflow/scripts/datamart/gold/label_store"
TARGET_COLUMN = 'label'
MODEL_TYPE = 'LightGBM'  # or 'CatBoost', parametrize as needed
MODEL_NAME = 'credit_scoring_model'  # or parametrize as needed

def decide_pipeline_path(**context):
    """
    Directs the pipeline based on the execution date.
    - Before 2023: Skip all model-related tasks.
    - On June 1, 2024: Trigger the dedicated initial model training flow.
    - After June 1, 2024: Run the standard monthly lifecycle.
    """
    logical_date = context["logical_date"]
    initial_training_date = datetime(2024, 6, 1)

    if logical_date.replace(tzinfo=None) < initial_training_date:
        return 'skip_run'
    elif logical_date.date() == initial_training_date.date():
        return 'run_initial_training_flow'
    else:
        return 'run_monthly_lifecycle_flow'


def check_retraining_trigger(**context):
    """
    Decide whether to trigger retraining based on specific dates.
    Triggers retraining on Jun 1, Sep 1, and Dec 1, 2024.
    """
    logical_date = context['logical_date']
    
    # Define retraining dates (as date objects, not datetime)
    retraining_dates = [
        datetime(2024, 6, 1).date(),
        datetime(2024, 9, 1).date(), 
        datetime(2024, 12, 1).date()
    ]
    
    # Check if current date is a retraining date
    current_date = logical_date.replace(tzinfo=None).date()
    
    # Debug logging
    print(f"[Trigger] Current date: {current_date}")
    print(f"[Trigger] Current date type: {type(current_date)}")
    print(f"[Trigger] Retraining dates: {retraining_dates}")
    print(f"[Trigger] Date comparison: {current_date in retraining_dates}")
    
    if current_date in retraining_dates:
        print(f"[Trigger] Retraining triggered on {current_date.strftime('%Y-%m-%d')}")
        return 'trigger_retraining'
    else:
        print(f"[Trigger] No retraining needed on {current_date.strftime('%Y-%m-%d')}")
        return 'skip_retraining'


def extract_mlflow_run_id_from_logs(task_id: str = None, **context):
    """
    Extract MLflow run ID from the training script logs.
    This function parses the logs to find the MLflow run ID.
    
    Args:
        task_id: Specific task ID to extract from (if None, uses current task)
        context: Airflow context
    """
    # Get the task instance to access logs
    task_instance = context['task_instance']
    
    if task_id:
        # Get logs from a specific task
        log_content = task_instance.xcom_pull(task_ids=task_id, key='return_value')
        if not log_content:
            # Try to get logs directly
            try:
                log_content = task_instance.log.read(task_id=task_id)
            except:
                log_content = ""
    else:
        # Get the log content for the current task
        log_content = task_instance.log.read()
    
    # Look for MLflow run ID in the logs
    # The pattern might be something like "run_id: abc123" or similar
    run_id_pattern = r'run_id[:\s]+([a-f0-9]+)'
    match = re.search(run_id_pattern, log_content, re.IGNORECASE)
    
    if match:
        run_id = match.group(1)
        print(f"Extracted MLflow run ID from {task_id or 'current task'}: {run_id}")
        return run_id
    else:
        # If we can't find the run ID in logs, we'll need to query MLflow directly
        print(f"Could not extract run ID from {task_id or 'current task'} logs, will query MLflow directly")
        return None


def extract_metrics_from_logs(task_id: str = None, **context):
    """
    Extract performance metrics from the training script logs.
    
    Args:
        task_id: Specific task ID to extract from (if None, uses current task)
        context: Airflow context
    """
    task_instance = context['task_instance']
    
    if task_id:
        # Get logs from a specific task
        log_content = task_instance.xcom_pull(task_ids=task_id, key='return_value')
        if not log_content:
            # Try to get logs directly
            try:
                log_content = task_instance.log.read(task_id=task_id)
            except:
                log_content = ""
    else:
        log_content = task_instance.log.read()
    
    # Look for Macro F1 score in logs
    f1_pattern = r'Macro F1 Score[:\s]+([0-9.]+)'
    match = re.search(f1_pattern, log_content, re.IGNORECASE)
    
    if match:
        macro_f1 = float(match.group(1))
        print(f"Extracted Macro F1 Score from {task_id or 'current task'}: {macro_f1}")
        return macro_f1
    else:
        print(f"Could not extract Macro F1 score from {task_id or 'current task'} logs")
        return None


def query_mlflow_for_run_info(model_type: str = None, **context) -> Tuple[Optional[str], Optional[float]]:
    """
    Query MLflow to get run information and metrics.
    This is a fallback if we can't extract from logs.
    
    Args:
        model_type: Optional model type to filter results
        context: Airflow context
    """
    # Set up MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    
    # Get the current execution date
    logical_date = context['logical_date']
    
    # Query recent runs from the experiment
    experiment_name = "assignment2"  # Updated from "test" to "assignment2"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found")
        return None, None
    
    # Get runs from the last hour (should include our recent training run)
    start_time = int((logical_date - timedelta(hours=1)).timestamp() * 1000)  # ms since epoch
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"start_time >= {start_time}",
        max_results=10,  # Increased to catch both models
        order_by=["start_time DESC"]
    )
    
    if runs.empty:
        print("No recent runs found")
        return None, None
    
    # If model_type is specified, look for runs with that model type
    if model_type:
        # Look for runs with model type in the run name or parameters
        for _, run in runs.iterrows():
            run_name = run.get('tags.mlflow.runName', '')
            if model_type.lower() in run_name.lower():
                run_id = run['run_id']
                macro_f1 = run.get('metrics.macro_f1_score', None)
                print(f"Found {model_type} run: {run_id}, Macro F1: {macro_f1}")
                return run_id, macro_f1
    
    # If no specific model type or not found, return the most recent run
    latest_run = runs.iloc[0]
    run_id = latest_run['run_id']
    macro_f1 = latest_run.get('metrics.macro_f1_score', None)
    
    print(f"Found recent run: {run_id}, Macro F1: {macro_f1}")
    return run_id, macro_f1


def select_best_model_initial(**context):
    """
    Extract results from both training tasks and select the best model.
    This combines result extraction and model selection for efficiency.
    """
    print("Extracting results from both training tasks and selecting best model...")
    
    # Extract LightGBM results
    print("Extracting LightGBM results...")
    lightgbm_run_id = extract_mlflow_run_id_from_logs(task_id='train_lightgbm_initial', **context)
    lightgbm_f1 = extract_metrics_from_logs(task_id='train_lightgbm_initial', **context)
    
    # If extraction failed, query MLflow
    if lightgbm_run_id is None or lightgbm_f1 is None:
        print("LightGBM log extraction failed, querying MLflow directly...")
        lightgbm_run_id, lightgbm_f1 = query_mlflow_for_run_info(model_type='lightgbm', **context)
    
    # Extract CatBoost results
    print("Extracting CatBoost results...")
    catboost_run_id = extract_mlflow_run_id_from_logs(task_id='train_catboost_initial', **context)
    catboost_f1 = extract_metrics_from_logs(task_id='train_catboost_initial', **context)
    
    # If extraction failed, query MLflow
    if catboost_run_id is None or catboost_f1 is None:
        print("CatBoost log extraction failed, querying MLflow directly...")
        catboost_run_id, catboost_f1 = query_mlflow_for_run_info(model_type='catboost', **context)
    
    # Handle missing results
    if lightgbm_run_id is None:
        print("Warning: Could not get LightGBM run ID")
        lightgbm_run_id = "unknown"
        lightgbm_f1 = 0.0
    
    if catboost_run_id is None:
        print("Warning: Could not get CatBoost run ID")
        catboost_run_id = "unknown"
        catboost_f1 = 0.0
    
    if lightgbm_f1 is None:
        print("Warning: Could not get LightGBM Macro F1 score")
        lightgbm_f1 = 0.0
    
    if catboost_f1 is None:
        print("Warning: Could not get CatBoost Macro F1 score")
        catboost_f1 = 0.0
    
    # Push individual results to XComs for debugging/auditing
    context['task_instance'].xcom_push(key='lightgbm_run_id', value=lightgbm_run_id)
    context['task_instance'].xcom_push(key='catboost_run_id', value=catboost_run_id)
    context['task_instance'].xcom_push(key='lightgbm_macro_f1', value=lightgbm_f1)
    context['task_instance'].xcom_push(key='catboost_macro_f1', value=catboost_f1)
    
    print(f"LightGBM Run ID: {lightgbm_run_id}, Macro F1: {lightgbm_f1:.4f}")
    print(f"CatBoost Run ID: {catboost_run_id}, Macro F1: {catboost_f1:.4f}")
    
    # Select best model
    if lightgbm_f1 > catboost_f1:
        best_run_id = lightgbm_run_id
        best_model_type = "LightGBM"
        best_f1 = lightgbm_f1
    else:
        best_run_id = catboost_run_id
        best_model_type = "CatBoost"
        best_f1 = catboost_f1
    
    print(f"Selected {best_model_type} as best model with Macro F1: {best_f1:.4f}")
    
    # Push best model info to XComs
    context['task_instance'].xcom_push(key='best_run_id', value=best_run_id)
    context['task_instance'].xcom_push(key='best_model_type', value=best_model_type)
    context['task_instance'].xcom_push(key='best_macro_f1', value=best_f1)
    
    return f"Best model selected: {best_model_type} (Run ID: {best_run_id}, Macro F1: {best_f1:.4f})"


def register_model_initial(**context):
    """
    Register the best model to MLflow Model Registry and promote to Production.
    """
    # Get best model info from XComs
    best_run_id = context['task_instance'].xcom_pull(key='best_run_id')
    best_model_type = context['task_instance'].xcom_pull(key='best_model_type')
    best_f1 = context['task_instance'].xcom_pull(key='best_macro_f1')
    
    print(f"Registering {best_model_type} model (Run ID: {best_run_id}) to MLflow Registry")
    
    # Set up MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    
    # Get the model URI from the run
    model_uri = f"runs:/{best_run_id}/model"
    
    # Register the model
    model_name = "credit_scoring_model"
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )
    
    # Transition to Production
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Production"
    )
    
    print(f"Model registered successfully: {model_name} v{model_version.version}")
    print(f"Model promoted to Production stage")
    
    # Update retraining tracker
    retraining_tracker_path = "/opt/airflow/logs/last_retraining_date.json"
    tracker_data = {
        'last_retraining_date': context['logical_date'].strftime('%Y-%m-%d'),
        'model_name': model_name,
        'model_version': model_version.version,
        'model_type': best_model_type,
        'macro_f1_score': best_f1,
        'registration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    os.makedirs(os.path.dirname(retraining_tracker_path), exist_ok=True)
    with open(retraining_tracker_path, 'w') as f:
        json.dump(tracker_data, f, indent=2)
    
    print(f"Retraining tracker updated: {retraining_tracker_path}")
    
    return f"Model registered and promoted to Production: {model_name} v{model_version.version}"


def evaluate_production_model(**context):
    """
    Pull the latest metric from Postgres for the previous week to evaluate model performance.
    If it's the week after ANY retraining event (initial or subsequent), just succeed.
    Returns a standard Python dict with metrics for XCom passing.
    """
    logical_date = context['logical_date']
    current_week_date = logical_date.strftime('%Y_%m_%d')
    # Get previous week's date for evaluation
    prev_week_date = (logical_date - timedelta(weeks=1)).strftime('%Y_%m_%d')
    print(f"[Eval] Evaluating production model performance for previous week: {prev_week_date}")

    # Check if this is the week after ANY retraining event
    retraining_tracker_path = "/opt/airflow/logs/last_retraining_date.json"
    if os.path.exists(retraining_tracker_path):
        with open(retraining_tracker_path, 'r') as f:
            tracker_data = json.load(f)
            last_retraining_date = tracker_data.get('last_retraining_date')
            if last_retraining_date:
                # Calculate the week after the last retraining
                last_retraining_datetime = datetime.strptime(last_retraining_date, '%Y-%m-%d')
                week_after_retraining = (last_retraining_datetime + timedelta(weeks=1)).strftime('%Y_%m_%d')
                if current_week_date == week_after_retraining:
                    print(f"[Eval] This is the week after the last retraining event ({last_retraining_date}). Skipping evaluation.")
                    return {"macro_f1": None, "message": "Week after retraining event, skipping evaluation."}

    # Query previous week's metrics from Postgres
    try:
        conn = psycopg2.connect(**PG_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        query = """
            SELECT * FROM model_performance_metrics
            WHERE week_date = %s
            ORDER BY evaluation_date DESC
            LIMIT 1
        """
        cursor.execute(query, (prev_week_date,))
        result = cursor.fetchone()
        if not result:
            print(f"[Eval] No metrics found for previous week {prev_week_date}")
            return {"macro_f1": None, "message": f"No metrics for previous week {prev_week_date}"}
        print(f"[Eval] Found metrics for previous week {prev_week_date}:")
        print(f"  - Model: {result['model_name']}")
        print(f"  - Macro F1: {result['macro_f1']:.4f}")
        print(f"  - Accuracy: {result['accuracy']:.4f}")
        print(f"  - Total Samples: {result['total_samples']}")
        # Return as standard dict for XCom
        return {
            "macro_f1": result["macro_f1"],
            "model_name": result["model_name"],
            "accuracy": result["accuracy"],
            "total_samples": result["total_samples"],
            "message": "success"
        }
    except Exception as e:
        print(f"[Eval] Error querying Postgres: {e}")
        return {"macro_f1": None, "message": f"Error querying Postgres: {e}"}
    finally:
        if conn:
            conn.close()


def prepare_training_data_monthly(**context):
    """
    Load the rolling 12-month training data for monthly retraining.
    """
    # TODO: Implement monthly training data preparation
    # This would load data from the last 12 months for retraining
    print("Monthly training data preparation - to be implemented")
    return "Monthly training data prepared"


def check_static_data_loaded(**context):
    """
    Check if static data (attributes and financials) has been loaded.
    Static data should only be loaded once on 2023-01-01.
    """
    logical_date = context['logical_date']
    
    # Static data should only be loaded on 2023-01-01
    static_data_date = datetime(2023, 1, 1)
    
    if logical_date.date() == static_data_date.date():
        print(f"[Static Data] Loading static data on {logical_date.date()}")
        # Return the actual task IDs for static data processing
        return ['dep_check_source_attributes', 'dep_check_source_financials']
    else:
        print(f"[Static Data] Skipping static data load on {logical_date.date()} (only loads on 2023-01-01)")
        return ['skip_static_data']


def check_data_availability(**context):
    """
    Check if all required data is available for the current month.
    For static data (attributes, financials), check if static files exist.
    For monthly data (clickstream, lms), check if monthly files exist.
    """
    logical_date = context['logical_date']
    snapshot_date_str = logical_date.strftime('%Y-%m-%d')
    
    # Check static data files (should exist for all months after 2023-01-01)
    static_data_date = datetime(2023, 1, 1)
    if logical_date.date() > static_data_date.date():
        static_files = [
            "/opt/airflow/scripts/datamart/bronze/attributes/bronze_attr_static.csv",
            "/opt/airflow/scripts/datamart/bronze/financials/bronze_fin_static.csv",
            "/opt/airflow/scripts/datamart/silver/attributes/silver_attributes_static.parquet",
            "/opt/airflow/scripts/datamart/silver/financials/silver_financials_static.parquet"
        ]
        
        missing_static_files = [f for f in static_files if not os.path.exists(f)]
        if missing_static_files:
            print(f"[Data Check] Missing static data files: {missing_static_files}")
            return 'data_unavailable'
    
    # Check monthly data files
    monthly_files = [
        f"/opt/airflow/scripts/datamart/bronze/clickstream/bronze_clks_mthly_{snapshot_date_str.replace('-','_')}.csv",
        f"/opt/airflow/scripts/datamart/bronze/lms/bronze_loan_daily_{snapshot_date_str.replace('-','_')}.csv",
        f"/opt/airflow/scripts/datamart/silver/clickstream/silver_clickstream_mthly_{snapshot_date_str.replace('-','_')}.parquet",
        f"/opt/airflow/scripts/datamart/silver/lms/silver_lms_mthly_{snapshot_date_str.replace('-','_')}.parquet"
    ]
    
    missing_monthly_files = [f for f in monthly_files if not os.path.exists(f)]
    if missing_monthly_files:
        print(f"[Data Check] Missing monthly data files: {missing_monthly_files}")
        return 'data_unavailable'
    
    print(f"[Data Check] All required data files available for {snapshot_date_str}")
    return 'data_available'


def run_model_inference(**context):
    """
    Run model inference on the current month's data with comprehensive metrics and PostgreSQL saving.
    """
    from pyspark.sql import SparkSession
    from model_inference_utils import load_weekly_data, prepare_features, evaluate_weekly_performance, load_deployed_model_from_registry, list_models_in_registry
    
    # Get the current execution date
    logical_date = context['logical_date']
    month_date = logical_date.strftime('%Y_%m_%d')
    
    print(f"[Inference] Running model inference for month: {month_date}")
    
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("ModelInference") \
        .master("local[*]") \
        .getOrCreate()
    
    try:
        # List available models in registry for debugging
        list_models_in_registry()
        
        # Load the currently deployed model from MLflow's model registry
        print(f"[Inference] Loading deployed model from registry...")
        try:
            # Try loading version 1 specifically
            model_info = load_deployed_model_from_registry(MODEL_NAME, version="1")
        except Exception as e:
            print(f"[Inference] Failed to load version 1, trying latest: {e}")
            # Fallback to latest version
            model_info = load_deployed_model_from_registry(MODEL_NAME)
        print(f"[Inference] Loaded model version {model_info['model_version']} (run_id: {model_info['run_id']})")
        
        # Load monthly data
        monthly_data = load_weekly_data(spark, month_date, FEATURE_STORE_PATH, LABEL_STORE_PATH)
        if monthly_data is None:
            print(f"[Inference] No data available for month: {month_date}")
            return "Model inference skipped - no data available"
        
        # Prepare features for binary classification
        feature_names = model_info["feature_names"]
        X, y_raw, _ = prepare_features(monthly_data, feature_names, TARGET_COLUMN)
        
        # For binary classification, no label encoding needed
        y_encoded = y_raw  # Binary labels are already 0/1
        
        # Get model (no grade mapping needed for binary classification)
        model = model_info["model"]
        
        # Get customer IDs and snapshot dates for inference saving
        customer_ids = monthly_data['loan_id'].tolist()
        snapshot_dates = monthly_data['snapshot_date'].tolist()
        
        # Run inference with saving to PostgreSQL (binary classification)
        inference_results = run_model_inference_with_saving(
            model=model,
            X=X,
            customer_ids=customer_ids,
            snapshot_dates=snapshot_dates,
            model_name=MODEL_NAME,
            month_date=month_date,
            model_version=model_info['model_version'],
            mlflow_run_id=model_info['run_id'],
            pg_config=PG_CONFIG
        )
        
        # Calculate comprehensive metrics
        y_pred = model.predict(X)
        
        # Handle probability prediction
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X)
        else:
            # Try to get probabilities from the underlying model
            try:
                # For CatBoost pyfunc models, we can access the underlying model
                if hasattr(model, '_model_impl') and hasattr(model._model_impl, 'cb_model'):
                    # Access the underlying CatBoost model
                    cb_model = model._model_impl.cb_model
                    y_pred_proba = cb_model.predict_proba(X)
                else:
                    # If no predict_proba, use predictions as probabilities
                    y_pred_proba = None
            except Exception as e:
                print(f"[Inference] Could not get probabilities for AUC: {e}")
                y_pred_proba = None
        
        comprehensive_metrics = calculate_comprehensive_metrics(
            y_true=y_encoded,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba
        )
        
        # Save comprehensive metrics to model_metrics table
        save_model_metrics_to_postgres(
            metrics=comprehensive_metrics,
            month_date=month_date,
            run_id=model_info['run_id'],
            model_name=MODEL_NAME,
            pg_config=PG_CONFIG
        )
        
        print(f"[Inference] Monthly inference completed successfully")
        print(f"[Inference] Accuracy: {comprehensive_metrics['accuracy']:.4f}")
        print(f"[Inference] Precision: {comprehensive_metrics['precision']:.4f}")
        print(f"[Inference] Recall: {comprehensive_metrics['recall']:.4f}")
        if comprehensive_metrics['auc'] is not None:
            print(f"[Inference] AUC: {comprehensive_metrics['auc']:.4f}")
        else:
            print(f"[Inference] AUC: Not available (no probabilities)")
        print(f"[Inference] Macro F1: {comprehensive_metrics['macro_f1']:.4f}")
        print(f"[Inference] Total samples: {comprehensive_metrics['total_samples']}")
        
        return f"Monthly inference completed - AUC: {comprehensive_metrics['auc']:.4f}, F1: {comprehensive_metrics['macro_f1']:.4f}"
        
    except Exception as e:
        print(f"[Inference] Error during monthly inference: {e}")
        raise
    finally:
        spark.stop()


def verify_grade_mapping_in_mlflow(run_id, client=None):
    """
    Utility to print and verify the grade mapping stored in MLflow for a given run_id.
    """
    import mlflow
    if client is None:
        client = mlflow.tracking.MlflowClient()
    try:
        run = client.get_run(run_id)
        grade_mapping_param = run.data.params.get('grade_mapping', '')
        if grade_mapping_param:
            grade_mapping = json.loads(grade_mapping_param)
            print(f"[VERIFY] grade_mapping param in MLflow for run {run_id}: {grade_mapping}")
        else:
            print(f"[VERIFY] No grade_mapping param found in MLflow for run {run_id}")
        # Optionally, check the artifact as well
        try:
            grade_mapping_artifact = mlflow.artifacts.load_dict(f"runs:/{run_id}/grade_mapping.json")
            print(f"[VERIFY] grade_mapping artifact in MLflow for run {run_id}: {grade_mapping_artifact}")
        except Exception as e:
            print(f"[VERIFY] Could not load grade_mapping artifact: {e}")
    except Exception as e:
        print(f"[VERIFY] Error accessing MLflow run {run_id}: {e}")


def decode_predictions(y_pred, grade_mapping):
    """
    Given a list/array of predicted indices and a grade_mapping (grade->idx),
    return the decoded grade labels.
    """
    reverse_grade_mapping = {v: k for k, v in grade_mapping.items()}
    return [reverse_grade_mapping.get(idx, "Unknown") for idx in y_pred]


def get_training_data_window(current_month_date):
    """
    Calculate the 12-month training data window ending on the previous month.
    This ensures we don't leak current month's data into training.
    """
    from datetime import datetime, timedelta
    from dateutil.relativedelta import relativedelta
    
    # Parse current month date
    current_date = datetime.strptime(current_month_date, '%Y_%m_%d')
    
    # Training window ends on previous month
    training_end_date = current_date - relativedelta(months=1)
    
    # Training window starts 12 months before the end date
    training_start_date = training_end_date - relativedelta(months=11)  # 12 months total
    
    # Generate list of all month dates in the training window
    training_months = []
    current_month = training_start_date
    while current_month <= training_end_date:
        training_months.append(current_month.strftime('%Y_%m_%d'))
        current_month += relativedelta(months=1)
    
    print(f"[Training] Data window: {len(training_months)} months from {training_start_date.strftime('%Y_%m_%d')} to {training_end_date.strftime('%Y_%m_%d')}")
    return training_months


def train_lightgbm_monthly(**context):
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    import mlflow
    import os
    import pandas as pd
    from pyspark.sql import SparkSession
    from datetime import datetime
    from dateutil.relativedelta import relativedelta

    # 1. Get rolling 12-month window ending previous month
    month_date = context['logical_date'].strftime('%Y_%m_%d')
    training_months = get_training_data_window(month_date)
    feature_dir = "/opt/airflow/scripts/datamart/gold/feature_store"
    label_dir = "/opt/airflow/scripts/datamart/gold/label_store"

    def get_files_for_months(folder, prefix, months):
        files = []
        for m in months:
            fname = f"{prefix}{m}.parquet"
            fpath = os.path.join(folder, fname)
            if os.path.exists(fpath):
                files.append(fpath)
        return sorted(files)

    feature_files = get_files_for_months(feature_dir, "gold_feature_store_", training_months)
    label_files = get_files_for_months(label_dir, "gold_label_store_", training_months)
    spark = SparkSession.builder.appName("LightGBMMonthlyTraining").getOrCreate()
    df_features = spark.read.parquet(*feature_files)
    df_labels = spark.read.parquet(*label_files)
    df = df_features.join(
        df_labels.select("loan_id", "label", "snapshot_date"),
        on=["loan_id"],
        how="inner"
    )
    df_pd = df.toPandas()
    spark.stop()

    # 2. Prepare features and label
    label_col = 'label'
    exclude_cols = ['loan_id', 'snapshot_date', label_col]
    feature_cols = [c for c in df_pd.columns if c not in exclude_cols]
    X = df_pd[feature_cols]
    y = df_pd[label_col]

    # 3. Train/test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train LightGBM
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'max_depth': 4,
        'learning_rate': 0.1,
        'n_estimators': 150,
        'num_leaves': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbose': -1
    }
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        num_boost_round=100
    )

    # 5. Evaluate
    y_pred = model.predict(X_val)
    auc = roc_auc_score(y_val, y_pred)
    print(f"[Monthly Training] LightGBM Validation AUC: {auc:.4f}")

    # 6. Log to MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("assignment2")
    with mlflow.start_run(run_name="lightgbm_monthly"):
        mlflow.log_params(params)
        mlflow.log_metric("auc", auc)
        mlflow.lightgbm.log_model(model, "model")
        mlflow.log_param("feature_names", feature_cols)
    print("LightGBM monthly model training and logging complete.")


def train_catboost_monthly(**context):
    from catboost import CatBoostClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    import mlflow
    import os
    import pandas as pd
    from pyspark.sql import SparkSession
    from datetime import datetime
    from dateutil.relativedelta import relativedelta

    # 1. Get rolling 12-month window ending previous month
    month_date = context['logical_date'].strftime('%Y_%m_%d')
    training_months = get_training_data_window(month_date)
    feature_dir = "/opt/airflow/scripts/datamart/gold/feature_store"
    label_dir = "/opt/airflow/scripts/datamart/gold/label_store"

    def get_files_for_months(folder, prefix, months):
        files = []
        for m in months:
            fname = f"{prefix}{m}.parquet"
            fpath = os.path.join(folder, fname)
            if os.path.exists(fpath):
                files.append(fpath)
        return sorted(files)

    feature_files = get_files_for_months(feature_dir, "gold_feature_store_", training_months)
    label_files = get_files_for_months(label_dir, "gold_label_store_", training_months)
    spark = SparkSession.builder.appName("CatBoostMonthlyTraining").getOrCreate()
    df_features = spark.read.parquet(*feature_files)
    df_labels = spark.read.parquet(*label_files)
    df = df_features.join(
        df_labels.select("loan_id", "label", "snapshot_date"),
        on=["loan_id"],
        how="inner"
    )
    df_pd = df.toPandas()
    spark.stop()

    # 2. Prepare features and label
    label_col = 'label'
    exclude_cols = ['loan_id', 'snapshot_date', label_col]
    feature_cols = [c for c in df_pd.columns if c not in exclude_cols]
    X = df_pd[feature_cols]
    y = df_pd[label_col]

    # 3. Train/test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train CatBoost
    catboost_info_dir = "/opt/airflow/scripts/utils/catboost_info"
    os.makedirs(catboost_info_dir, exist_ok=True)
    params = {
        'iterations': 150,
        'depth': 4,
        'learning_rate': 0.1,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'random_seed': 42,
        'verbose': False,
        'train_dir': catboost_info_dir
    }
    model = CatBoostClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
        verbose=10
    )

    # 5. Evaluate
    y_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    print(f"[Monthly Training] CatBoost Validation AUC: {auc:.4f}")

    # 6. Log to MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("assignment2")
    with mlflow.start_run(run_name="catboost_monthly"):
        mlflow.log_params(params)
        mlflow.log_metric("auc", auc)
        mlflow.catboost.log_model(model, "model")
        mlflow.log_param("feature_names", feature_cols)
    print("CatBoost monthly model training and logging complete.")


def select_best_model_monthly(**context):
    """
    Extract results from both monthly training tasks and select the best model.
    Same logic as initial training.
    """
    print("Extracting results from both monthly training tasks and selecting best model...")
    
    # Extract LightGBM results
    print("Extracting LightGBM results...")
    lightgbm_run_id = extract_mlflow_run_id_from_logs(task_id='train_lightgbm_monthly', **context)
    lightgbm_f1 = extract_metrics_from_logs(task_id='train_lightgbm_monthly', **context)
    
    # If extraction failed, query MLflow
    if lightgbm_run_id is None or lightgbm_f1 is None:
        print("LightGBM log extraction failed, querying MLflow directly...")
        lightgbm_run_id, lightgbm_f1 = query_mlflow_for_run_info(model_type='lightgbm', **context)
    
    # Extract CatBoost results
    print("Extracting CatBoost results...")
    catboost_run_id = extract_mlflow_run_id_from_logs(task_id='train_catboost_monthly', **context)
    catboost_f1 = extract_metrics_from_logs(task_id='train_catboost_monthly', **context)
    
    # If extraction failed, query MLflow
    if catboost_run_id is None or catboost_f1 is None:
        print("CatBoost log extraction failed, querying MLflow directly...")
        catboost_run_id, catboost_f1 = query_mlflow_for_run_info(model_type='catboost', **context)
    
    # Handle missing results
    if lightgbm_run_id is None:
        print("Warning: Could not get LightGBM run ID")
        lightgbm_run_id = "unknown"
        lightgbm_f1 = 0.0
    
    if catboost_run_id is None:
        print("Warning: Could not get CatBoost run ID")
        catboost_run_id = "unknown"
        catboost_f1 = 0.0
    
    if lightgbm_f1 is None:
        print("Warning: Could not get LightGBM Macro F1 score")
        lightgbm_f1 = 0.0
    
    if catboost_f1 is None:
        print("Warning: Could not get CatBoost Macro F1 score")
        catboost_f1 = 0.0
    
    # Push individual results to XComs for debugging/auditing
    context['task_instance'].xcom_push(key='lightgbm_run_id_monthly', value=lightgbm_run_id)
    context['task_instance'].xcom_push(key='catboost_run_id_monthly', value=catboost_run_id)
    context['task_instance'].xcom_push(key='lightgbm_macro_f1_monthly', value=lightgbm_f1)
    context['task_instance'].xcom_push(key='catboost_macro_f1_monthly', value=catboost_f1)
    
    print(f"LightGBM Run ID: {lightgbm_run_id}, Macro F1: {lightgbm_f1:.4f}")
    print(f"CatBoost Run ID: {catboost_run_id}, Macro F1: {catboost_f1:.4f}")
    
    # Select best model
    if lightgbm_f1 > catboost_f1:
        best_run_id = lightgbm_run_id
        best_model_type = "LightGBM"
        best_f1 = lightgbm_f1
    else:
        best_run_id = catboost_run_id
        best_model_type = "CatBoost"
        best_f1 = catboost_f1
    
    print(f"Selected {best_model_type} as best monthly model with Macro F1: {best_f1:.4f}")
    
    # Push best model info to XComs
    context['task_instance'].xcom_push(key='best_run_id_monthly', value=best_run_id)
    context['task_instance'].xcom_push(key='best_model_type_monthly', value=best_model_type)
    context['task_instance'].xcom_push(key='best_macro_f1_monthly', value=best_f1)
    
    return f"Best monthly model selected: {best_model_type} (Run ID: {best_run_id}, Macro F1: {best_f1:.4f})"


def register_model_monthly(**context):
    """
    Register the best monthly model to MLflow Model Registry and promote to Production.
    Same logic as initial registration but updates retraining tracker.
    """
    # Get best model info from XComs
    best_run_id = context['task_instance'].xcom_pull(key='best_run_id_monthly')
    best_model_type = context['task_instance'].xcom_pull(key='best_model_type_monthly')
    best_f1 = context['task_instance'].xcom_pull(key='best_macro_f1_monthly')
    
    print(f"Registering {best_model_type} monthly model (Run ID: {best_run_id}) to MLflow Registry")
    
    # Set up MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    
    # Get the model URI from the run
    model_uri = f"runs:/{best_run_id}/model"
    
    # Register the model (creates new version)
    model_name = "credit_scoring_model"
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )
    
    # Transition to Production
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Production"
    )
    
    print(f"Monthly model registered successfully: {model_name} v{model_version.version}")
    print(f"Model promoted to Production stage")
    
    # Update retraining tracker with current date (resets 90-day timer)
    retraining_tracker_path = "/opt/airflow/logs/last_retraining_date.json"
    tracker_data = {
        'last_retraining_date': context['logical_date'].strftime('%Y-%m-%d'),
        'model_name': model_name,
        'model_version': model_version.version,
        'model_type': best_model_type,
        'macro_f1_score': best_f1,
        'registration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'retraining_type': 'monthly'
    }
    
    os.makedirs(os.path.dirname(retraining_tracker_path), exist_ok=True)
    with open(retraining_tracker_path, 'w') as f:
        json.dump(tracker_data, f, indent=2)
    
    print(f"Retraining tracker updated: {retraining_tracker_path}")
    
    return f"Monthly model registered and promoted to Production: {model_name} v{model_version.version}"


def train_lightgbm_initial(**context):
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    import mlflow
    import os
    import pandas as pd
    from pyspark.sql import SparkSession

    # 1. Load data (same logic as your notebook)
    spark = SparkSession.builder.appName("LightGBMInitialTraining").getOrCreate()
    start_date = "2023-07-01"
    end_date = "2024-06-01"
    feature_dir = "/opt/airflow/scripts/datamart/gold/feature_store"
    label_dir = "/opt/airflow/scripts/datamart/gold/label_store"

    def get_files_in_range(folder, prefix, start_date, end_date):
        files = []
        for fname in os.listdir(folder):
            if fname.startswith(prefix) and fname.endswith('.parquet'):
                date_str = fname.replace(prefix, '').replace('.parquet', '').replace('_', '-')
                if start_date <= date_str <= end_date:
                    files.append(os.path.join(folder, fname))
        return sorted(files)

    feature_files = get_files_in_range(feature_dir, "gold_feature_store_", start_date, end_date)
    label_files = get_files_in_range(label_dir, "gold_label_store_", start_date, end_date)
    df_features = spark.read.parquet(*feature_files)
    df_labels = spark.read.parquet(*label_files)
    df = df_features.join(
        df_labels.select("loan_id", "label", "snapshot_date"),
        on=["loan_id"],
        how="inner"
    )
    df_pd = df.toPandas()
    spark.stop()

    # 2. Prepare features and label
    label_col = 'label'
    exclude_cols = ['loan_id', 'snapshot_date', label_col]
    feature_cols = [c for c in df_pd.columns if c not in exclude_cols]
    X = df_pd[feature_cols]
    y = df_pd[label_col]

    # 3. Train/test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train LightGBM
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'max_depth': 4,
        'learning_rate': 0.1,
        'n_estimators': 150,
        'num_leaves': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbose': -1
    }
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        num_boost_round=100
    )

    # 5. Evaluate
    y_pred = model.predict(X_val)
    auc = roc_auc_score(y_val, y_pred)
    print(f"Validation AUC: {auc:.4f}")

    # 6. Log to MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("assignment2")
    with mlflow.start_run(run_name="lightgbm_initial"):
        mlflow.log_params(params)
        mlflow.log_metric("auc", auc)
        mlflow.lightgbm.log_model(model, "model")
        mlflow.log_param("feature_names", feature_cols)
    print("LightGBM initial model training and logging complete.")


def train_catboost_initial(**context):
    from catboost import CatBoostClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    import mlflow
    import os
    import pandas as pd
    from pyspark.sql import SparkSession

    # 1. Load data (same logic as above)
    spark = SparkSession.builder.appName("CatBoostInitialTraining").getOrCreate()
    start_date = "2023-07-01"
    end_date = "2024-06-01"
    feature_dir = "/opt/airflow/scripts/datamart/gold/feature_store"
    label_dir = "/opt/airflow/scripts/datamart/gold/label_store"

    def get_files_in_range(folder, prefix, start_date, end_date):
        files = []
        for fname in os.listdir(folder):
            if fname.startswith(prefix) and fname.endswith('.parquet'):
                date_str = fname.replace(prefix, '').replace('.parquet', '').replace('_', '-')
                if start_date <= date_str <= end_date:
                    files.append(os.path.join(folder, fname))
        return sorted(files)

    feature_files = get_files_in_range(feature_dir, "gold_feature_store_", start_date, end_date)
    label_files = get_files_in_range(label_dir, "gold_label_store_", start_date, end_date)
    df_features = spark.read.parquet(*feature_files)
    df_labels = spark.read.parquet(*label_files)
    df = df_features.join(
        df_labels.select("loan_id", "label", "snapshot_date"),
        on=["loan_id"],
        how="inner"
    )
    df_pd = df.toPandas()
    spark.stop()

    # 2. Prepare features and label
    label_col = 'label'
    exclude_cols = ['loan_id', 'snapshot_date', label_col]
    feature_cols = [c for c in df_pd.columns if c not in exclude_cols]
    X = df_pd[feature_cols]
    y = df_pd[label_col]

    # 3. Train/test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train CatBoost
    catboost_info_dir = "/opt/airflow/scripts/utils/catboost_info"
    os.makedirs(catboost_info_dir, exist_ok=True)
    params = {
        'iterations': 150,
        'depth': 4,
        'learning_rate': 0.1,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'random_seed': 42,
        'verbose': False,
        'train_dir': catboost_info_dir
    }
    model = CatBoostClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
        verbose=10
    )

    # 5. Evaluate
    y_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    print(f"CatBoost Validation AUC: {auc:.4f}")

    # 6. Log to MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("assignment2")
    with mlflow.start_run(run_name="catboost_initial"):
        mlflow.log_params(params)
        mlflow.log_metric("auc", auc)
        mlflow.catboost.log_model(model, "model")
        mlflow.log_param("feature_names", feature_cols)
    print("CatBoost initial model training and logging complete.")


# Future functions for monthly lifecycle (to be implemented) 