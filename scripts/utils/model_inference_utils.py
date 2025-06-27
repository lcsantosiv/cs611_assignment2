import pandas as pd
import mlflow
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder
from pyspark.sql import SparkSession
import os
from typing import List, Dict, Any
import numpy as np

# Configuration (can be overridden by caller)
PG_CONFIG = {
    'host': 'postgres',
    'port': 5432,
    'database': 'airflow',
    'user': 'airflow',
    'password': 'airflow'
}

MLFLOW_TRACKING_URI = "http://mlflow:5000"

# --- Model Loading ---
def load_model_from_mlflow(run_id: str, model_type: str):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    feature_names = json.loads(run.data.params.get('feature_names', '[]'))
    grade_mapping = {}
    grade_mapping_param = run.data.params.get('grade_mapping', '')
    if grade_mapping_param:
        grade_mapping = json.loads(grade_mapping_param)
    return model, grade_mapping, feature_names

def list_models_in_registry():
    """
    List all models and their versions in the MLflow model registry.
    Useful for debugging model loading issues.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    
    try:
        # List all registered models using the correct API
        models = client.search_registered_models()
        print(f"[Registry] Found {len(models)} registered models:")
        
        for model in models:
            print(f"  - Model: {model.name}")
            # Get versions for this model
            versions = client.search_model_versions(f"name='{model.name}'")
            for version in versions:
                print(f"    Version {version.version}: stage={version.current_stage}, run_id={version.run_id}")
                
    except Exception as e:
        print(f"Error listing models: {e}")

def load_deployed_model_from_registry(model_name: str = "credit_scoring_model", version: str = None):
    """
    Load the currently deployed model from MLflow's model registry.
    
    Args:
        model_name: Name of the model in the registry
        version: Specific version to load (if None, tries Production stage first)
        
    Returns:
        Dictionary containing model, feature_names, and grade_mapping
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    
    try:
        # Determine model URI based on version or stage
        if version:
            model_uri = f"models:/{model_name}/{version}"
            print(f"[Registry] Loading specific version: {version}")
        else:
            # Try Production stage first, then fall back to latest version
            try:
                model_version = client.get_latest_versions(model_name, stages=["Production"])[0]
                model_uri = f"models:/{model_name}/Production"
                print(f"[Registry] Loading Production stage model")
            except:
                # If no Production model, get the latest version
                model_versions = client.search_model_versions(f"name='{model_name}'")
                if not model_versions:
                    raise Exception(f"No versions found for model '{model_name}'")
                latest_version = max(model_versions, key=lambda v: v.version)
                model_uri = f"models:/{model_name}/{latest_version.version}"
                print(f"[Registry] Loading latest version: {latest_version.version}")
        
        # Try different model loading methods
        model = None
        
        # Method 1: Try sklearn flavor
        try:
            model = mlflow.sklearn.load_model(model_uri)
            print(f"[Registry] Loaded model using sklearn flavor")
        except Exception as e:
            print(f"[Registry] sklearn loading failed: {e}")
        
        # Method 2: Try lightgbm flavor
        if model is None:
            try:
                model = mlflow.lightgbm.load_model(model_uri)
                print(f"[Registry] Loaded model using lightgbm flavor")
            except Exception as e:
                print(f"[Registry] lightgbm loading failed: {e}")
        
        # Method 3: Try python_function flavor (generic)
        if model is None:
            try:
                model = mlflow.pyfunc.load_model(model_uri)
                print(f"[Registry] Loaded model using python_function flavor")
            except Exception as e:
                print(f"[Registry] python_function loading failed: {e}")
        
        # Method 4: Try catboost flavor
        if model is None:
            try:
                model = mlflow.catboost.load_model(model_uri)
                print(f"[Registry] Loaded model using catboost flavor")
            except Exception as e:
                print(f"[Registry] catboost loading failed: {e}")
        
        if model is None:
            raise Exception("Could not load model with any available flavor")
        
        # Get the run information to extract feature names and grade mapping
        # Extract version from model_uri for getting run info
        if version:
            model_version = client.get_model_version(model_name, version)
        else:
            try:
                model_version = client.get_latest_versions(model_name, stages=["Production"])[0]
            except:
                model_versions = client.search_model_versions(f"name='{model_name}'")
                model_version = max(model_versions, key=lambda v: v.version)
        
        run = client.get_run(model_version.run_id)
        
        # Extract feature_names with better error handling
        feature_names = []
        feature_names_param = run.data.params.get('feature_names', '[]')
        try:
            if feature_names_param and feature_names_param.strip():
                feature_names = json.loads(feature_names_param)
            else:
                print(f"[Registry] No feature_names param found, will use default features")
        except json.JSONDecodeError as e:
            print(f"[Registry] Error parsing feature_names JSON '{feature_names_param}': {e}")
            print(f"[Registry] Will use default features")
            feature_names = []
        
        # Extract grade_mapping with better error handling
        grade_mapping = {}
        grade_mapping_param = run.data.params.get('grade_mapping', '')
        try:
            if grade_mapping_param and grade_mapping_param.strip():
                grade_mapping = json.loads(grade_mapping_param)
        except json.JSONDecodeError as e:
            print(f"[Registry] Error parsing grade_mapping JSON '{grade_mapping_param}': {e}")
            grade_mapping = {}
        
        return {
            "model": model,
            "feature_names": feature_names,
            "grade_mapping": grade_mapping,
            "run_id": model_version.run_id,
            "model_version": model_version.version
        }
        
    except Exception as e:
        print(f"Error loading deployed model: {e}")
        raise

# --- Data Loading ---
def load_weekly_data(spark: SparkSession, week_date: str, feature_store_path: str, label_store_path: str) -> pd.DataFrame:
    feature_path = os.path.join(feature_store_path, f"gold_feature_store_{week_date}.parquet")
    label_path = os.path.join(label_store_path, f"gold_label_store_{week_date}.parquet")
    if not os.path.exists(feature_path) or not os.path.exists(label_path):
        print(f"Feature or label directory not found for week {week_date}")
        print(f"Looking for: {feature_path}")
        print(f"Looking for: {label_path}")
        return None
    
    feature_df = spark.read.parquet(feature_path)
    label_df = spark.read.parquet(label_path)
    
    # Print column names for debugging
    print(f"[Data Loading] Feature columns: {feature_df.columns}")
    print(f"[Data Loading] Label columns: {label_df.columns}")
    
    # Check which join column to use
    if 'id' in feature_df.columns and 'id' in label_df.columns:
        join_column = 'id'
    elif 'loan_id' in feature_df.columns and 'id' in label_df.columns:
        # Rename loan_id to id in feature_df for consistency
        feature_df = feature_df.withColumnRenamed('loan_id', 'id')
        join_column = 'id'
    elif 'loan_id' in feature_df.columns and 'loan_id' in label_df.columns:
        join_column = 'loan_id'
    else:
        print(f"[Data Loading] Error: No common join column found between feature and label DataFrames")
        return None
    
    print(f"[Data Loading] Using join column: {join_column}")
    full_df = feature_df.join(label_df, on=join_column, how='inner')
    return full_df.toPandas()

# --- Feature Preparation ---
def prepare_features(df: pd.DataFrame, feature_names: List[str], target_column: str = 'label'):
    df = df.copy()
    
    # For binary classification, no need for label encoding
    if target_column == 'label':
        # Binary label is already 0/1, no encoding needed
        y = df[target_column]
    else:
        # For multi-class (grade prediction), use label encoding
        label_encoder = LabelEncoder()
        df['grade_encoded'] = label_encoder.fit_transform(df[target_column])
        y = df['grade_encoded']
    
    features_to_drop = [
        'id', 'loan_id', 'customer_id', target_column, 'snapshot_date', 'earliest_cr_date',
        'snapshot_month', 'earliest_cr_month', 'months_since_earliest_cr_line', 'label_def'
    ]
    df_cleaned = df.drop(columns=features_to_drop + ['grade_encoded'], errors='ignore')
    
    if not feature_names:
        feature_names = list(df_cleaned.columns)
    else:
        missing_features = set(feature_names) - set(df_cleaned.columns)
        if missing_features:
            missing_cols_dict = {feature: 0 for feature in missing_features}
            df_cleaned = df_cleaned.assign(**missing_cols_dict)
    
    X = df_cleaned[feature_names]
    return X, y, None  # No label_encoder needed for binary

# --- Evaluation ---
def evaluate_weekly_performance(model, X, y, grade_mapping):
    y_pred = model.predict(X)
    if hasattr(y_pred, 'ndim') and y_pred.ndim > 1:
        y_pred = y_pred.flatten()
    accuracy = accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average='macro')
    weighted_f1 = f1_score(y, y_pred, average='weighted')
    f1_per_class = f1_score(y, y_pred, average=None)
    reverse_grade_mapping = {idx: grade for grade, idx in grade_mapping.items()}
    f1_by_grade = {}
    for i, score in enumerate(f1_per_class):
        if i in reverse_grade_mapping:
            f1_by_grade[reverse_grade_mapping[i]] = score
        else:
            f1_by_grade[f'Unknown_{i}'] = score
    metrics = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'f1_by_grade': f1_by_grade,
        'total_samples': len(y),
        'predictions_distribution': pd.Series(y_pred).value_counts().to_dict()
    }
    return metrics, y_pred

# --- Metrics Saving ---
def save_metrics_to_postgres(metrics: Dict[str, Any], week_date: str, run_id: str, model_name: str, pg_config: Dict[str, Any] = None):
    if pg_config is None:
        pg_config = PG_CONFIG
    try:
        conn = psycopg2.connect(**pg_config)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS model_performance_metrics (
            id SERIAL PRIMARY KEY,
            evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            week_date VARCHAR(10),
            mlflow_run_id VARCHAR(50),
            model_name VARCHAR(100),
            accuracy DECIMAL(5,4),
            macro_f1 DECIMAL(5,4),
            weighted_f1 DECIMAL(5,4),
            total_samples INTEGER,
            f1_by_grade JSONB,
            predictions_distribution JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cursor.execute(create_table_sql)
        insert_sql = """
        INSERT INTO model_performance_metrics 
        (week_date, mlflow_run_id, model_name, accuracy, macro_f1, weighted_f1, 
         total_samples, f1_by_grade, predictions_distribution)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(insert_sql, (
            week_date,
            run_id,
            model_name,
            metrics['accuracy'],
            metrics['macro_f1'],
            metrics['weighted_f1'],
            metrics['total_samples'],
            json.dumps(metrics['f1_by_grade']),
            json.dumps(metrics['predictions_distribution'])
        ))
        conn.commit()
        print(f"✅ Metrics saved to PostgreSQL for {model_name} - week {week_date}")
    except Exception as e:
        print(f"❌ Error saving to PostgreSQL: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

def calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate comprehensive model metrics including AUC, precision, recall, and accuracy.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional, for AUC calculation)
    
    Returns:
        Dictionary containing all metrics
    """
    metrics = {}
    
    # Basic classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # AUC calculation if probabilities are provided
    if y_pred_proba is not None:
        try:
            # For binary classification, use the positive class probability
            if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
                metrics['auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:
                metrics['auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
        except Exception as e:
            print(f"Warning: Could not calculate AUC: {e}")
            metrics['auc'] = None
    else:
        metrics['auc'] = None
    
    # Additional metrics
    metrics['total_samples'] = len(y_true)
    
    # Calculate per-class metrics
    class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics['f1_by_grade'] = {}
    for class_name, class_metrics in class_report.items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            metrics['f1_by_grade'][class_name] = class_metrics.get('f1-score', 0)
    
    # Prediction distribution
    unique, counts = np.unique(y_pred, return_counts=True)
    metrics['predictions_distribution'] = dict(zip(unique, counts))
    
    return metrics

def save_model_metrics_to_postgres(metrics: Dict[str, Any], month_date: str, run_id: str, model_name: str, pg_config: Dict[str, Any] = None):
    """
    Save comprehensive model metrics to PostgreSQL model_metrics table.
    
    Args:
        metrics: Dictionary containing model metrics
        month_date: Date string for the month being evaluated
        run_id: MLflow run ID
        model_name: Name of the model
        pg_config: PostgreSQL configuration
    """
    if pg_config is None:
        pg_config = PG_CONFIG
    
    try:
        conn = psycopg2.connect(**pg_config)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Create model_metrics table if it doesn't exist
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS model_metrics (
            id SERIAL PRIMARY KEY,
            evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            month_date VARCHAR(10),
            mlflow_run_id VARCHAR(50),
            model_name VARCHAR(100),
            accuracy DECIMAL(5,4),
            precision DECIMAL(5,4),
            recall DECIMAL(5,4),
            auc DECIMAL(5,4),
            macro_f1 DECIMAL(5,4),
            weighted_f1 DECIMAL(5,4),
            total_samples INTEGER,
            f1_by_grade JSONB,
            predictions_distribution JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cursor.execute(create_table_sql)
        
        # Insert metrics
        insert_sql = """
        INSERT INTO model_metrics 
        (month_date, mlflow_run_id, model_name, accuracy, precision, recall, auc,
         macro_f1, weighted_f1, total_samples, f1_by_grade, predictions_distribution)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        cursor.execute(insert_sql, (
            month_date,
            run_id,
            model_name,
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['auc'],
            metrics['macro_f1'],
            metrics['weighted_f1'],
            metrics['total_samples'],
            json.dumps({k: float(v) if isinstance(v, (np.integer, np.floating)) else v for k, v in metrics['f1_by_grade'].items()}),
            json.dumps({int(k): int(v) if isinstance(v, (np.integer, np.floating)) else v for k, v in metrics['predictions_distribution'].items()})
        ))
        
        conn.commit()
        print(f"✅ Comprehensive metrics saved to model_metrics table for {model_name} - month {month_date}")
        
    except Exception as e:
        print(f"❌ Error saving to model_metrics table: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

def save_model_inference_to_postgres(inference_data: pd.DataFrame, month_date: str, model_name: str, model_version: str = None, mlflow_run_id: str = None, execution_context: str = 'monthly_inference', pg_config: Dict[str, Any] = None):
    """
    Save model inference results to PostgreSQL model_inference table for binary classification.
    
    Args:
        inference_data: DataFrame containing inference results with columns:
                       - id: Customer ID
                       - snapshot_date: Date of the snapshot
                       - predicted_default: Binary prediction (0 or 1)
                       - default_probability: Probability of default
                       - model_name: Name of the model used
        month_date: Date string for the month being processed
        model_name: Name of the model
        model_version: Version of the model used
        mlflow_run_id: MLflow run ID
        execution_context: Context of the inference (e.g., 'monthly_inference')
        pg_config: PostgreSQL configuration
    """
    if pg_config is None:
        pg_config = PG_CONFIG
    
    try:
        conn = psycopg2.connect(**pg_config)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Create model_inference table for binary classification
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS model_inference (
            id SERIAL PRIMARY KEY,
            customer_id VARCHAR(50),
            snapshot_date DATE,
            predicted_default INTEGER,  -- 0 or 1
            default_probability DECIMAL(5,4),  -- Probability of default
            model_name VARCHAR(100),
            model_version VARCHAR(20),  -- Model version used
            mlflow_run_id VARCHAR(50),  -- MLflow run ID
            inference_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            month_partition VARCHAR(10),
            execution_context VARCHAR(100)  -- e.g., 'monthly_inference', 'batch_inference'
        );
        """
        cursor.execute(create_table_sql)
        
        # Create index for better query performance
        create_index_sql = """
        CREATE INDEX IF NOT EXISTS idx_model_inference_customer_date 
        ON model_inference(customer_id, snapshot_date);
        """
        cursor.execute(create_index_sql)
        
        # Prepare data for insertion
        insert_data = []
        for _, row in inference_data.iterrows():
            insert_data.append((
                row.get('id', row.get('Customer_ID', '')),
                row.get('snapshot_date'),
                row.get('predicted_default', 0),
                row.get('default_probability', 0.0),
                model_name,
                model_version or '',
                mlflow_run_id or '',
                pd.Timestamp.now(),  # inference_date
                month_date,  # month_partition
                execution_context
            ))
        
        # Insert inference results
        insert_sql = """
        INSERT INTO model_inference 
        (customer_id, snapshot_date, predicted_default, default_probability, model_name, model_version, mlflow_run_id, inference_date, month_partition, execution_context)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        cursor.executemany(insert_sql, insert_data)
        conn.commit()
        
        print(f"✅ Model inference results saved to model_inference table for {model_name} - month {month_date}")
        print(f"   Records inserted: {len(insert_data)}")
        
    except Exception as e:
        print(f"❌ Error saving to model_inference table: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

def run_model_inference_with_saving(model, X, customer_ids, snapshot_dates, model_name, month_date, model_version=None, mlflow_run_id=None, pg_config=None):
    """
    Run model inference and save results for binary classification.
    
    Args:
        model: Trained model
        X: Feature matrix
        customer_ids: List of customer IDs
        snapshot_dates: List of snapshot dates
        model_name: Name of the model
        month_date: Month date string
        model_version: Version of the model
        mlflow_run_id: MLflow run ID
        pg_config: PostgreSQL configuration
    
    Returns:
        DataFrame with inference results
    """
    # Get predictions
    y_pred = model.predict(X)
    
    # For PyFuncModel, we might need to handle the output differently
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X)
        # For binary classification, save default probability and prediction
        default_probabilities = y_pred_proba[:, 1]  # Probability of default (class 1)
    else:
        # Try to get probabilities from the underlying model
        try:
            # For CatBoost pyfunc models, we can access the underlying model
            if hasattr(model, '_model_impl') and hasattr(model._model_impl, 'cb_model'):
                # Access the underlying CatBoost model
                cb_model = model._model_impl.cb_model
                y_pred_proba = cb_model.predict_proba(X)
                default_probabilities = y_pred_proba[:, 1]  # Probability of default (class 1)
            else:
                # If no predict_proba, use the prediction as probability (0 or 1)
                default_probabilities = y_pred
        except Exception as e:
            print(f"[Inference] Could not get probabilities: {e}")
            # If no predict_proba, use the prediction as probability (0 or 1)
            default_probabilities = y_pred
    
    predicted_defaults = y_pred  # Binary prediction (0 or 1)
    
    # Create inference results DataFrame
    inference_results = pd.DataFrame({
        'id': customer_ids,
        'snapshot_date': snapshot_dates,
        'predicted_default': predicted_defaults,
        'default_probability': default_probabilities,
        'model_name': model_name
    })
    
    # Save to PostgreSQL
    save_model_inference_to_postgres(
        inference_results, 
        month_date, 
        model_name, 
        model_version, 
        mlflow_run_id, 
        'monthly_inference', 
        pg_config
    )
    
    return inference_results 