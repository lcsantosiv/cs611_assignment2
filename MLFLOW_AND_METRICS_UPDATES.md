# MLflow and Metrics Updates Summary

## Overview
This document summarizes the changes made to implement the three requested updates:
1. MLflow experiment name changed to "assignment2"
2. Comprehensive model metrics calculation and logging to `model_metrics` table
3. Model inference results saved to `model_inference` table

## 1. MLflow Experiment Name Update

### Files Modified:
- `dags/dag_functions.py`
- `scripts/utils/LightGBM_training_run.py`
- `scripts/utils/CatBoost_training_run.py`
- `scripts/utils/weekly_evaluation.py`

### Changes:
- Updated MLflow experiment name from "test" to "assignment2" in all training and evaluation scripts
- This ensures all model runs are logged under the "assignment2" experiment in MLflow

## 2. Comprehensive Model Metrics

### New Functions Added:
- `calculate_comprehensive_metrics()` in `scripts/utils/model_inference_utils.py`
- `save_model_metrics_to_postgres()` in `scripts/utils/model_inference_utils.py`

### Metrics Calculated:
- **Accuracy**: Overall classification accuracy
- **Precision**: Weighted precision score
- **Recall**: Weighted recall score
- **AUC**: Area Under the ROC Curve (if probabilities available)
- **Macro F1**: Macro-averaged F1 score
- **Weighted F1**: Weighted F1 score
- **Per-class F1 scores**: F1 score for each grade
- **Prediction distribution**: Count of predictions per class

### PostgreSQL Table Schema:
```sql
CREATE TABLE model_metrics (
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
```

### Files Updated:
- `scripts/utils/model_inference_utils.py` - Added comprehensive metrics functions
- `dags/dag_functions.py` - Updated inference function to use comprehensive metrics
- `scripts/utils/LightGBM_training_run.py` - Updated to calculate and save comprehensive metrics
- `scripts/utils/CatBoost_training_run.py` - Updated to calculate and save comprehensive metrics

## 3. Model Inference Saving

### New Functions Added:
- `save_model_inference_to_postgres()` in `scripts/utils/model_inference_utils.py`
- `run_model_inference_with_saving()` in `scripts/utils/model_inference_utils.py`

### Inference Data Saved:
- Customer ID
- Snapshot date
- Predicted grade
- Prediction probability
- Model name
- Inference timestamp
- Month partition for efficient querying

### PostgreSQL Table Schema:
```sql
CREATE TABLE model_inference (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50),
    snapshot_date DATE,
    predicted_grade VARCHAR(10),
    prediction_probability DECIMAL(5,4),
    model_name VARCHAR(100),
    inference_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    month_partition VARCHAR(10)
);

-- Index for better query performance
CREATE INDEX idx_model_inference_customer_date 
ON model_inference(customer_id, snapshot_date);
```

### Files Updated:
- `scripts/utils/model_inference_utils.py` - Added inference saving functions
- `dags/dag_functions.py` - Updated to save inference results to PostgreSQL

## 4. Integration Points

### Training Pipeline:
1. Models are trained using LightGBM or CatBoost
2. Comprehensive metrics are calculated on test set
3. Metrics are logged to MLflow under "assignment2" experiment
4. Metrics are saved to `model_metrics` table in PostgreSQL

### Inference Pipeline:
1. Production model is loaded from MLflow
2. Inference is run on monthly data
3. Predictions are saved to `model_inference` table
4. Performance metrics are calculated and saved to `model_metrics` table

### Evaluation Pipeline:
1. Models are evaluated on historical data
2. Comprehensive metrics are calculated
3. Results are saved to both MLflow and PostgreSQL

## 5. Usage Examples

### Querying Model Metrics:
```sql
-- Get latest model performance
SELECT * FROM model_metrics 
WHERE model_name = 'credit_scoring_model' 
ORDER BY evaluation_date DESC 
LIMIT 1;

-- Compare model performance over time
SELECT month_date, accuracy, precision, recall, auc, macro_f1 
FROM model_metrics 
WHERE model_name = 'credit_scoring_model' 
ORDER BY month_date;
```

### Querying Model Inferences:
```sql
-- Get predictions for a specific customer
SELECT * FROM model_inference 
WHERE customer_id = 'CUS_123' 
ORDER BY snapshot_date DESC;

-- Get predictions for a specific month
SELECT * FROM model_inference 
WHERE month_partition = '2024_01_01' 
AND model_name = 'credit_scoring_model';
```

## 6. Benefits

1. **Centralized Experiment Tracking**: All models logged under "assignment2" experiment
2. **Comprehensive Evaluation**: Multiple metrics provide better model assessment
3. **Persistent Storage**: All metrics and inferences stored in PostgreSQL for analysis
4. **Audit Trail**: Complete history of model performance and predictions
5. **Scalability**: Efficient indexing and partitioning for large datasets

## 7. Testing Recommendations

1. **Verify MLflow Experiment**: Check that new runs appear under "assignment2" experiment
2. **Test Metrics Calculation**: Run training scripts and verify metrics in PostgreSQL
3. **Test Inference Saving**: Run inference pipeline and verify data in `model_inference` table
4. **Performance Testing**: Test with large datasets to ensure efficient database operations
5. **Data Validation**: Verify that all metrics and inference data are correctly saved

## 8. Monitoring

- Monitor PostgreSQL table sizes and query performance
- Set up alerts for failed metric calculations or inference saving
- Track model performance trends using the saved metrics
- Monitor MLflow experiment storage usage 