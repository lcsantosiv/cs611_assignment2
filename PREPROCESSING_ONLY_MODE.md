# Preprocessing-Only Mode Configuration

## Overview
The ML pipeline has been temporarily disabled to focus on testing the data preprocessing pipeline. All ML-related tasks are commented out and replaced with dummy operators.

## Current Pipeline Flow

### Active Components:
1. **Data Source Sensors** - Check for source CSV files
2. **Bronze Layer Processing** - Process raw data into bronze tables
3. **Silver Layer Processing** - Transform bronze data into silver tables
4. **Gold Layer Processing** - Create feature and label stores
5. **Dummy Completion** - Simple completion markers

### Disabled Components:
- ❌ ML Pipeline decision logic
- ❌ Model training (LightGBM & CatBoost)
- ❌ Model evaluation and retraining triggers
- ❌ Model inference
- ❌ MLflow integration
- ❌ PostgreSQL metrics saving

## Pipeline Structure

```
wait_for_previous_run
    ↓
start_pipeline
    ↓
start_preprocessing
    ↓
[dep_check_source_data_bronze_1, dep_check_source_data_bronze_2, 
 dep_check_source_data_bronze_3, dep_check_source_data_bronze_4]
    ↓
[bronze_table_1, bronze_table_2, bronze_table_3, bronze_table_4]
    ↓
[silver_table_1, silver_table_2, silver_table_3, silver_table_4]
    ↓
gold_feature_and_label_store
    ↓
gold_table_completed
    ↓
ml_pipeline_disabled (dummy)
    ↓
preprocessing_complete (dummy)
```

## Files Modified

### `dags/dag.py`
- ✅ Commented out all ML pipeline tasks
- ✅ Added dummy operators: `ml_pipeline_disabled`, `preprocessing_complete`
- ✅ Updated external task sensor to wait for `preprocessing_complete`
- ✅ Updated pipeline dependencies to skip ML pipeline

### ML Pipeline Tasks (Commented Out):
```python
# decide_pipeline_path_task
# skip_run
# run_initial_training_flow
# train_lightgbm_initial
# train_catboost_initial
# select_best_model_initial_task
# register_model_initial_task
# run_monthly_lifecycle_flow
# evaluate_production_model_task
# check_retraining_trigger_task
# skip_retraining
# trigger_retraining
# train_lightgbm_monthly_task
# train_catboost_monthly_task
# select_best_model_monthly_task
# register_model_monthly_task
# run_model_inference_task
# end
```

## Testing Strategy

### 1. Data Preprocessing Validation
- Run backfills to test data processing pipeline
- Verify bronze, silver, and gold table creation
- Check data quality and completeness
- Validate file paths and naming conventions

### 2. Monthly Dependencies
- Test external task sensor functionality
- Verify proper monthly sequencing
- Check data consistency across months

### 3. Error Handling
- Test with missing source files
- Verify error propagation
- Check retry mechanisms

## Re-enabling ML Pipeline

To re-enable the ML pipeline later:

1. **Uncomment ML Pipeline Tasks**: Remove `#` from all commented ML tasks
2. **Update Dependencies**: Restore ML pipeline dependency chains
3. **Update External Task Sensor**: Change back to `end_pipeline`
4. **Test Incrementally**: Enable one path at a time

### Quick Re-enable Steps:
```python
# 1. Uncomment all ML pipeline tasks
# 2. Update external task sensor:
external_task_id='end_pipeline'

# 3. Restore dependencies:
gold_table_completed >> decide_pipeline_path_task
# ... (rest of ML dependencies)
```

## Benefits of This Approach

1. **Focused Testing**: Isolate data preprocessing issues
2. **Faster Execution**: No ML training overhead
3. **Clear Separation**: Data vs ML pipeline concerns
4. **Easy Rollback**: Simple to re-enable ML components
5. **Resource Efficiency**: Lower computational requirements

## Monitoring

### Key Metrics to Monitor:
- Data processing completion rates
- File generation success
- Processing time per layer
- Data quality metrics
- Error rates and types

### Log Analysis:
- Check for data processing errors
- Verify file path resolution
- Monitor memory usage
- Track processing performance

## Next Steps

1. **Run Preprocessing Tests**: Execute backfills for multiple months
2. **Validate Data Quality**: Check output files and data integrity
3. **Performance Tuning**: Optimize processing if needed
4. **Re-enable ML Pipeline**: Once preprocessing is stable

This configuration allows you to thoroughly test the data preprocessing pipeline before adding the complexity of the ML components. 