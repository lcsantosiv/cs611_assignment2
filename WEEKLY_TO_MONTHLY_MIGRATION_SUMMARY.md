# Weekly to Monthly Migration Summary

## Overview
This document summarizes all the changes required to convert the ML pipeline from weekly to monthly execution.

## Key Changes Made

### 1. **DAG Schedule** ✅
- **File**: `dags/dag.py`
- **Change**: Schedule already set to monthly (`'0 0 1 * *'`)
- **Status**: No change needed

### 2. **External Task Sensor** ✅
- **File**: `dags/dag.py`
- **Change**: Updated to wait for previous month instead of previous week
- **Before**: `execution_delta=timedelta(weeks=1)`
- **After**: `execution_delta=timedelta(days=30)`
- **Additional**: Updated `SafeExternalTaskSensor` to use `relativedelta(months=1)`

### 3. **Training Data Window** ✅
- **File**: `dags/dag_functions.py`
- **Function**: `get_training_data_window()`
- **Change**: Changed from 50 weeks to 12 months
- **Before**: `training_end_date = current_date - timedelta(weeks=1)`
- **After**: `training_end_date = current_date - relativedelta(months=1)`
- **Before**: `training_start_date = training_end_date - timedelta(weeks=49)`
- **After**: `training_start_date = training_end_date - relativedelta(months=11)`

### 4. **Function Names** ✅
- **File**: `dags/dag_functions.py`
- **Changes**:
  - `train_lightgbm_weekly()` → `train_lightgbm_monthly()`
  - `train_catboost_weekly()` → `train_catboost_monthly()`
  - `select_best_model_weekly()` → `select_best_model_monthly()`
  - `register_model_weekly()` → `register_model_monthly()`
  - `prepare_training_data_weekly()` → `prepare_training_data_monthly()`

### 5. **Task Names** ✅
- **File**: `dags/dag.py`
- **Changes**:
  - `train_lightgbm_weekly_task` → `train_lightgbm_monthly_task`
  - `train_catboost_weekly_task` → `train_catboost_monthly_task`
  - `select_best_model_weekly_task` → `select_best_model_monthly_task`
  - `register_model_weekly_task` → `register_model_monthly_task`
  - `run_weekly_lifecycle_flow` → `run_monthly_lifecycle_flow`

### 6. **XCom Keys** ✅
- **File**: `dags/dag_functions.py`
- **Changes**:
  - `lightgbm_run_id_weekly` → `lightgbm_run_id_monthly`
  - `catboost_run_id_weekly` → `catboost_run_id_monthly`
  - `lightgbm_macro_f1_weekly` → `lightgbm_macro_f1_monthly`
  - `catboost_macro_f1_weekly` → `catboost_macro_f1_monthly`
  - `best_run_id_weekly` → `best_run_id_monthly`
  - `best_model_type_weekly` → `best_model_type_monthly`
  - `best_macro_f1_weekly` → `best_macro_f1_monthly`

### 7. **Processing Mode** ✅
- **File**: `dag.py`
- **Change**: Updated bronze table processing mode from 'weekly' to 'monthly'
- **Before**: `op_args=[..., 'weekly']`
- **After**: `op_args=[..., 'monthly']`

### 8. **Comments and Documentation** ✅
- **Files**: `dags/dag.py`, `dags/dag_functions.py`
- **Changes**: Updated all comments and docstrings from "weekly" to "monthly"

### 9. **Retraining Tracker** ✅
- **File**: `dags/dag_functions.py`
- **Change**: Updated retraining type from 'weekly' to 'monthly'

## Dependencies Added

### 1. **dateutil.relativedelta** ✅
- **Purpose**: For proper month-based date calculations
- **Added to**: `dags/dag.py`, `dags/dag_functions.py`
- **Usage**: `from dateutil.relativedelta import relativedelta`

## Files Modified

1. **`dags/dag_functions.py`** - Core function updates
2. **`dags/dag.py`** - DAG structure and task updates
3. **`dag.py`** - Main DAG file updates

## Files NOT Modified (Still Contain Weekly References)

The following files still contain weekly references but are not part of the main DAG execution:

1. **`scripts/utils/weekly_evaluation.py`** - Legacy evaluation script
2. **`scripts/utils/query_model_performance.py`** - Query utilities
3. **`scripts/utils/model_inference_utils.py`** - Utility functions (function names unchanged)
4. **`scripts/utils/LightGBM_training_run.py`** - Training script
5. **`scripts/utils/CatBoost_training_run.py`** - Training script

## Testing Recommendations

### 1. **Backfill Testing**
```bash
# Test monthly execution
airflow dags backfill Assignment_2 --start-date 2023-01-01 --end-date 2023-03-01
```

### 2. **Data Availability Check**
- Verify that 12 months of data are available for training
- Check that monthly partitions exist in the datamart

### 3. **Model Training Verification**
- Confirm that training scripts work with monthly data windows
- Verify MLflow run creation and model registration

### 4. **Dependency Chain Testing**
- Test that external task sensor works correctly with monthly intervals
- Verify that retraining triggers work as expected

## Potential Issues to Monitor

### 1. **Data Volume**
- **Issue**: Monthly data might be larger than weekly data
- **Mitigation**: Monitor memory usage and processing times

### 2. **Training Frequency**
- **Issue**: Less frequent retraining might impact model performance
- **Mitigation**: Monitor model drift and adjust retraining triggers if needed

### 3. **Dependency Timing**
- **Issue**: Monthly dependencies might cause longer delays
- **Mitigation**: Ensure proper error handling and retry mechanisms

## Rollback Plan

If issues arise, you can rollback by:

1. **Reverting function names** in `dags/dag_functions.py`
2. **Reverting task names** in `dags/dag.py`
3. **Reverting time calculations** to use weeks instead of months
4. **Updating XCom keys** back to weekly versions

## Next Steps

1. **Deploy changes** to staging environment
2. **Run test executions** with monthly schedule
3. **Monitor performance** and resource usage
4. **Validate model quality** with monthly retraining
5. **Update documentation** and team training materials

## Summary

The migration from weekly to monthly execution has been completed with the following key changes:

- ✅ **Schedule**: Already monthly
- ✅ **Training Window**: 50 weeks → 12 months
- ✅ **Function Names**: All updated to monthly
- ✅ **Task Names**: All updated to monthly
- ✅ **Dependencies**: Updated to wait for previous month
- ✅ **Processing Mode**: Updated to monthly
- ✅ **Documentation**: Updated throughout

The pipeline is now configured for monthly execution while maintaining all the same functionality and error handling mechanisms. 