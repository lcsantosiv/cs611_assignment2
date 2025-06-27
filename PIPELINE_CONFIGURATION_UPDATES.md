# Pipeline Configuration Updates Summary

## Overview
This document summarizes the configuration updates made to align the ML pipeline with the user's requirements:
- Start date: January 1, 2023
- Initial model training: June 1, 2024
- Training data window: 12 months
- Correct parquet file paths

## 1. Start Date Configuration

### Files Modified:
- `dags/dag.py`

### Changes:
- **Before**: `start_date: datetime(2022, 12, 25)`
- **After**: `start_date: datetime(2023, 1, 1)`

### Impact:
- Pipeline will start processing from January 1, 2023
- All runs before 2023 will be skipped
- Ensures proper historical data handling

## 2. Initial Training Date

### Files Modified:
- `dags/dag_functions.py`

### Changes:
- **Before**: Initial training triggered on `datetime(2023, 1, 1)`
- **After**: Initial training triggered on `datetime(2024, 6, 1)`

### Impact:
- Initial model training will occur on June 1, 2024
- Uses 12 months of data ending in May 2024 for training
- Ensures sufficient historical data is available

## 3. Training Data Window

### Files Modified:
- `scripts/utils/LightGBM_training_run.py`
- `scripts/utils/CatBoost_training_run.py`
- `scripts/utils/model_operations.py`

### Changes:
- **Before**: 50 weeks of data
- **After**: 12 months of data
- Updated function names and parameters from `weeks` to `months`
- Updated chunking logic (6 months per chunk instead of 12 weeks)

### Impact:
- More appropriate data window for monthly pipeline
- Better memory management with monthly chunks
- Consistent with monthly execution schedule

## 4. Parquet File Paths

### Files Modified:
- `dags/dag_functions.py`
- `scripts/utils/model_operations.py`
- `scripts/utils/LightGBM_training_run.py`
- `scripts/utils/CatBoost_training_run.py`

### Changes:
- **Before**: `/opt/airflow/datamart/gold/feature_store`
- **After**: `/opt/airflow/scripts/datamart/gold/feature_store`

### File Naming Convention Handling:
The pipeline now handles both naming conventions:
- **Newer format**: `gold_feature_store_YYYY_MM_01.parquet`
- **Older format**: `YYYY_MM_01.parquet`

### Impact:
- Correct file paths for Docker container
- Handles inconsistent file naming in the datamart
- Robust data loading with fallback options

## 5. Pipeline Flow Summary

### Timeline:
1. **Jan 1, 2023 - May 31, 2024**: Data preprocessing only (skip ML)
2. **Jun 1, 2024**: Initial model training with 12 months of data (May 2023 - Apr 2024)
3. **Jul 1, 2024 onwards**: Monthly lifecycle (evaluation → retraining decision → inference)

### Data Window for Initial Training:
- **End Date**: May 1, 2024
- **Start Date**: June 1, 2023 (12 months back)
- **Data Range**: June 2023 - April 2024

### Monthly Retraining:
- **Trigger**: 90 days since last retraining
- **Data Window**: 12 months ending on previous month
- **Example**: For July 2024 retraining, uses June 2023 - May 2024 data

## 6. File Structure Verification

### Expected Parquet Files:
```
scripts/datamart/gold/feature_store/
├── gold_feature_store_2024_05_01.parquet/
├── gold_feature_store_2024_04_01.parquet/
├── ...
└── 2023_06_01.parquet/

scripts/datamart/gold/label_store/
├── gold_label_store_2024_05_01.parquet/
├── gold_label_store_2024_04_01.parquet/
├── ...
└── 2023_06_01.parquet/
```

### Data Availability Check:
The pipeline will automatically check for data availability and use the appropriate naming convention for each month.

## 7. Testing Recommendations

1. **Verify Data Availability**: Ensure parquet files exist for the 12-month training window
2. **Test Initial Training**: Run a backfill for June 1, 2024 to test initial training
3. **Test Monthly Flow**: Run a backfill for July 1, 2024 to test monthly lifecycle
4. **Monitor File Paths**: Check logs for any file path issues during data loading

## 8. Configuration Summary

| Configuration | Value |
|---------------|-------|
| Start Date | January 1, 2023 |
| Initial Training | June 1, 2024 |
| Training Window | 12 months |
| Retraining Trigger | 90 days |
| Feature Store Path | `/opt/airflow/scripts/datamart/gold/feature_store` |
| Label Store Path | `/opt/airflow/scripts/datamart/gold/label_store` |
| MLflow Experiment | `assignment2` |

All configurations are now aligned with the user's requirements and should work correctly with the monthly pipeline execution. 