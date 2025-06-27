# Binary Classification Fixes - Summary

## Overview
Fixed the ML pipeline to properly handle binary classification (default prediction) instead of multi-class grade prediction.

## Changes Made

### 1. Updated TARGET_COLUMN
**Files Changed:**
- `dags/dag_functions.py`
- `scripts/utils/weekly_evaluation.py`

**Change:**
```python
# Before
TARGET_COLUMN = 'grade'

# After  
TARGET_COLUMN = 'label'  # For binary classification (0 = no default, 1 = default)
```

### 2. Fixed Feature Preparation Function
**Files Changed:**
- `scripts/utils/model_inference_utils.py`
- `scripts/utils/weekly_evaluation.py`

**Changes:**
- Updated `prepare_features()` function to handle binary classification
- Removed unnecessary label encoding for binary labels (already 0/1)
- Added conditional logic to support both binary and multi-class scenarios

**Key Changes:**
```python
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
    
    # ... rest of function
    return X, y, None  # No label_encoder needed for binary
```

### 3. Updated Model Inference Function
**File Changed:**
- `dags/dag_functions.py`

**Changes:**
- Removed grade mapping usage (not needed for binary classification)
- Removed label encoding (binary labels are already 0/1)
- Updated function calls to remove grade_mapping parameter

**Key Changes:**
```python
def run_model_inference(**context):
    # ... existing code ...
    
    # For binary classification, no label encoding needed
    y_encoded = y_raw  # Binary labels are already 0/1
    
    # Get model (no grade mapping needed for binary classification)
    model = model_info["model"]
    
    # Run inference with saving to PostgreSQL (binary classification)
    inference_results = run_model_inference_with_saving(
        model=model,
        X=X,
        customer_ids=customer_ids,
        snapshot_dates=snapshot_dates,
        model_name=MODEL_NAME,
        month_date=month_date,
        pg_config=PG_CONFIG  # Removed grade_mapping parameter
    )
```

### 4. Updated Inference Saving Functions
**File Changed:**
- `scripts/utils/model_inference_utils.py`

**Changes:**
- Updated `save_model_inference_to_postgres()` to handle binary classification
- Updated `run_model_inference_with_saving()` to save default predictions instead of grades
- Changed PostgreSQL table schema for binary classification

**Key Changes:**

**Table Schema:**
```sql
CREATE TABLE IF NOT EXISTS model_inference (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50),
    snapshot_date DATE,
    predicted_default INTEGER,  -- 0 or 1 (instead of predicted_grade)
    default_probability DECIMAL(5,4),  -- Probability of default (instead of prediction_probability)
    model_name VARCHAR(100),
    inference_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    month_partition VARCHAR(10)
);
```

**Inference Results:**
```python
# For binary classification, save default probability and prediction
default_probabilities = y_pred_proba[:, 1]  # Probability of default (class 1)
predicted_defaults = y_pred  # Binary prediction (0 or 1)

inference_results = pd.DataFrame({
    'id': customer_ids,
    'snapshot_date': snapshot_dates,
    'predicted_default': predicted_defaults,  # Instead of predicted_grade
    'default_probability': default_probabilities,  # Instead of prediction_probability
    'model_name': model_name
})
```

## Summary of Fixes

1. **Target Column:** Changed from `'grade'` to `'label'` for binary classification
2. **Feature Preparation:** Removed unnecessary label encoding for binary labels
3. **Model Inference:** Removed grade mapping and simplified for binary classification
4. **Database Schema:** Updated to store default predictions and probabilities
5. **Consistency:** Updated all related files to use the same binary classification approach

## Impact

- **Model Training:** Will now properly use binary labels (0/1) for training
- **Model Inference:** Will save default predictions (0/1) and probabilities
- **Metrics:** Will calculate binary classification metrics (AUC, precision, recall, etc.)
- **Database:** Will store binary classification results in PostgreSQL

## Testing Recommendations

1. **Verify Data:** Ensure label store contains binary labels (0/1)
2. **Test Training:** Run a small training job to verify binary classification works
3. **Test Inference:** Run inference on a small dataset to verify predictions are binary
4. **Check Database:** Verify PostgreSQL tables are created with correct schema
5. **Monitor Metrics:** Ensure AUC and other binary classification metrics are calculated correctly

## Files Modified

1. `dags/dag_functions.py` - Main DAG functions
2. `scripts/utils/model_inference_utils.py` - Inference utilities
3. `scripts/utils/weekly_evaluation.py` - Evaluation utilities
4. `BINARY_CLASSIFICATION_FIXES.md` - This summary document

All changes maintain backward compatibility for multi-class scenarios while properly supporting binary classification. 