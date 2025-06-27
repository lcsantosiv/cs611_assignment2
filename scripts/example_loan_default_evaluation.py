"""
Example script demonstrating comprehensive loan default model evaluation.
This shows how to use the recommended metrics for loan default prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import sys
import os

# Add the utils directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from model_evaluation_metrics import (
    evaluate_loan_default_model, 
    print_evaluation_summary, 
    plot_model_performance
)

def generate_sample_loan_data(n_samples=10000, default_rate=0.15):
    """
    Generate sample loan data for demonstration purposes.
    """
    np.random.seed(42)
    
    # Generate features
    data = {
        'credit_score': np.random.normal(650, 100, n_samples),
        'income': np.random.lognormal(10.5, 0.5, n_samples),
        'debt_to_income': np.random.beta(2, 5, n_samples) * 100,
        'loan_amount': np.random.lognormal(10, 0.3, n_samples),
        'employment_length': np.random.exponential(5, n_samples),
        'number_of_accounts': np.random.poisson(8, n_samples),
        'delinquencies_2y': np.random.poisson(0.5, n_samples),
        'inquiries_6m': np.random.poisson(1, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create a realistic default probability based on features
    default_prob = (
        -0.01 * df['credit_score'] + 
        0.02 * df['debt_to_income'] + 
        0.001 * df['loan_amount'] + 
        -0.1 * df['employment_length'] + 
        0.05 * df['delinquencies_2y'] + 
        0.1 * df['inquiries_6m'] +
        np.random.normal(0, 0.5, n_samples)
    )
    
    # Apply sigmoid to get probabilities between 0 and 1
    default_prob = 1 / (1 + np.exp(-default_prob))
    
    # Generate actual defaults based on probability
    defaults = np.random.binomial(1, default_prob)
    
    # Adjust to match target default rate
    current_rate = defaults.mean()
    if current_rate > default_rate:
        # Remove some defaults randomly
        default_indices = np.where(defaults == 1)[0]
        to_remove = int(len(default_indices) * (1 - default_rate / current_rate))
        remove_indices = np.random.choice(default_indices, to_remove, replace=False)
        defaults[remove_indices] = 0
    else:
        # Add some defaults randomly
        non_default_indices = np.where(defaults == 0)[0]
        to_add = int(len(non_default_indices) * (default_rate / (1 - current_rate) - 1))
        add_indices = np.random.choice(non_default_indices, to_add, replace=False)
        defaults[add_indices] = 1
    
    df['default'] = defaults
    
    return df

def train_sample_model(X_train, y_train, X_test, y_test):
    """
    Train a simple Random Forest model for demonstration.
    """
    print("Training Random Forest model...")
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_binary = model.predict(X_test)
    
    return model, y_pred_proba, y_pred_binary

def main():
    """
    Main function demonstrating comprehensive loan default evaluation.
    """
    print("=" * 60)
    print("LOAN DEFAULT PREDICTION - COMPREHENSIVE EVALUATION")
    print("=" * 60)
    
    # Generate sample data
    print("\n1. Generating sample loan data...")
    df = generate_sample_loan_data(n_samples=10000, default_rate=0.15)
    print(f"   Generated {len(df)} loan records")
    print(f"   Default rate: {df['default'].mean():.2%}")
    
    # Prepare features and target
    feature_cols = ['credit_score', 'income', 'debt_to_income', 'loan_amount', 
                   'employment_length', 'number_of_accounts', 'delinquencies_2y', 'inquiries_6m']
    X = df[feature_cols]
    y = df['default']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # Train model
    print("\n2. Training model...")
    model, y_pred_proba, y_pred_binary = train_sample_model(X_train, y_train, X_test, y_test)
    
    # Evaluate with different thresholds
    print("\n3. Evaluating model performance...")
    
    thresholds = [0.3, 0.5, 0.7]
    
    for threshold in thresholds:
        print(f"\n--- Evaluation with threshold {threshold} ---")
        results = evaluate_loan_default_model(y_test, y_pred_proba, threshold=threshold)
        print_evaluation_summary(results)
    
    # Find optimal threshold (you can customize this based on business needs)
    print("\n4. Finding optimal threshold...")
    
    # Example: Find threshold that maximizes F1-score
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in np.arange(0.1, 0.9, 0.05):
        results = evaluate_loan_default_model(y_test, y_pred_proba, threshold=threshold)
        if results['f1_score'] > best_f1:
            best_f1 = results['f1_score']
            best_threshold = threshold
    
    print(f"   Optimal threshold (max F1): {best_threshold:.2f}")
    print(f"   Best F1-score: {best_f1:.4f}")
    
    # Final evaluation with optimal threshold
    print(f"\n5. Final evaluation with optimal threshold ({best_threshold:.2f})...")
    final_results = evaluate_loan_default_model(y_test, y_pred_proba, threshold=best_threshold)
    print_evaluation_summary(final_results)
    
    # Create performance plots
    print("\n6. Generating performance plots...")
    plot_model_performance(y_test, y_pred_proba)
    
    # Business interpretation
    print("\n7. Business Interpretation:")
    print("   📊 ROC AUC: Measures the model's ability to rank loans by risk")
    print("   🎯 KS Statistic: Shows separation between good and bad loan distributions")
    print("   📈 Lift: Indicates how much better the model is vs random selection")
    print("   💰 Cost Analysis: Helps optimize threshold based on business costs")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main() 