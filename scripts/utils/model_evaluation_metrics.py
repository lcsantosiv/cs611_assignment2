import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    precision_score, recall_score, f1_score, confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_ks_statistic(y_true, y_pred_proba):
    """
    Calculate Kolmogorov-Smirnov statistic for credit risk models.
    KS measures the maximum separation between good and bad loan distributions.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    ks_statistic = max(tpr - fpr)
    return ks_statistic

def calculate_gini_coefficient(y_true, y_pred_proba):
    """
    Calculate Gini coefficient (related to AUC: Gini = 2*AUC - 1).
    Commonly used in credit scoring.
    """
    auc_score = roc_auc_score(y_true, y_pred_proba)
    gini = 2 * auc_score - 1
    return gini

def calculate_lift_at_percentile(y_true, y_pred_proba, percentile=10):
    """
    Calculate lift at a given percentile.
    Shows how much better the model performs compared to random selection.
    """
    # Sort by predicted probability in descending order
    df = pd.DataFrame({'y_true': y_true, 'y_pred_proba': y_pred_proba})
    df = df.sort_values('y_pred_proba', ascending=False)
    
    # Calculate the number of samples in the top percentile
    n_samples = len(df)
    n_top = int(n_samples * percentile / 100)
    
    # Calculate default rate in top percentile vs overall
    default_rate_top = df.head(n_top)['y_true'].mean()
    default_rate_overall = df['y_true'].mean()
    
    lift = default_rate_top / default_rate_overall if default_rate_overall > 0 else 0
    return lift

def evaluate_loan_default_model(y_true, y_pred_proba, y_pred_binary=None, threshold=0.5):
    """
    Comprehensive evaluation for loan default prediction models.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels (0 = no default, 1 = default)
    y_pred_proba : array-like
        Predicted probabilities
    y_pred_binary : array-like, optional
        Binary predictions (if not provided, will be calculated using threshold)
    threshold : float, default=0.5
        Threshold for converting probabilities to binary predictions
    
    Returns:
    --------
    dict : Dictionary containing all evaluation metrics
    """
    
    if y_pred_binary is None:
        y_pred_binary = (y_pred_proba >= threshold).astype(int)
    
    # Basic classification metrics
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    
    # AUC metrics
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    # Precision-Recall AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall_curve, precision_curve)
    
    # Credit risk specific metrics
    ks_statistic = calculate_ks_statistic(y_true, y_pred_proba)
    gini_coefficient = calculate_gini_coefficient(y_true, y_pred_proba)
    
    # Lift metrics
    lift_10 = calculate_lift_at_percentile(y_true, y_pred_proba, 10)
    lift_20 = calculate_lift_at_percentile(y_true, y_pred_proba, 20)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    
    # Additional business metrics
    total_loans = len(y_true)
    default_rate = np.mean(y_true)
    approval_rate = np.mean(y_pred_binary)
    
    # Cost metrics (example - adjust based on your business)
    # Assuming: cost of false positive (rejecting good loan) = $100
    #           cost of false negative (approving bad loan) = $1000
    fp_cost = 100
    fn_cost = 1000
    total_cost = fp * fp_cost + fn * fn_cost
    
    results = {
        # Classification metrics
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        
        # Credit risk metrics
        'ks_statistic': ks_statistic,
        'gini_coefficient': gini_coefficient,
        'lift_at_10_percentile': lift_10,
        'lift_at_20_percentile': lift_20,
        
        # Confusion matrix
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
        
        # Business metrics
        'total_loans': total_loans,
        'default_rate': default_rate,
        'approval_rate': approval_rate,
        'total_cost': total_cost,
        
        # Threshold used
        'threshold': threshold
    }
    
    return results

def plot_model_performance(y_true, y_pred_proba, save_path=None):
    """
    Create comprehensive plots for loan default model evaluation.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 0].set_xlim([0.0, 1.0])
    axes[0, 0].set_ylim([0.0, 1.05])
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].grid(True)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    axes[0, 1].plot(recall, precision, color='blue', lw=2,
                    label=f'PR curve (AUC = {pr_auc:.3f})')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curve')
    axes[0, 1].legend(loc="lower left")
    axes[0, 1].grid(True)
    
    # Score Distribution
    df = pd.DataFrame({'y_true': y_true, 'y_pred_proba': y_pred_proba})
    good_loans = df[df['y_true'] == 0]['y_pred_proba']
    bad_loans = df[df['y_true'] == 1]['y_pred_proba']
    
    axes[1, 0].hist(good_loans, bins=50, alpha=0.7, label='Good Loans', color='green')
    axes[1, 0].hist(bad_loans, bins=50, alpha=0.7, label='Bad Loans', color='red')
    axes[1, 0].set_xlabel('Predicted Probability')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Score Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Lift Chart
    df_sorted = df.sort_values('y_pred_proba', ascending=False)
    percentiles = np.arange(1, 101)
    lifts = []
    
    for p in percentiles:
        n_samples = int(len(df_sorted) * p / 100)
        if n_samples > 0:
            default_rate_top = df_sorted.head(n_samples)['y_true'].mean()
            default_rate_overall = df_sorted['y_true'].mean()
            lift = default_rate_top / default_rate_overall if default_rate_overall > 0 else 0
            lifts.append(lift)
        else:
            lifts.append(0)
    
    axes[1, 1].plot(percentiles, lifts, color='purple', lw=2)
    axes[1, 1].axhline(y=1, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Percentile')
    axes[1, 1].set_ylabel('Lift')
    axes[1, 1].set_title('Lift Chart')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def print_evaluation_summary(results):
    """
    Print a formatted summary of evaluation results.
    """
    print("=" * 60)
    print("LOAN DEFAULT MODEL EVALUATION SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 CLASSIFICATION METRICS:")
    print(f"   ROC AUC:           {results['roc_auc']:.4f}")
    print(f"   Precision-Recall AUC: {results['pr_auc']:.4f}")
    print(f"   Precision:         {results['precision']:.4f}")
    print(f"   Recall:            {results['recall']:.4f}")
    print(f"   F1-Score:          {results['f1_score']:.4f}")
    
    print(f"\n🎯 CREDIT RISK METRICS:")
    print(f"   KS Statistic:      {results['ks_statistic']:.4f}")
    print(f"   Gini Coefficient:  {results['gini_coefficient']:.4f}")
    print(f"   Lift at 10%:       {results['lift_at_10_percentile']:.2f}x")
    print(f"   Lift at 20%:       {results['lift_at_20_percentile']:.2f}x")
    
    print(f"\n📈 BUSINESS METRICS:")
    print(f"   Total Loans:       {results['total_loans']:,}")
    print(f"   Default Rate:      {results['default_rate']:.2%}")
    print(f"   Approval Rate:     {results['approval_rate']:.2%}")
    print(f"   Total Cost:        ${results['total_cost']:,.0f}")
    
    print(f"\n🔍 CONFUSION MATRIX:")
    print(f"   True Negatives:    {results['true_negatives']:,}")
    print(f"   False Positives:   {results['false_positives']:,}")
    print(f"   False Negatives:   {results['false_negatives']:,}")
    print(f"   True Positives:    {results['true_positives']:,}")
    
    print(f"\n⚙️  MODEL SETTINGS:")
    print(f"   Threshold:         {results['threshold']:.3f}")
    
    print("=" * 60) 