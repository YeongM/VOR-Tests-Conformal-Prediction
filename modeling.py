import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score, auc, PrecisionRecallDisplay, roc_curve
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import re
from itertools import cycle
import os
import joblib
from datetime import datetime
import time
from scipy import stats

def preprocess_data(df, seed=42):
    df = df.dropna(subset=['Label'])
    
    X = df.drop('Label', axis=1)
    y = df['Label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    
    return X_train, X_test, y_train, y_test

# AUC calculation functions with additional weighted option
def calculate_multiclass_auc(y_true, y_score, average='macro'):
    from sklearn.preprocessing import LabelEncoder
    
    # Encode labels if they aren't already
    le = LabelEncoder()
    y_true_encoded = le.fit_transform(y_true)
    n_classes = len(np.unique(y_true_encoded))
    
    # Convert to one-hot encoding for micro/macro AUC
    y_true_onehot = np.zeros((len(y_true_encoded), n_classes))
    for i, val in enumerate(y_true_encoded):
        y_true_onehot[i, val] = 1
    
    # Calculate class-wise AUC scores
    class_aucs = []
    for i in range(n_classes):
        y_true_binary = (y_true_encoded == i).astype(int)
        y_score_binary = y_score[:, i]
        try:
            class_auc = roc_auc_score(y_true_binary, y_score_binary)
            class_aucs.append(class_auc)
        except:
            class_aucs.append(np.nan)
    
    # Calculate average AUC based on specified method
    try:
        if average == 'macro':
            # Simple average of class AUCs
            avg_auc = np.nanmean(class_aucs)
        elif average == 'micro':
            # Calculate AUC after flattening predictions and targets
            avg_auc = roc_auc_score(y_true_onehot, y_score, average='micro')
        elif average == 'weighted':
            # Weighted average based on class frequency
            class_weights = np.bincount(y_true_encoded) / len(y_true_encoded)
            valid_indices = ~np.isnan(class_aucs)
            if np.any(valid_indices):
                avg_auc = np.average(
                    np.array(class_aucs)[valid_indices], 
                    weights=class_weights[valid_indices]
                )
            else:
                avg_auc = 0
        else:
            avg_auc = np.nanmean(class_aucs)  # Default to macro if invalid
    except Exception as e:
        print(f"Error calculating {average} AUC: {e}")
        avg_auc = 0
    
    return avg_auc, class_aucs

# Enhanced PR-AUC calculation function
def calculate_precision_recall_metrics(y_true, y_score):
    from sklearn.preprocessing import LabelEncoder
    
    # Encode labels if they aren't already
    le = LabelEncoder()
    y_true_encoded = le.fit_transform(y_true)
    n_classes = len(np.unique(y_true_encoded))
    
    # Convert to one-hot encoding for micro/macro calculations
    y_true_onehot = np.zeros((len(y_true_encoded), n_classes))
    for i, val in enumerate(y_true_encoded):
        y_true_onehot[i, val] = 1
    
    # Calculate per-class PR-AUC
    class_ap = {}
    for i in range(n_classes):
        y_true_binary = (y_true_encoded == i).astype(int)
        y_score_binary = y_score[:, i]
        try:
            ap = average_precision_score(y_true_binary, y_score_binary)
            class_ap[i] = ap
        except Exception as e:
            print(f"Error calculating AP for class {i}: {e}")
            class_ap[i] = 0
    
    # Calculate micro and macro average PR-AUC
    try:
        micro_ap = average_precision_score(y_true_onehot, y_score, average='micro')
    except Exception as e:
        print(f"Error calculating micro AP: {e}")
        micro_ap = 0
    
    try:
        macro_ap = average_precision_score(y_true_onehot, y_score, average='macro')
    except Exception as e:
        print(f"Error calculating macro AP: {e}")
        macro_ap = 0
    
    # Calculate weighted average PR-AUC
    try:
        class_weights = np.bincount(y_true_encoded) / len(y_true_encoded)
        weighted_ap = np.average(
            [class_ap[i] for i in range(n_classes)], 
            weights=class_weights
        )
    except Exception as e:
        print(f"Error calculating weighted AP: {e}")
        weighted_ap = 0
    
    return {
        'class_ap': class_ap,
        'micro_ap': micro_ap,
        'macro_ap': macro_ap,
        'weighted_ap': weighted_ap
    }

# Enhanced PRC visualization function
def plot_precision_recall_curve(y_true, y_score, model_name):
    """
    Calculate and visualize precision-recall curves with all averaging methods
    """
    from sklearn.preprocessing import LabelEncoder
    
    # Encode labels if they aren't already
    le = LabelEncoder()
    y_true_encoded = le.fit_transform(y_true)
    n_classes = len(np.unique(y_true_encoded))
    
    # Calculate precision-recall curves for each class
    precision = {}
    recall = {}
    average_precision = {}
    
    for i in range(n_classes):
        y_true_binary = (y_true_encoded == i).astype(int)
        y_score_binary = y_score[:, i]
        
        try:
            precision[i], recall[i], _ = precision_recall_curve(y_true_binary, y_score_binary)
            average_precision[i] = average_precision_score(y_true_binary, y_score_binary)
        except Exception as e:
            print(f"Error calculating PRC for class {i}: {e}")
            precision[i], recall[i] = [], []
            average_precision[i] = 0
    
    # Convert to one-hot encoding for micro/macro calculations
    y_true_onehot = np.zeros((len(y_true_encoded), n_classes))
    for i, val in enumerate(y_true_encoded):
        y_true_onehot[i, val] = 1
    
    # Calculate micro and macro average PR-AUC
    try:
        micro_ap = average_precision_score(y_true_onehot, y_score, average='micro')
        macro_ap = average_precision_score(y_true_onehot, y_score, average='macro')
        
        # Calculate weighted average
        class_weights = np.bincount(y_true_encoded) / len(y_true_encoded)
        weighted_ap = np.average(
            [average_precision[i] for i in range(n_classes) if i in average_precision], 
            weights=class_weights
        )
    except Exception as e:
        print(f"Error calculating average AP: {e}")
        micro_ap, macro_ap, weighted_ap = 0, 0, 0
    
    # Create plot
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'green', 'red'])
    labels = ['Normal', 'Abnormal Right', 'Abnormal Left', 'Abnormal Unknown Direction']
    
    # Plot per-class PR curves
    for i, color in zip(range(n_classes), colors):
        if len(precision[i]) > 0 and len(recall[i]) > 0:
            display = PrecisionRecallDisplay(
                recall=recall[i],
                precision=precision[i],
                average_precision=average_precision[i],
            )
    
    # Plot micro-average PR curve
    try:
        micro_precision, micro_recall, _ = precision_recall_curve(
            y_true_onehot.ravel(), y_score.ravel()
        )
        display = PrecisionRecallDisplay(
            recall=micro_recall,
            precision=micro_precision,
            average_precision=micro_ap
        )
    except Exception as e:
        print(f"Error plotting micro-average PRC: {e}")
    
    # Return all metrics
    return {
        'class_ap': average_precision,
        'micro_ap': micro_ap,
        'macro_ap': macro_ap,
        'weighted_ap': weighted_ap
    }

# Enhanced model evaluation function
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Evaluate a model with enhanced metrics including weighted averages
    """
    if not hasattr(model, 'fit'):
        # Model is already fitted
        print(f"Model {model_name} is already fitted, skipping training.")
    else:
        # Train the model
        model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate probabilities for AUC/PRC metrics
    y_proba = None
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
    
    # Initialize metric variables
    accuracy = accuracy_score(y_test, y_pred)
    auc_metrics = {}
    pr_metrics = {}
    
    # Calculate AUC metrics if probabilities are available
    if y_proba is not None:
        auc_metrics['macro_auc'], class_aucs = calculate_multiclass_auc(y_test, y_proba, 'macro')
        auc_metrics['micro_auc'], _ = calculate_multiclass_auc(y_test, y_proba, 'micro')
        auc_metrics['weighted_auc'], _ = calculate_multiclass_auc(y_test, y_proba, 'weighted')
        
        # Calculate PR metrics
        pr_metrics = calculate_precision_recall_metrics(y_test, y_proba)
    
    # Print evaluation summary
    print(f"===== {model_name} Model Evaluation =====")
    print(f"Accuracy: {accuracy:.4f}")
    
    if y_proba is not None:
        print(f"\nROC Curve Metrics:")
        print(f"  - Macro AUC: {auc_metrics.get('macro_auc', 0):.4f}")
        print(f"  - Micro AUC: {auc_metrics.get('micro_auc', 0):.4f}")
        print(f"  - Weighted AUC: {auc_metrics.get('weighted_auc', 0):.4f}")
        
        print(f"\nPrecision-Recall Curve Metrics:")
        print(f"  - Micro AP: {pr_metrics.get('micro_ap', 0):.4f}")
        print(f"  - Macro AP: {pr_metrics.get('macro_ap', 0):.4f}")
        print(f"  - Weighted AP: {pr_metrics.get('weighted_ap', 0):.4f}")
        
        # Print class-wise metrics
        labels = ['Normal', 'Abnormal Right', 'Abnormal Left', 'Abnormal Unknown Direction']
        print("\nPer-class AUC:")
        for i, auc_val in enumerate(class_aucs):
            print(f"  - {labels[i]}: {auc_val:.4f}")
        
        print("\nPer-class AP (Average Precision):")
        for i in range(len(labels)):
            if i in pr_metrics.get('class_ap', {}):
                print(f"  - {labels[i]}: {pr_metrics['class_ap'][i]:.4f}")
    
    # Create confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred)
    labels = ['Normal', 'Abnormal Right', 'Abnormal Left', 'Abnormal Unknown Direction']
    
    # Visualize PR curves if probabilities available
    if y_proba is not None:
        plot_precision_recall_curve(y_test, y_proba, model_name)
    
    # Create result dictionary
    result = {
        'model': model,
        'accuracy': accuracy,
        'auc_metrics': auc_metrics,
        'pr_metrics': pr_metrics
    }
    
    return result

# Functions for visualizing model performance with confidence intervals
def calculate_95_ci(data):
    """Calculate 95% confidence interval for a data series"""
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)  # Standard error of the mean
    ci_95 = se * stats.t.ppf((1 + 0.95) / 2, n-1)  # 95% CI
    return mean, mean - ci_95, mean + ci_95  # mean, lower bound, upper bound

def plot_aggregate_metrics_with_ci(results_df, output_dir):
    """
    Plot aggregate metrics with 95% confidence intervals across all seeds
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing results for all seeds
    output_dir : str
        Directory to save the plots
    """
    plt.figure(figsize=(15, 20))
    
    # Define metrics to plot
    metric_groups = [
        ["_macro_auc", "_micro_auc", "_weighted_auc"],
        ["_macro_ap", "_micro_ap", "_weighted_ap"]
    ]
    
    metric_titles = [
        "ROC AUC Metrics",
        "PR-AUC (Average Precision) Metrics"
    ]
    
    model_prefixes = ['xgb', 'lgb', 'cat']
    model_names = ['XGBoost', 'LightGBM', 'CatBoost']
    
    for group_idx, metric_group in enumerate(metric_groups):
        plt.subplot(2, 1, group_idx + 1)
        
        # Calculate metrics for each model
        x_positions = np.arange(len(metric_group))
        width = 0.25  # Width of the bars
        
        for i, (prefix, name) in enumerate(zip(model_prefixes, model_names)):
            means = []
            ci_lows = []
            ci_highs = []
            metric_labels = []
            
            for metric_suffix in metric_group:
                col_name = f"{prefix}{metric_suffix}"
                mean, ci_low, ci_high = calculate_95_ci(results_df[col_name])
                means.append(mean)
                ci_lows.append(ci_low)
                ci_highs.append(ci_high)
                
                # Create readable metric labels
                metric_label = metric_suffix.strip('_').replace('_', ' ').title()
                metric_labels.append(metric_label)
            
            # Plot bars with error bars
            plt.bar(x_positions + i*width, means, width, 
                   label=name, yerr=[np.array(means)-np.array(ci_lows), np.array(ci_highs)-np.array(means)],
                   capsize=5, alpha=0.7)
        
        # Set labels and title
        plt.xticks(x_positions + width, metric_labels)
        plt.ylabel("Score")
        plt.title(metric_titles[group_idx])
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.ylim(0.5, 1.0)  # Adjust as needed
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "aggregate_metrics_with_ci.png"), dpi=300)
    plt.show()

# Functions for ROC and PR curves visualization
def plot_roc_curve(y_true, y_score, model_name, output_path=None):
    """Plot ROC curves for a model"""
    from sklearn.preprocessing import LabelEncoder
    
    # Encode labels if they aren't already
    le = LabelEncoder()
    y_true_encoded = le.fit_transform(y_true)
    n_classes = len(np.unique(y_true_encoded))
    
    # Convert to one-hot encoding for micro calculation
    y_true_onehot = np.zeros((len(y_true_encoded), n_classes))
    for i, val in enumerate(y_true_encoded):
        y_true_onehot[i, val] = 1
    
    # Calculate ROC curve and AUC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    plt.figure(figsize=(12, 10))
    
    for i in range(n_classes):
        y_true_binary = (y_true_encoded == i).astype(int)
        y_score_binary = y_score[:, i]
        
        try:
            fpr[i], tpr[i], _ = roc_curve(y_true_binary, y_score_binary)
            roc_auc[i] = roc_auc_score(y_true_binary, y_score_binary)
            
            plt.plot(fpr[i], tpr[i], lw=2,
                    label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
        except Exception as e:
            print(f"Error plotting ROC curve for class {i}: {e}")
    
    # Compute micro-average ROC curve and AUC
    try:
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_onehot.ravel(), y_score.ravel())
        roc_auc["micro"] = roc_auc_score(y_true_onehot, y_score, average="micro")
        plt.plot(fpr["micro"], tpr["micro"], 
                label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})',
                lw=2, linestyle=':', color='deeppink')
    except Exception as e:
        print(f"Error plotting micro-average ROC curve: {e}")
    
    # Plot the diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Set plot properties
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves: {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300)
    
    plt.show()
    
    return roc_auc

# Main function to perform grid search and evaluation for each seed
def run_optimization_for_seeds(df, start_seed=0, end_seed=300, output_dir="models"):
    """
    Run optimization for each seed and save models and datasets
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    start_seed : int
        Starting seed value
    end_seed : int
        Ending seed value (inclusive)
    output_dir : str
        Directory to save models and results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create results dataframe to store metrics
    results_columns = [
        'seed', 
        'xgb_accuracy', 'xgb_macro_auc', 'xgb_micro_auc', 'xgb_weighted_auc', 
                       'xgb_macro_ap', 'xgb_micro_ap', 'xgb_weighted_ap',
        'lgb_accuracy', 'lgb_macro_auc', 'lgb_micro_auc', 'lgb_weighted_auc', 
                       'lgb_macro_ap', 'lgb_micro_ap', 'lgb_weighted_ap',
        'cat_accuracy', 'cat_macro_auc', 'cat_micro_auc', 'cat_weighted_auc', 
                       'cat_macro_ap', 'cat_micro_ap', 'cat_weighted_ap'
    ]
    results_df = pd.DataFrame(columns=results_columns)
    
    # Store best model info for later visualization
    best_model_info = {
        'overall_score': 0,
        'model_type': '',
        'seed': 0,
        'model': None,
        'X_test': None,
        'y_test': None
    }
    
    # Common parameter grids for each model
    xgb_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
    }
    
    lgb_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
    }
    
    cat_param_grid = {
        'iterations': [50, 100, 200],
        'depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    
    # Custom scorer function for grid search that optimizes for overall performance
    def multiclass_combined_scorer(estimator, X, y):
        # Make predictions
        y_pred = estimator.predict(X)
        y_proba = estimator.predict_proba(X)
        
        # Calculate base accuracy
        accuracy = accuracy_score(y, y_pred)
        
        # Calculate AUC metrics
        macro_auc, _ = calculate_multiclass_auc(y, y_proba, 'macro')
        micro_auc, _ = calculate_multiclass_auc(y, y_proba, 'micro')
        weighted_auc, _ = calculate_multiclass_auc(y, y_proba, 'weighted')
        
        # Calculate PR metrics
        pr_metrics = calculate_precision_recall_metrics(y, y_proba)
        
        # Combine all metrics into a single score
        combined_score = (
            accuracy + 
            macro_auc + micro_auc + weighted_auc + 
            pr_metrics['macro_ap'] + pr_metrics['micro_ap'] + pr_metrics['weighted_ap']
        ) / 7
        
        return combined_score
    
    # Time tracking
    total_seeds = end_seed - start_seed + 1
    start_time = time.time()

    # Process each seed
    for i, seed in enumerate(range(start_seed, end_seed + 1)):
        seed_start_time = time.time()
        print(f"\n{'='*70}")
        print(f"Processing seed {seed} ({i+1}/{total_seeds})")
        print(f"{'='*70}")
        
        # Create seed-specific output directory
        seed_dir = os.path.join(output_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        
        # Preprocess data with current seed
        X_train, X_test, y_train, y_test = preprocess_data(df, seed=seed)
        
        # Save train/test datasets
        joblib.dump((X_train, y_train), os.path.join(seed_dir, "train_data.pkl"))
        joblib.dump((X_test, y_test), os.path.join(seed_dir, "test_data.pkl"))
        
        # Initialize models
        xgb_model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=4,
            missing=np.nan,
            random_state=seed
        )
        
        lgb_model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=4,
            random_state=seed,
            verbose=-1
        )
        
        cat_model = CatBoostClassifier(
            loss_function='MultiClass',
            random_seed=seed,
            verbose=0
        )
        
        # Set up models and parameter grids
        models = {
            'XGBoost': {
                'model': xgb_model,
                'param_grid': xgb_param_grid,
                'file_prefix': 'xgb'
            },
            'LightGBM': {
                'model': lgb_model,
                'param_grid': lgb_param_grid,
                'file_prefix': 'lgb'
            },
            'CatBoost': {
                'model': cat_model,
                'param_grid': cat_param_grid,
                'file_prefix': 'cat'
            }
        }
        
        # Initialize results row
        results_row = {'seed': seed}
        
        # Train and evaluate each model
        for model_name, model_info in models.items():
            print(f"\nTraining {model_name} model with GridSearchCV...")
            
            # Create and fit grid search
            grid_search = GridSearchCV(
                estimator=model_info['model'],
                param_grid=model_info['param_grid'],
                cv=5,
                scoring=multiclass_combined_scorer,
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Get best model and parameters
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            # Save best model
            model_path = os.path.join(seed_dir, f"{model_info['file_prefix']}_model.pkl")
            joblib.dump(best_model, model_path)
            
            # Save best parameters
            params_path = os.path.join(seed_dir, f"{model_info['file_prefix']}_params.txt")
            with open(params_path, 'w') as f:
                for param, value in best_params.items():
                    f.write(f"{param}: {value}\n")
            
            # Evaluate model (don't visualize for individual models)
            prefix = model_info['file_prefix']
            print(f"\nEvaluating {model_name} model with best parameters...")
            # Use evaluate_model but prevent visualizations
            with plt.ioff():  # Turn off interactive plotting
                eval_result = evaluate_model(best_model, X_train, X_test, y_train, y_test, model_name)
            
            # Store results
            results_row[f'{prefix}_accuracy'] = eval_result['accuracy']
            
            if 'auc_metrics' in eval_result:
                results_row[f'{prefix}_macro_auc'] = eval_result['auc_metrics'].get('macro_auc', 0)
                results_row[f'{prefix}_micro_auc'] = eval_result['auc_metrics'].get('micro_auc', 0)
                results_row[f'{prefix}_weighted_auc'] = eval_result['auc_metrics'].get('weighted_auc', 0)
            
            if 'pr_metrics' in eval_result:
                results_row[f'{prefix}_macro_ap'] = eval_result['pr_metrics'].get('macro_ap', 0)
                results_row[f'{prefix}_micro_ap'] = eval_result['pr_metrics'].get('micro_ap', 0)
                results_row[f'{prefix}_weighted_ap'] = eval_result['pr_metrics'].get('weighted_ap', 0)
            
            # Check if this model is the best overall so far
            current_score = (
                eval_result['accuracy'] + 
                eval_result['auc_metrics'].get('macro_auc', 0) + 
                eval_result['auc_metrics'].get('micro_auc', 0) + 
                eval_result['auc_metrics'].get('weighted_auc', 0) + 
                eval_result['pr_metrics'].get('macro_ap', 0) + 
                eval_result['pr_metrics'].get('micro_ap', 0) + 
                eval_result['pr_metrics'].get('weighted_ap', 0)
            ) / 7
            
            if current_score > best_model_info['overall_score']:
                best_model_info['overall_score'] = current_score
                best_model_info['model_type'] = model_name
                best_model_info['seed'] = seed
                best_model_info['model'] = best_model
                best_model_info['X_test'] = X_test
                best_model_info['y_test'] = y_test
                print(f"New best model found: {model_name} with seed {seed}, score: {current_score:.4f}")
        
        # Add results row to DataFrame
        results_df = pd.concat([results_df, pd.DataFrame([results_row])], ignore_index=True)
        
        # Save updated results to CSV
        results_df.to_csv(os.path.join(output_dir, "all_seeds_results.csv"), index=False)
        
        # Calculate time remaining
        elapsed_time = time.time() - start_time
        avg_time_per_seed = elapsed_time / (i + 1)
        remaining_seeds = total_seeds - (i + 1)
        estimated_time_remaining = avg_time_per_seed * remaining_seeds
        
        # Print progress
        print(f"\nSeed {seed} completed in {time.time() - seed_start_time:.2f} seconds")
        print(f"Progress: {i+1}/{total_seeds} seeds processed ({(i+1)/total_seeds*100:.1f}%)")
        print(f"Estimated time remaining: {estimated_time_remaining/60:.1f} minutes")
    
    # Final calculations and summary
    print("\n" + "="*70)
    print("All seeds processed successfully!")
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    print("="*70)
    
    # Plot aggregate metrics with CI
    print("\nPlotting aggregate metrics with 95% confidence intervals...")
    plot_aggregate_metrics_with_ci(results_df, output_dir)
    
    # Visualize best overall model
    print(f"\nVisualizing best overall model: {best_model_info['model_type']} with seed {best_model_info['seed']}")
    
    # Create confusion matrix for best model
    plt.figure(figsize=(10, 8))
    y_pred = best_model_info['model'].predict(best_model_info['X_test'])
    cm = confusion_matrix(best_model_info['y_test'], y_pred)
    
    labels = ['Normal', 'Abnormal Right', 'Abnormal Left', 'Abnormal Unknown Direction']
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels,
                    yticklabels=labels,
                    annot_kws={"size": 16})
    
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(2)
    
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('Actual', fontsize=14)
    plt.title(f'Best Model: {best_model_info["model_type"]} (Seed {best_model_info["seed"]}) Confusion Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "best_model_confusion_matrix.png"), dpi=300)
    plt.show()
    
    # Plot ROC curve for best model
    y_proba = best_model_info['model'].predict_proba(best_model_info['X_test'])
    roc_curve_path = os.path.join(output_dir, "best_model_roc_curve.png")
    plot_roc_curve(best_model_info['y_test'], y_proba, 
                  f"Best Model: {best_model_info['model_type']} (Seed {best_model_info['seed']})",
                  roc_curve_path)
    
    # Plot PR curve for best model
    pr_curve_path = os.path.join(output_dir, "best_model_pr_curve.png")
    plot_precision_recall_curve(best_model_info['y_test'], y_proba, 
                               f"Best Model: {best_model_info['model_type']} (Seed {best_model_info['seed']})")
    
    # Calculate average performance across seeds
    print("\nAverage performance across all seeds with 95% confidence intervals:")
    for metric in results_columns[1:]:  # Skip the 'seed' column
        mean, ci_low, ci_high = calculate_95_ci(results_df[metric])
        print(f"{metric}: {mean:.4f} (95% CI: {ci_low:.4f} - {ci_high:.4f})")
    
    # Find best seed for each model
    model_prefixes = ['xgb', 'lgb', 'cat']
    best_seeds = {}
    
    for prefix in model_prefixes:
        # Calculate combined score for each seed (average of all metrics)
        combined_scores = results_df[[col for col in results_df.columns if col.startswith(prefix)]].mean(axis=1)
        best_seed_idx = combined_scores.idxmax()
        best_seed_value = results_df.loc[best_seed_idx, 'seed']
        best_seeds[prefix] = best_seed_value
        
        print(f"\nBest seed for {prefix.upper()}: {best_seed_value}")
        for col in results_df.columns:
            if col.startswith(prefix):
                print(f"  {col}: {results_df.loc[best_seed_idx, col]:.4f}")
    
    return results_df, best_seeds, best_model_info

# Main execution
if __name__ == "__main__":
    df = pd.read_csv('./your data')

    # Set output directory
    output_dir = f"./save_your_dir"

    # Run optimization
    results_df, best_seeds, best_model_info = run_optimization_for_seeds(df, start_seed=1, end_seed=300, output_dir=output_dir)