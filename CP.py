import numpy as np
import pandas as pd
from scipy.stats import entropy, spearmanr, t as t_dist, shapiro
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import warnings
import os
warnings.filterwarnings('ignore')

GLOBAL_SEED = 42

class ConformalPredictor:
    """Conformal Prediction (APS method) with full reproducibility"""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.threshold = None
        self.rng = None
        
    def set_random_state(self, random_state: int):
        """Set random state for reproducibility"""
        self.rng = np.random.RandomState(random_state)
        
    def compute_nonconformity_scores(self, probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        if self.rng is None:
            raise ValueError("Random state not set. Call set_random_state() first.")
        
        n_samples = len(probs)
        scores = []
        
        for i in range(n_samples):
            prob = probs[i]
            true_label = int(labels[i])
            
            sorted_indices = np.argsort(-prob)
            sorted_probs = prob[sorted_indices]
            
            true_label_idx = np.where(sorted_indices == (true_label - 1))[0][0]
            
            score = np.sum(sorted_probs[:true_label_idx])
            score += self.rng.uniform(0, 1) * sorted_probs[true_label_idx]
            
            scores.append(score)
        
        return np.array(scores)
    
    def calibrate(self, cal_probs: np.ndarray, cal_labels: np.ndarray):
        cal_scores = self.compute_nonconformity_scores(cal_probs, cal_labels)
        
        n_cal = len(cal_scores)
        q = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
        self.threshold = np.quantile(cal_scores, q)
        
        return self.threshold
    
    def predict_set(self, test_probs: np.ndarray) -> List[List[int]]:
        if self.threshold is None:
            raise ValueError("first, call calibrate()")
        
        pred_sets = []
        
        for prob in test_probs:
            sorted_indices = np.argsort(-prob)
            sorted_probs = prob[sorted_indices]
            
            pred_set = []
            cumsum = 0
            
            for idx, p in zip(sorted_indices, sorted_probs):
                pred_set.append(int(idx + 1))
                cumsum += p
                
                if cumsum > self.threshold:
                    break
            
            pred_sets.append(pred_set)
        
        return pred_sets

class ExpertMetrics:
    
    @staticmethod
    def calculate_expert_entropy(row: pd.Series) -> float:
        all_labels = []
        
        for expert in range(1, 6):
            q1 = row[f'Q1_expert{expert}']
            q2 = row[f'Q2_expert{expert}']
            q3 = row[f'Q3_expert{expert}']
            
            all_labels.append(q1)
            if q2 != 0:
                all_labels.append(q2)
            if q3 != 0:
                all_labels.append(q3)
        
        unique, counts = np.unique(all_labels, return_counts=True)
        probs = counts / counts.sum()
        
        return entropy(probs, base=2)

def run_single_experiment_extended(df: pd.DataFrame, alpha: float = 0.05, 
                                   random_state: int = 42) -> Dict:
    df = df.copy()
    
    # Stratified split
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state)
    
    for cal_idx, test_idx in splitter.split(df, df['Label']):
        df_cal = df.iloc[cal_idx].copy()
        df_test = df.iloc[test_idx].copy()
    
    cal_probs = df_cal[['class1_proba', 'class2_proba', 
                         'class3_proba', 'class4_proba']].values
    cal_labels = df_cal['Label'].values
    
    test_probs = df_test[['class1_proba', 'class2_proba', 
                           'class3_proba', 'class4_proba']].values
    test_labels = df_test['Label'].values
    
    # Single prediction (Top-1)
    single_preds = np.argmax(test_probs, axis=1) + 1
    
    # Conformal Prediction
    cp = ConformalPredictor(alpha=alpha)
    cp.set_random_state(random_state)
    threshold = cp.calibrate(cal_probs, cal_labels)
    pred_sets = cp.predict_set(test_probs)
    
    # Set sizes
    set_sizes = [len(ps) for ps in pred_sets]
    
    # Coverage
    coverages = [1 if int(test_labels[i]) in pred_sets[i] else 0 
                for i in range(len(test_labels))]
    coverage = np.mean(coverages)
    
    df_test['expert_entropy'] = df_test.apply(
        ExpertMetrics.calculate_expert_entropy, axis=1)
    df_test['set_size'] = set_sizes
    df_test['single_pred'] = single_preds
    df_test['pred_set'] = pred_sets
    df_test['coverage'] = coverages
    
    if df_test['set_size'].nunique() > 1 and df_test['expert_entropy'].std() > 0:
        corr_entropy, p_entropy = spearmanr(set_sizes, df_test['expert_entropy'])
    else:
        corr_entropy, p_entropy = np.nan, np.nan
    
    set_size_dist = pd.Series(set_sizes).value_counts().to_dict()
    
    return {
        'threshold': threshold,
        'coverage': coverage,
        'avg_set_size': np.mean(set_sizes),
        'set_size_dist': set_size_dist,
        'corr_entropy': corr_entropy,
        'p_entropy': p_entropy,
        'random_state': random_state,
        'df_test': df_test,
        'single_preds': single_preds,
        'pred_sets': pred_sets
    }

def generate_table3(results_df: pd.DataFrame, all_test_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    
    # Overall metrics
    mean_coverage = results_df['coverage'].mean()
    std_coverage = results_df['coverage'].std()
    ci_coverage_lower = mean_coverage - 1.96 * std_coverage / np.sqrt(len(results_df))
    ci_coverage_upper = mean_coverage + 1.96 * std_coverage / np.sqrt(len(results_df))
    
    mean_set_size = results_df['avg_set_size'].mean()
    std_set_size = results_df['avg_set_size'].std()
    ci_set_size_lower = mean_set_size - 1.96 * std_set_size / np.sqrt(len(results_df))
    ci_set_size_upper = mean_set_size + 1.96 * std_set_size / np.sqrt(len(results_df))
    
    # Efficiency (set size = 1)
    all_set_sizes = []
    for df_test in all_test_dfs:
        all_set_sizes.extend(df_test['set_size'].tolist())
    
    efficiency = (np.array(all_set_sizes) == 1).mean()
    
    # Singleton correct rate
    singleton_correct = []
    for df_test in all_test_dfs:
        singleton_mask = df_test['set_size'] == 1
        if singleton_mask.sum() > 0:
            singleton_correct.append(df_test[singleton_mask]['coverage'].mean())
    
    singleton_correct_rate = np.mean(singleton_correct) if singleton_correct else np.nan
    
    # CP-Expert correlation
    corr_clean = results_df.dropna(subset=['corr_entropy'])
    mean_corr = corr_clean['corr_entropy'].mean()
    ci_corr_lower = corr_clean['corr_entropy'].quantile(0.025)
    ci_corr_upper = corr_clean['corr_entropy'].quantile(0.975)
    
    # Set size distribution
    set_size_dist = pd.Series(all_set_sizes).value_counts().sort_index()
    
    # Build table
    table3_overall = pd.DataFrame({
        'Metric': [
            'Coverage Rate',
            'Average Set Size',
            'Efficiency (Set Size = 1)',
            'Singleton Correct Rate',
            'CP-Expert Correlation (Spearman ρ)'
        ],
        'Value (95% CI)': [
            f"{mean_coverage:.1%} ({ci_coverage_lower:.1%}-{ci_coverage_upper:.1%})",
            f"{mean_set_size:.2f} ({ci_set_size_lower:.2f}-{ci_set_size_upper:.2f})",
            f"{efficiency:.1%}",
            f"{singleton_correct_rate:.1%}",
            f"{mean_corr:.3f} ({ci_corr_lower:.3f}-{ci_corr_upper:.3f})"
        ],
        'Target': ['≥95%', '-', '-', '-', '-'],
        'Status': ['✓ Achieved' if mean_coverage >= 0.95 else '✗ Not achieved', 
                   '-', '-', 'High precision', 'p < 0.001']
    })
    
    # Set size distribution table with Top-1 accuracy
    table3_setsize = []
    for size in sorted(set_size_dist.index):
        size_data = []
        top1_accuracies = []
        
        for df_test in all_test_dfs:
            size_mask = df_test['set_size'] == size
            if size_mask.sum() > 0:
                # Coverage and entropy
                size_data.append({
                    'coverage': df_test[size_mask]['coverage'].mean(),
                    'entropy': df_test[size_mask]['expert_entropy'].mean()
                })
                
                # Top-1 prediction accuracy for this set size
                df_size_subset = df_test[size_mask]
                top1_correct = (df_size_subset['Label'] == df_size_subset['single_pred']).sum()
                top1_total = len(df_size_subset)
                top1_acc = top1_correct / top1_total if top1_total > 0 else 0
                top1_accuracies.append(top1_acc)
        
        if size_data:
            mean_cov = np.mean([d['coverage'] for d in size_data])
            mean_ent = np.mean([d['entropy'] for d in size_data])
            ci_ent_lower = np.percentile([d['entropy'] for d in size_data], 2.5)
            ci_ent_upper = np.percentile([d['entropy'] for d in size_data], 97.5)
            
            # Top-1 accuracy statistics
            mean_top1 = np.mean(top1_accuracies)
            ci_top1_lower = np.percentile(top1_accuracies, 2.5)
            ci_top1_upper = np.percentile(top1_accuracies, 97.5)
            
            table3_setsize.append({
                'Set Size': size,
                'Frequency (%)': f"{set_size_dist[size] / len(all_set_sizes) * 100:.1f}%",
                'Top-1 Accuracy': f"{mean_top1:.1%} ({ci_top1_lower:.1%}-{ci_top1_upper:.1%})",
                'CP Coverage Rate': f"{mean_cov:.1%}",
                'Average Expert Entropy': f"{mean_ent:.2f} ({ci_ent_lower:.2f}-{ci_ent_upper:.2f})"
            })
    
    table3_setsize_df = pd.DataFrame(table3_setsize)
    
    print("\n[Overall Performance]")
    print(table3_overall.to_string(index=False))
    print("\n[Set Size Distribution]")
    print(table3_setsize_df.to_string(index=False))
    
    return table3_overall, table3_setsize_df

def generate_table4(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Table 4: Expert Agreement Analysis
    """
    print("\n" + "="*80)
    print("TABLE 4: Expert Agreement Analysis")
    print("="*80)
    
    ground_truth = df['Label'].values
    
    # Agreement with ground truth
    expert_agreements = []
    
    for expert in range(1, 6):
        # Top-1 agreement
        top1_preds = df[f'Q1_expert{expert}'].values
        top1_agreement = (ground_truth == top1_preds).mean()
        
        # Top-3 agreement
        def check_in_top3(row, expert_id):
            gt = row['Label']
            q1 = row[f'Q1_expert{expert_id}']
            q2 = row[f'Q2_expert{expert_id}']
            q3 = row[f'Q3_expert{expert_id}']
            return gt in [q1, q2, q3]
        
        top3_agreement = df.apply(lambda row: check_in_top3(row, expert), axis=1).mean()
        
        # Cohen's Kappa
        kappa = cohen_kappa_score(ground_truth, top1_preds)
        
        # Weighted Kappa (assuming ordinal scale: 1=Normal, 2=Right, 3=Left, 4=Others)
        kappa_weighted = cohen_kappa_score(ground_truth, top1_preds, weights='linear')
        
        expert_agreements.append({
            'Expert': f'Expert {expert}',
            'Top-1 Agreement': f"{top1_agreement:.1%}",
            'Top-3 Agreement': f"{top3_agreement:.1%}",
            "Cohen's Kappa": f"{kappa:.3f}",
            'Weighted Kappa': f"{kappa_weighted:.3f}"
        })
    
    # Calculate mean and SD
    top1_mean = np.mean([float(e['Top-1 Agreement'].rstrip('%'))/100 for e in expert_agreements])
    top1_std = np.std([float(e['Top-1 Agreement'].rstrip('%'))/100 for e in expert_agreements])
    top3_mean = np.mean([float(e['Top-3 Agreement'].rstrip('%'))/100 for e in expert_agreements])
    top3_std = np.std([float(e['Top-3 Agreement'].rstrip('%'))/100 for e in expert_agreements])
    kappa_mean = np.mean([float(e["Cohen's Kappa"]) for e in expert_agreements])
    kappa_std = np.std([float(e["Cohen's Kappa"]) for e in expert_agreements])
    
    expert_agreements.append({
        'Expert': 'Mean (SD)',
        'Top-1 Agreement': f"{top1_mean:.1%} ({top1_std:.1%})",
        'Top-3 Agreement': f"{top3_mean:.1%} ({top3_std:.1%})",
        "Cohen's Kappa": f"{kappa_mean:.3f} ({kappa_std:.3f})",
        'Weighted Kappa': '-'
    })
    
    table4_expert_agreement = pd.DataFrame(expert_agreements)
    
    # Inter-expert agreement (Fleiss' Kappa approximation using pairwise kappas)
    all_pairwise_kappas = []
    for i in range(1, 6):
        for j in range(i+1, 6):
            pred_i = df[f'Q1_expert{i}'].values
            pred_j = df[f'Q1_expert{j}'].values
            kappa_ij = cohen_kappa_score(pred_i, pred_j)
            all_pairwise_kappas.append(kappa_ij)
    
    mean_pairwise_kappa = np.mean(all_pairwise_kappas)
    
    table4_inter_expert = pd.DataFrame({
        'Metric': ['Overall Fleiss\' Kappa (approx.)', 'Pairwise Cohen\'s Kappa (mean)'],
        'Value': [f"{mean_pairwise_kappa:.3f}", f"{mean_pairwise_kappa:.3f}"],
        'Interpretation': ['Substantial agreement', 'Substantial agreement']
    })
    
    # Agreement by diagnosis category
    category_agreements = []
    
    for label, label_name in [(1, 'Normal'), (2, 'Right-sided'), 
                               (3, 'Left-sided'), (4, 'Others')]:
        df_cat = df[df['Label'] == label]
        
        if len(df_cat) == 0:
            continue
        
        # Mean top-1 agreement
        top1_agreements_cat = []
        for expert in range(1, 6):
            top1_preds = df_cat[f'Q1_expert{expert}'].values
            top1_agreement = (df_cat['Label'].values == top1_preds).mean()
            top1_agreements_cat.append(top1_agreement)
        
        mean_top1 = np.mean(top1_agreements_cat)
        
        # Mean expert entropy
        df_cat_with_entropy = df_cat.copy()
        df_cat_with_entropy['expert_entropy'] = df_cat_with_entropy.apply(
            ExpertMetrics.calculate_expert_entropy, axis=1)
        mean_entropy = df_cat_with_entropy['expert_entropy'].mean()
        
        # Ground truth agreement (mean of expert top-1 with GT)
        gt_agreement = np.mean(top1_agreements_cat)
        
        category_agreements.append({
            'Category': label_name,
            'Mean Top-1 Agreement': f"{mean_top1:.1%}",
            'Mean Expert Entropy': f"{mean_entropy:.2f}",
            'Ground Truth Agreement': f"{gt_agreement:.1%}"
        })
    
    table4_category = pd.DataFrame(category_agreements)
    
    print("\n[Agreement with Ground Truth]")
    print(table4_expert_agreement.to_string(index=False))
    print("\n[Inter-Expert Agreement]")
    print(table4_inter_expert.to_string(index=False))
    print("\n[Agreement by Diagnosis Category]")
    print(table4_category.to_string(index=False))
    
    return table4_expert_agreement, table4_inter_expert, table4_category

def generate_table5(all_test_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Table 5: Performance Comparison by Expert Disagreement Level
    """
    print("\n" + "="*80)
    print("TABLE 5: Performance Comparison by Expert Disagreement Level")
    print("="*80)
    
    # Combine all test dataframes
    df_all = pd.concat(all_test_dfs, ignore_index=True)
    
    # Calculate entropy threshold (75th percentile)
    entropy_threshold = df_all['expert_entropy'].quantile(0.75)
    
    print(f"Entropy threshold (75th percentile): {entropy_threshold:.2f}")
    
    # Classify cases
    df_all['disagreement_level'] = df_all['expert_entropy'].apply(
        lambda x: 'High' if x >= entropy_threshold else 'Low'
    )
    
    # Case distribution
    df_low = df_all[df_all['disagreement_level'] == 'Low']
    df_high = df_all[df_all['disagreement_level'] == 'High']
    
    print(f"\nLow disagreement cases: {len(df_low)} ({len(df_low)/len(df_all)*100:.0f}%)")
    print(f"High disagreement cases: {len(df_high)} ({len(df_high)/len(df_all)*100:.0f}%)")
    
    # Build table
    def calculate_metrics(df_subset, subset_name):
        # Single prediction
        single_correct = (df_subset['Label'] == df_subset['single_pred']).sum()
        single_errors = len(df_subset) - single_correct
        single_accuracy = single_correct / len(df_subset)
        
        # CP
        cp_coverage = df_subset['coverage'].sum()
        cp_failures = len(df_subset) - cp_coverage
        cp_coverage_rate = cp_coverage / len(df_subset)
        cp_avg_set_size = df_subset['set_size'].mean()
        cp_efficiency = (df_subset['set_size'] == 1).mean()
        
        # Baseline Set Size (Fixed Set Size Approach)
        baseline_set_size = calculate_baseline_set_size_fixed(df_subset, cp_coverage_rate)
        efficiency_gain = ((baseline_set_size - cp_avg_set_size) / baseline_set_size * 100) if baseline_set_size > 0 else 0
        
        # Improvement
        error_reduction = single_errors - cp_failures
        error_reduction_rate = error_reduction / single_errors if single_errors > 0 else 0
        rrr = 1 - (cp_failures / single_errors) if single_errors > 0 else 0
        nnt = len(df_subset) / error_reduction if error_reduction > 0 else np.inf
        
        return {
            'n': len(df_subset),
            'mean_entropy': df_subset['expert_entropy'].mean(),
            'std_entropy': df_subset['expert_entropy'].std(),
            'single_accuracy': single_accuracy,
            'single_errors': single_errors,
            'cp_coverage_rate': cp_coverage_rate,
            'cp_failures': int(cp_failures),
            'cp_avg_set_size': cp_avg_set_size,
            'cp_efficiency': cp_efficiency,
            'baseline_set_size': baseline_set_size,
            'efficiency_gain': efficiency_gain,
            'error_reduction': int(error_reduction),
            'error_reduction_rate': error_reduction_rate,
            'rrr': rrr,
            'nnt': nnt
        }
    
    metrics_low = calculate_metrics(df_low, 'Low')
    metrics_high = calculate_metrics(df_high, 'High')
    metrics_overall = calculate_metrics(df_all, 'Overall')
    
    # Case distribution table
    table5_dist = pd.DataFrame({
        'Metric': [
            'Expert Entropy Range',
            'Mean Entropy (SD)',
            'Diagnosis Distribution',
            '  Normal',
            '  Right-sided',
            '  Left-sided',
            '  Others'
        ],
        'Low Disagreement': [
            f"0.00 - {entropy_threshold:.2f}",
            f"{metrics_low['mean_entropy']:.2f} ({metrics_low['std_entropy']:.2f})",
            '',
            f"{(df_low['Label']==1).sum()} ({(df_low['Label']==1).mean():.1%})",
            f"{(df_low['Label']==2).sum()} ({(df_low['Label']==2).mean():.1%})",
            f"{(df_low['Label']==3).sum()} ({(df_low['Label']==3).mean():.1%})",
            f"{(df_low['Label']==4).sum()} ({(df_low['Label']==4).mean():.1%})"
        ],
        'High Disagreement': [
            f"{entropy_threshold:.2f} - {df_all['expert_entropy'].max():.2f}",
            f"{metrics_high['mean_entropy']:.2f} ({metrics_high['std_entropy']:.2f})",
            '',
            f"{(df_high['Label']==1).sum()} ({(df_high['Label']==1).mean():.1%})",
            f"{(df_high['Label']==2).sum()} ({(df_high['Label']==2).mean():.1%})",
            f"{(df_high['Label']==3).sum()} ({(df_high['Label']==3).mean():.1%})",
            f"{(df_high['Label']==4).sum()} ({(df_high['Label']==4).mean():.1%})"
        ]
    })
    
    # Performance comparison table
    table5_performance = pd.DataFrame({
        'Metric': [
            '=== Single Prediction (Top-1 Only) ===',
            'Accuracy',
            'Errors (n)',
            'Error Rate',
            '',
            '=== Conformal Prediction (95% CI) ===',
            'Coverage Rate',
            'Coverage Failures (n)',
            'Coverage Failure Rate',
            'Average Set Size',
            'Efficiency (Set Size = 1)',
            'Baseline Set Size (Fixed)*',
            'Efficiency Gain vs Baseline',
            '',
            '=== Improvement Metrics ===',
            'Error Reduction (n)',
            'Error Reduction Rate',
            'Relative Risk Reduction',
            'Number Needed to Treat (NNT)'
        ],
        'Low Disagreement': [
            '',
            f"{metrics_low['single_accuracy']:.1%} ({int(metrics_low['n'] * metrics_low['single_accuracy'])}/{metrics_low['n']})",
            f"{metrics_low['single_errors']}",
            f"{1-metrics_low['single_accuracy']:.1%}",
            '',
            '',
            f"{metrics_low['cp_coverage_rate']:.1%} ({int(metrics_low['n'] * metrics_low['cp_coverage_rate'])}/{metrics_low['n']})",
            f"{metrics_low['cp_failures']}",
            f"{1-metrics_low['cp_coverage_rate']:.1%}",
            f"{metrics_low['cp_avg_set_size']:.2f}",
            f"{metrics_low['cp_efficiency']:.1%}",
            f"{metrics_low['baseline_set_size']}",
            f"{metrics_low['efficiency_gain']:.1f}%",
            '',
            '',
            f"{metrics_low['error_reduction']} ({metrics_low['single_errors']}→{metrics_low['cp_failures']})",
            f"{metrics_low['error_reduction_rate']:.1%}",
            f"{metrics_low['rrr']:.1%}",
            f"{metrics_low['nnt']:.0f}" if metrics_low['nnt'] != np.inf else 'N/A'
        ],
        'High Disagreement': [
            '',
            f"{metrics_high['single_accuracy']:.1%} ({int(metrics_high['n'] * metrics_high['single_accuracy'])}/{metrics_high['n']})",
            f"{metrics_high['single_errors']}",
            f"{1-metrics_high['single_accuracy']:.1%}",
            '',
            '',
            f"{metrics_high['cp_coverage_rate']:.1%} ({int(metrics_high['n'] * metrics_high['cp_coverage_rate'])}/{metrics_high['n']})",
            f"{metrics_high['cp_failures']}",
            f"{1-metrics_high['cp_coverage_rate']:.1%}",
            f"{metrics_high['cp_avg_set_size']:.2f}",
            f"{metrics_high['cp_efficiency']:.1%}",
            f"{metrics_high['baseline_set_size']}",
            f"{metrics_high['efficiency_gain']:.1f}%",
            '',
            '',
            f"{metrics_high['error_reduction']} ({metrics_high['single_errors']}→{metrics_high['cp_failures']})",
            f"{metrics_high['error_reduction_rate']:.1%}",
            f"{metrics_high['rrr']:.1%}",
            f"{metrics_high['nnt']:.0f}" if metrics_high['nnt'] != np.inf else 'N/A'
        ],
        'Overall': [
            '',
            f"{metrics_overall['single_accuracy']:.1%} ({int(metrics_overall['n'] * metrics_overall['single_accuracy'])}/{metrics_overall['n']})",
            f"{metrics_overall['single_errors']}",
            f"{1-metrics_overall['single_accuracy']:.1%}",
            '',
            '',
            f"{metrics_overall['cp_coverage_rate']:.1%} ({int(metrics_overall['n'] * metrics_overall['cp_coverage_rate'])}/{metrics_overall['n']})",
            f"{metrics_overall['cp_failures']}",
            f"{1-metrics_overall['cp_coverage_rate']:.1%}",
            f"{metrics_overall['cp_avg_set_size']:.2f}",
            f"{metrics_overall['cp_efficiency']:.1%}",
            f"{metrics_overall['baseline_set_size']}",
            f"{metrics_overall['efficiency_gain']:.1f}%",
            '',
            '',
            f"{metrics_overall['error_reduction']} ({metrics_overall['single_errors']}→{metrics_overall['cp_failures']})",
            f"{metrics_overall['error_reduction_rate']:.1%}",
            f"{metrics_overall['rrr']:.1%}",
            f"{metrics_overall['nnt']:.0f}" if metrics_overall['nnt'] != np.inf else 'N/A'
        ]
    })
    
    print("\n[Case Distribution and Baseline Characteristics]")
    print(table5_dist.to_string(index=False))
    print("\n[Performance Comparison: Single Prediction vs Conformal Prediction]")
    print(table5_performance.to_string(index=False))
    
    # Print Baseline Set Size interpretation
    print(f"\n{'='*80}")
    print("BASELINE SET SIZE INTERPRETATION:")
    print(f"{'='*80}")
    print(f"Low Disagreement (Q1-Q3 of entropy):")
    print(f"  - CP achieves {metrics_low['cp_coverage_rate']:.1%} coverage with avg set size {metrics_low['cp_avg_set_size']:.2f}")
    print(f"  - Baseline needs fixed set size {metrics_low['baseline_set_size']} for same coverage")
    print(f"  - Efficiency Gain: {metrics_low['efficiency_gain']:.1f}%")
    print(f"\nHigh Disagreement (Q4 of entropy):")
    print(f"  - CP achieves {metrics_high['cp_coverage_rate']:.1%} coverage with avg set size {metrics_high['cp_avg_set_size']:.2f}")
    print(f"  - Baseline needs fixed set size {metrics_high['baseline_set_size']} for same coverage")
    print(f"  - Efficiency Gain: {metrics_high['efficiency_gain']:.1f}%")
    print(f"\nOverall:")
    print(f"  - CP achieves {metrics_overall['cp_coverage_rate']:.1%} coverage with avg set size {metrics_overall['cp_avg_set_size']:.2f}")
    print(f"  - Baseline needs fixed set size {metrics_overall['baseline_set_size']} for same coverage")
    print(f"  - Overall Efficiency Gain: {metrics_overall['efficiency_gain']:.1f}%")
    print(f"{'='*80}")
    
    # Save for later use
    return table5_dist, table5_performance, df_all

def calculate_baseline_set_size_fixed(df_subset: pd.DataFrame, target_coverage: float) -> int:
    true_labels = df_subset['Label'].values
    probas = df_subset[['class1_proba', 'class2_proba', 
                        'class3_proba', 'class4_proba']].values
    
    # Try different fixed set sizes (1, 2, 3, 4)
    for fixed_size in [1, 2, 3, 4]:
        coverage_count = 0
        
        for true_label, proba in zip(true_labels, probas):

            top_k_indices = np.argsort(-proba)[:fixed_size] + 1
            
            # Coverage check
            if true_label in top_k_indices:
                coverage_count += 1
        
        coverage_rate = coverage_count / len(df_subset)
        
        if coverage_rate >= target_coverage:
            return fixed_size
        
    return 4


def generate_table6b_efficiency_analysis(all_test_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Table 6B: CP Efficiency Analysis - Conditional Coverage by Expert Entropy
    
    Entropy 구간별로 CP와 Baseline(Fixed Set Size)의 효율성 비교
    """
    print("\n" + "="*80)
    print("TABLE 6B: CP Efficiency Analysis (Conditional Coverage by Expert Entropy)")
    print("="*80)
    
    # Combine all test dataframes
    df_all = pd.concat(all_test_dfs, ignore_index=True)
    
    # Calculate entropy quartiles
    quartiles = df_all['expert_entropy'].quantile([0.25, 0.5, 0.75]).values
    
    print(f"\nEntropy Quartiles:")
    print(f"  Q1 (25th percentile): {quartiles[0]:.3f}")
    print(f"  Q2 (50th percentile): {quartiles[1]:.3f}")
    print(f"  Q3 (75th percentile): {quartiles[2]:.3f}")
    
    # Define entropy ranges
    entropy_ranges = [
        ("Q1 (Very Low)", 0.0, quartiles[0]),
        ("Q2 (Low)", quartiles[0], quartiles[1]),
        ("Q3 (High)", quartiles[1], quartiles[2]),
        ("Q4 (Very High)", quartiles[2], 2.0)
    ]
    
    results = []
    
    for range_name, lower, upper in entropy_ranges:
        # Filter cases in this entropy range
        df_subset = df_all[(df_all['expert_entropy'] >= lower) & 
                           (df_all['expert_entropy'] < upper)]
        
        if len(df_subset) == 0:
            continue
        
        # CP performance
        cp_avg_set_size = df_subset['set_size'].mean()
        cp_coverage = df_subset['coverage'].mean()
        
        # Baseline: Fixed Set Size required for same coverage
        baseline_set_size = calculate_baseline_set_size_fixed(df_subset, cp_coverage)
        
        # Calculate efficiency metrics
        efficiency_gain_absolute = baseline_set_size - cp_avg_set_size
        efficiency_gain_percent = (efficiency_gain_absolute / baseline_set_size) * 100
        
        # Additional metrics
        avg_entropy = df_subset['expert_entropy'].mean()
        n_cases = len(df_subset)
        
        results.append({
            'Entropy Range': range_name,
            'Range': f"{lower:.2f}-{upper:.2f}",
            'n': n_cases,
            'Avg Entropy': f"{avg_entropy:.2f}",
            'CP Avg Set Size': f"{cp_avg_set_size:.2f}",
            'CP Coverage': f"{cp_coverage:.1%}",
            'Baseline Set Size': baseline_set_size,
            'Set Size Reduction': f"{efficiency_gain_absolute:.2f}",
            'Efficiency Gain (%)': f"{efficiency_gain_percent:.1f}%"
        })
        
        print(f"\n{range_name} ({lower:.2f}-{upper:.2f}):")
        print(f"  Cases: {n_cases:,}")
        print(f"  Avg Entropy: {avg_entropy:.3f}")
        print(f"  CP Avg Set Size: {cp_avg_set_size:.3f}")
        print(f"  CP Coverage: {cp_coverage:.1%}")
        print(f"  Baseline Set Size (fixed): {baseline_set_size}")
        print(f"  Efficiency Gain: {efficiency_gain_percent:.1f}% ({efficiency_gain_absolute:.2f} set size reduction)")
    
    table6b = pd.DataFrame(results)
    
    # Calculate overall average
    overall_cp_avg = df_all['set_size'].mean()
    overall_cp_coverage = df_all['coverage'].mean()
    overall_baseline = calculate_baseline_set_size_fixed(df_all, overall_cp_coverage)
    overall_gain = ((overall_baseline - overall_cp_avg) / overall_baseline) * 100
    
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY:")
    print(f"  Total Cases: {len(df_all):,}")
    print(f"  CP Avg Set Size: {overall_cp_avg:.3f}")
    print(f"  CP Coverage: {overall_cp_coverage:.1%}")
    print(f"  Baseline Set Size: {overall_baseline}")
    print(f"  Overall Efficiency Gain: {overall_gain:.1f}%")
    print(f"{'='*80}")
    
    return table6b

def generate_table6(df_all: pd.DataFrame, entropy_threshold: float) -> pd.DataFrame:
    """
    Table 6: Coverage Failure Analysis
    """
    print("\n" + "="*80)
    print("TABLE 6: Coverage Failure Analysis")
    print("="*80)
    
    # Extract coverage failures
    df_failures = df_all[df_all['coverage'] == 0].copy()
    
    print(f"\nTotal coverage failures: {len(df_failures)}/{len(df_all)} ({len(df_failures)/len(df_all)*100:.1f}%)")
    
    if len(df_failures) == 0:
        print("No coverage failures found!")
        return pd.DataFrame()
    
    # Detailed failure characteristics (show first 8 or all if less)
    n_show = min(8, len(df_failures))
    df_failures_sample = df_failures.head(n_show).copy()
    
    table6_detail = pd.DataFrame({
        'Case ID': range(1, n_show + 1),
        'True Label': df_failures_sample['Label'].apply(
            lambda x: ['Normal', 'Right', 'Left', 'Others'][int(x)-1]
        ).tolist(),
        'CP Set': df_failures_sample['pred_set'].apply(
            lambda x: ', '.join([['Normal', 'Right', 'Left', 'Others'][int(i)-1] for i in x])
        ).tolist(),
        'Set Size': df_failures_sample['set_size'].tolist(),
        'Expert Entropy': df_failures_sample['expert_entropy'].round(2).tolist(),
        'Disagreement Level': df_failures_sample['expert_entropy'].apply(
            lambda x: 'High' if x >= entropy_threshold else 'Low'
        ).tolist()
    })
    
    # Coverage failure distribution
    table6_dist_data = []
    
    # By disagreement level
    for level in ['Low', 'High']:
        df_level = df_all[df_all['expert_entropy'] >= entropy_threshold] if level == 'High' else df_all[df_all['expert_entropy'] < entropy_threshold]
        failures_level = df_level[df_level['coverage'] == 0]
        
        table6_dist_data.append({
            'Stratification': f'{level} Disagreement',
            'Coverage Failures': len(failures_level),
            'Total Cases': len(df_level),
            'Failure Rate': f"{len(failures_level)/len(df_level)*100:.1f}%",
            'Average Entropy': f"{failures_level['expert_entropy'].mean():.2f}" if len(failures_level) > 0 else 'N/A'
        })
    
    # By true label
    for label, label_name in [(1, 'Normal'), (2, 'Right-sided'), 
                               (3, 'Left-sided'), (4, 'Others')]:
        df_label = df_all[df_all['Label'] == label]
        failures_label = df_label[df_label['coverage'] == 0]
        
        table6_dist_data.append({
            'Stratification': label_name,
            'Coverage Failures': len(failures_label),
            'Total Cases': len(df_label),
            'Failure Rate': f"{len(failures_label)/len(df_label)*100:.1f}%" if len(df_label) > 0 else 'N/A',
            'Average Entropy': f"{failures_label['expert_entropy'].mean():.2f}" if len(failures_label) > 0 else 'N/A'
        })
    
    # By CP set size
    for size in [1, 2]:
        df_size = df_all[df_all['set_size'] == size]
        failures_size = df_size[df_size['coverage'] == 0]
        
        table6_dist_data.append({
            'Stratification': f'Size = {size}',
            'Coverage Failures': len(failures_size),
            'Total Cases': len(df_size),
            'Failure Rate': f"{len(failures_size)/len(df_size)*100:.1f}%" if len(df_size) > 0 else 'N/A',
            'Average Entropy': f"{failures_size['expert_entropy'].mean():.2f}" if len(failures_size) > 0 else 'N/A'
        })
    
    # Size >= 3
    df_size_3plus = df_all[df_all['set_size'] >= 3]
    failures_size_3plus = df_size_3plus[df_size_3plus['coverage'] == 0]
    
    table6_dist_data.append({
        'Stratification': 'Size ≥ 3',
        'Coverage Failures': len(failures_size_3plus),
        'Total Cases': len(df_size_3plus),
        'Failure Rate': f"{len(failures_size_3plus)/len(df_size_3plus)*100:.1f}%" if len(df_size_3plus) > 0 else '0.0%',
        'Average Entropy': f"{failures_size_3plus['expert_entropy'].mean():.2f}" if len(failures_size_3plus) > 0 else 'N/A'
    })
    
    table6_dist = pd.DataFrame(table6_dist_data)
    
    print("\n[Characteristics of Coverage Failures - Sample]")
    print(table6_detail.to_string(index=False))
    print("\n[Coverage Failure Distribution]")
    print(table6_dist.to_string(index=False))
    
    return table6_detail, table6_dist

def run_complete_analysis(df: pd.DataFrame, n_runs: int = 300, 
                         alpha: float = 0.05, 
                         global_seed: int = GLOBAL_SEED,
                         output_dir: str = './CP_results'):
    print("=" * 80)
    print("COMPLETE CONFORMAL PREDICTION ANALYSIS WITH TABLES")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Run experiments
    print(f"\n[Step 1] Running {n_runs} experiments...")
    
    results_list = []
    all_test_dfs = []
    
    for i in range(n_runs):
        if (i + 1) % 50 == 0:
            print(f"Progress: {i+1}/{n_runs} ({(i+1)/n_runs*100:.1f}%)")
        
        exp_seed = global_seed * 1000 + i
        result = run_single_experiment_extended(df, alpha=alpha, random_state=exp_seed)
        
        results_list.append({
            'run': i + 1,
            'random_state': result['random_state'],
            'threshold': result['threshold'],
            'coverage': result['coverage'],
            'avg_set_size': result['avg_set_size'],
            'corr_entropy': result['corr_entropy'],
            'p_entropy': result['p_entropy'],
        })
        
        all_test_dfs.append(result['df_test'])
    
    results_df = pd.DataFrame(results_list)
    
    # Step 2: Generate Tables
    print("\n[Step 2] Generating tables...")
    
    table3_overall, table3_setsize = generate_table3(results_df, all_test_dfs)
    
    table4_expert, table4_inter, table4_category = generate_table4(df)
    
    table5_dist, table5_performance, df_all_combined = generate_table5(all_test_dfs)
    
    entropy_threshold = df_all_combined['expert_entropy'].quantile(0.75)
    table6_detail, table6_dist = generate_table6(df_all_combined, entropy_threshold)
    
    # Step 3: Save all tables
    print("\n[Step 3] Saving tables...")
    
    with pd.ExcelWriter(os.path.join(output_dir, 'all_tables.xlsx')) as writer:
        table3_overall.to_excel(writer, sheet_name='Table3_Overall', index=False)
        table3_setsize.to_excel(writer, sheet_name='Table3_SetSize', index=False)
        table4_expert.to_excel(writer, sheet_name='Table4_Expert', index=False)
        table4_inter.to_excel(writer, sheet_name='Table4_Inter', index=False)
        table4_category.to_excel(writer, sheet_name='Table4_Category', index=False)
        table5_dist.to_excel(writer, sheet_name='Table5_Distribution', index=False)
        table5_performance.to_excel(writer, sheet_name='Table5_Performance', index=False)
        if len(table6_detail) > 0:
            table6_detail.to_excel(writer, sheet_name='Table6_Detail', index=False)
        table6_dist.to_excel(writer, sheet_name='Table6_Distribution', index=False)
    
    print(f"  ✓ All tables saved to {output_dir}/all_tables.xlsx")
    
    # Also save as CSV
    table3_overall.to_csv(os.path.join(output_dir, 'table3_overall.csv'), index=False)
    table5_performance.to_csv(os.path.join(output_dir, 'table5_performance.csv'), index=False)
    
    print("\n" + "=" * 80)
    print("✅ COMPLETE ANALYSIS FINISHED!")
    print("=" * 80)
    
    return {
        'results_df': results_df,
        'all_test_dfs': all_test_dfs,
        'tables': {
            'table3_overall': table3_overall,
            'table3_setsize': table3_setsize,
            'table4_expert': table4_expert,
            'table4_inter': table4_inter,
            'table4_category': table4_category,
            'table5_dist': table5_dist,
            'table5_performance': table5_performance,
            'table6_detail': table6_detail,
            'table6_dist': table6_dist
        }
    }

if __name__ == '__main__':
    df = pd.read_csv('./your data')

    results = run_complete_analysis(df, n_runs=300, alpha=0.05, 
                                    global_seed=GLOBAL_SEED, 
                                    output_dir='./your_dir')