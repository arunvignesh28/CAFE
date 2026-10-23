import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression

# Add project path to import local modules
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from cafe_main.logger import *
from cafe_main.task_mapping import base_path, task_measure, task_dict, task_type
from cafe_main.tools import downstream_task_new

# Import your causal map creation function
try:
    from cafe_causal_ablation import create_statistical_causal_map
except ImportError:
    info("Cannot import create_statistical_causal_map. Will create a local implementation.")
    def create_statistical_causal_map(df):
        feature_names = list(df.columns[:-1])
        target_name = df.columns[-1]
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        causal_map = {}
        
        # Method 1: Use mutual information to determine direct relationships
        try:
            # Calculate mutual information between each feature and target
            mi_scores = mutual_info_regression(X, y)
            
            # Normalize scores to [0,1]
            if len(mi_scores) > 0:
                mi_scores = mi_scores / max(mi_scores) if max(mi_scores) > 0 else mi_scores
                
            # Sort features by MI score
            feature_mi_pairs = list(zip(feature_names, mi_scores))
            feature_mi_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Determine thresholds - top 20% are direct, next 30% are indirect
            n_features = len(feature_names)
            direct_threshold = int(0.2 * n_features)
            indirect_threshold = int(0.5 * n_features)
            
            # Assign groups and scores
            for i, (feature, score) in enumerate(feature_mi_pairs):
                if i < direct_threshold:
                    group = 'direct'
                elif i < indirect_threshold:
                    group = 'indirect'
                else:
                    group = 'other'
                
                causal_map[feature] = {
                    'group': group,
                    'score': float(score)
                }
        except Exception as e:
            print(f"Error using mutual information: {e}. Falling back to correlation.")
            
            # Method 2: Fallback to correlation
            corr_scores = []
            for i, feature in enumerate(feature_names):
                corr = np.corrcoef(X[:, i], y)[0, 1]
                corr_scores.append(abs(corr))
            
            # Normalize scores
            max_corr = max(corr_scores) if corr_scores else 1
            corr_scores = [c/max_corr for c in corr_scores]
            
            # Sort features by correlation
            feature_corr_pairs = list(zip(feature_names, corr_scores))
            feature_corr_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Same thresholds as before
            n_features = len(feature_names)
            direct_threshold = int(0.2 * n_features)
            indirect_threshold = int(0.5 * n_features)
            
            # Assign groups and scores
            for i, (feature, score) in enumerate(feature_corr_pairs):
                if i < direct_threshold:
                    group = 'direct'
                elif i < indirect_threshold:
                    group = 'indirect'
                else:
                    group = 'other'
                
                causal_map[feature] = {
                    'group': group,
                    'score': float(score)
                }
        
        return causal_map
    
def load_causal_map(file_name, data_dir):
    """Load causal map from JSON file"""
    # Strip .hdf or .csv extension if present
    if file_name.endswith('.hdf'):
        file_name = file_name[:-4]
    elif file_name.endswith('.csv'):
        file_name = file_name[:-4]
    
    map_dir = os.path.join(data_dir, 'causal_maps')
    map_path = os.path.join(map_dir, f'causal_map_{file_name}.json')
    
    if os.path.exists(map_path):
        with open(map_path, 'r') as f:
            return json.load(f)
    
    # Try with dash instead of underscore
    map_path_dash = os.path.join(map_dir, f'causal_map_{file_name.replace("_", "-")}.json')
    if os.path.exists(map_path_dash):
        with open(map_path_dash, 'r') as f:
            return json.load(f)
    
    raise FileNotFoundError(f"Causal map not found: {map_path} or {map_path_dash}")

def load_model_results(dataset, model_variant):
    """
    Load best features from model results with more flexible file pattern matching
    """
    base_dir = os.path.join(BASE_DIR, 'tmp')
    
    # Create multiple search patterns based on model variant
    if model_variant == 'causal':
        # Look for directories containing the dataset name and 'enhanced' or 'causal'
        dir_patterns = [
            f'{dataset}*enhanced*',
            f'{dataset}*causal*',
            f'{dataset}*'  # Fallback
        ]
        # Look for files that indicate they're enhanced/causal best results
        file_patterns = [
            '*enhanced_best.csv',
            '*causal*best.csv',
            '*best.csv',  # Fallback
            '*.csv'       # Last resort
        ]
    else:  # statistical
        # Look for directories containing the dataset name and 'statistical' or 'ablation'
        dir_patterns = [
            f'{dataset}*statistical*',
            f'{dataset}*ablation*',
            f'{dataset}*'  # Fallback to any directory with dataset name
        ]
        # Look for files that indicate they're statistical/ablation best results
        file_patterns = [
            '*statistical_ablation_best.csv',
            '*statistical*best.csv',
            '*ablation*best.csv',
            '*best.csv',  # Fallback
            '*.csv'       # Last resort
        ]
    
    # Find matching directory
    result_dir = None
    for pattern in dir_patterns:
        matching_dirs = glob.glob(os.path.join(base_dir, pattern))
        if matching_dirs:
            result_dir = matching_dirs[0]
            info(f"Found directory matching '{pattern}': {result_dir}")
            break
    
    if not result_dir:
        raise FileNotFoundError(f"No result directory found for dataset: {dataset}")
    
    # Find result files
    result_files = []
    for pattern in file_patterns:
        matches = glob.glob(os.path.join(result_dir, pattern))
        if matches:
            result_files = matches
            info(f"Found {len(result_files)} files matching '{pattern}'")
            break
    
    if not result_files:
        raise FileNotFoundError(f"No result files found in {result_dir}")
    
    # Find best file by score in filename
    best_score = -float('inf')
    best_file = None
    
    for file in result_files:
        # Extract score from filename
        filename = os.path.basename(file)
        try:
            # Look for numeric values in filename parts
            parts = filename.replace('-', '_').split('_')
            for part in parts:
                if part.replace('.', '', 1).isdigit():
                    try:
                        score = float(part)
                        if score > best_score:
                            best_score = score
                            best_file = file
                            break
                    except ValueError:
                        pass
        except:
            pass
    
    if not best_file:
        # If score extraction failed, use first file
        best_file = result_files[0]
        info(f"Could not extract score from filenames. Using first file: {best_file}")
        # Try to get a reasonable score value
        best_score = 0.0
    else:
        info(f"Selected best file: {best_file} with score: {best_score}")
    
    try:
        # Read the CSV file with engineered features
        df = pd.read_csv(best_file)
        return df, best_score
    except Exception as e:
        error(f"Error reading {best_file}: {e}")
        # Try with different encoding or error handling
        try:
            df = pd.read_csv(best_file, encoding='latin1')
            return df, best_score
        except:
            # Last resort: try with error_bad_lines=False (skipinitialspace for pandas >= 1.3)
            try:
                df = pd.read_csv(best_file, encoding='latin1', on_bad_lines='skip')
                return df, best_score
            except:
                # Very old pandas versions
                df = pd.read_csv(best_file, encoding='latin1', error_bad_lines=False)
                return df, best_score

def add_gaussian_noise(X, intensity):
    """Add Gaussian noise to features based on intensity level"""
    noise_levels = {
        'low': 0.1,
        'medium': 0.5,
        'high': 1.0
    }
    noise_level = noise_levels.get(intensity, 0.1)
    
    # Get feature standard deviations to scale noise properly
    feature_stds = X.std(axis=0)
    
    # Generate noise scaled by each feature's standard deviation
    if isinstance(X, pd.DataFrame):
        noise = np.random.normal(0, noise_level, X.shape) * feature_stds.values.reshape(1, -1)
        X_noisy = X.copy()
        for i, col in enumerate(X.columns):
            X_noisy[col] = X[col].values + noise[:, i]
        return X_noisy
    else:
        noise = np.random.normal(0, noise_level, X.shape) * np.std(X, axis=0).reshape(1, -1)
        return X + noise

def mask_features(X, intensity, causal_map=None, feature_group_cache=None):
    """Mask (zero out) specific features based on intensity"""
    X_masked = X.copy()
    
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns
    else:
        feature_names = [f"X{i}" for i in range(X.shape[1])]
    
    # Get feature types if causal map is provided
    if causal_map and intensity in ['direct_causes', 'indirect_causes', 'other']:
        # Use feature_group_cache if available
        if feature_group_cache:
            mask_features = []
            for f in feature_names:
                if f not in feature_group_cache:
                    group = causal_map.get(f, {}).get('group', 'other')
                    feature_group_cache[f] = group
                
                if feature_group_cache[f] == intensity.replace('_causes', ''):
                    mask_features.append(f)
        else:
            # Direct lookup in causal map
            if intensity == 'direct_causes':
                mask_features = [f for f in feature_names if causal_map.get(f, {}).get('group') == 'direct']
            elif intensity == 'indirect_causes':
                mask_features = [f for f in feature_names if causal_map.get(f, {}).get('group') == 'indirect']
            else:  # other
                mask_features = [f for f in feature_names if causal_map.get(f, {}).get('group') == 'other']
    else:
        # Random masking based on percentage
        mask_ratios = {
            'low': 0.1,
            'medium': 0.3,
            'high': 0.5
        }
        mask_ratio = mask_ratios.get(intensity, 0.1)
        n_features = len(feature_names)
        n_mask = max(1, int(n_features * mask_ratio))  # At least 1 feature
        mask_features = np.random.choice(feature_names, n_mask, replace=False)
    
    # Apply mask
    if isinstance(X_masked, pd.DataFrame):
        for col in mask_features:
            if col in X_masked.columns:
                X_masked[col] = 0
    else:
        # For numpy array, need indices not names
        for i, name in enumerate(feature_names):
            if name in mask_features:
                X_masked[:, i] = 0
    
    return X_masked

def create_distribution_shift(X, intensity):
    """Create systematic distribution shifts"""
    X_shifted = X.copy()
    
    shift_factors = {
        'low': 0.5,
        'medium': 1.0,
        'high': 2.0
    }
    shift_factor = shift_factors.get(intensity, 0.5)
    
    # Apply different types of shifts
    if isinstance(X_shifted, pd.DataFrame):
        for col in X_shifted.columns:
            # Each feature gets a different transformation
            shift_type = np.random.choice(['multiply', 'add', 'none'])
            
            if shift_type == 'multiply':
                X_shifted[col] *= (1 + shift_factor * (np.random.random() - 0.5))
            elif shift_type == 'add':
                X_shifted[col] += shift_factor * X_shifted[col].std() * (np.random.random() - 0.5)
    else:
        # For numpy arrays
        for i in range(X_shifted.shape[1]):
            shift_type = np.random.choice(['multiply', 'add', 'none'])
            
            if shift_type == 'multiply':
                X_shifted[:, i] *= (1 + shift_factor * (np.random.random() - 0.5))
            elif shift_type == 'add':
                X_shifted[:, i] += shift_factor * np.std(X_shifted[:, i]) * (np.random.random() - 0.5)
    
    return X_shifted

def evaluate_robustness(dataset, model_variant, shift_type, intensity):
    """
    Evaluate model robustness under different distribution shifts
    
    Parameters:
    - dataset: Dataset name (e.g., 'openml_586')
    - model_variant: 'causal' or 'statistical'
    - shift_type: 'gaussian_noise', 'feature_mask', or 'distribution_shift'
    - intensity: Level of perturbation ('low', 'medium', 'high') or feature type
    
    Returns:
    - Dictionary with performance metrics
    """
    info(f"Evaluating {model_variant} model on {dataset} with {shift_type} ({intensity})")
    
    try:
        # 1. Load original dataset
        try:
            data_path = os.path.join(base_path, f"{dataset}.hdf")
            orig_df = pd.read_hdf(data_path)
            info(f"Loaded HDF dataset from {data_path}")
        except:
            try:
                data_path = os.path.join(base_path, f"{dataset}.csv")
                orig_df = pd.read_csv(data_path)
                info(f"Loaded CSV dataset from {data_path}")
            except:
                try:
                    data_path = os.path.join(base_path, 'GRFG_data', f"{dataset}.hdf")
                    orig_df = pd.read_hdf(data_path)
                    info(f"Loaded HDF dataset from GRFG_data folder: {data_path}")
                except:
                    data_path = os.path.join(base_path, 'GRFG_data', f"{dataset}.csv")
                    orig_df = pd.read_csv(data_path)
                    info(f"Loaded CSV dataset from GRFG_data folder: {data_path}")
        
        # Get task type
        task_name = task_dict.get(dataset, 'reg')  # Default to regression
        measure = task_measure[task_name]
        info(f"Task: {task_name}, Measure: {measure}")
        
        # 2. Load model features and split data
        try:
            model_df, best_score = load_model_results(dataset, model_variant)
            info(f"Loaded model results with score: {best_score}")
        except Exception as e:
            error(f"Failed to load model results: {str(e)}")
            raise
        
        # Get target column name
        y_col = orig_df.columns[-1]
        
        # Match columns between original data and model results
        # First check if model_df already has the target column
        if y_col in model_df.columns:
            X = model_df.drop(columns=[y_col])
            y = model_df[y_col]
            info(f"Target column {y_col} found in model results")
        else:
            # If not, use all columns as features and get target from original df
            X = model_df
            y = orig_df[y_col]
            info(f"Using all columns from model results as features, target from original dataset")
        
        # 3. Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        train_indices = np.arange(len(X_train))
        test_indices = np.arange(len(X_train), len(X_train) + len(X_test))
        
        # 4. Create combined DataFrame for evaluation
        if isinstance(X, pd.DataFrame):
            eval_df = pd.concat([X, pd.Series(y, name=y_col)], axis=1)
        else:
            eval_df = pd.concat([pd.DataFrame(X), pd.Series(y, name=y_col)], axis=1)
        
        # 5. Get baseline performance
        baseline_performance = downstream_task_new(
            eval_df, test_indices, train_indices, task_name, measure, state_num=42
        )
        info(f"Baseline performance: {baseline_performance:.4f}")
        
        # 6. Load or create causal map
        feature_group_cache = {}
        try:
            causal_map = load_causal_map(dataset, base_path)
            info(f"Loaded causal map for {dataset}")
        except FileNotFoundError:
            info(f"Causal map not found for {dataset}, creating statistical map")
            causal_map = create_statistical_causal_map(orig_df)
            info(f"Created statistical causal map with {len(causal_map)} features")
        
        # 7. Apply distribution shift to test data
        X_test_shifted = X_test.copy()
        
        if shift_type == 'gaussian_noise':
            X_test_shifted = add_gaussian_noise(X_test, intensity)
            info(f"Applied Gaussian noise ({intensity}) to test data")
        elif shift_type == 'feature_mask':
            X_test_shifted = mask_features(X_test, intensity, causal_map, feature_group_cache)
            info(f"Applied feature masking ({intensity}) to test data")
        elif shift_type == 'distribution_shift':
            X_test_shifted = create_distribution_shift(X_test, intensity)
            info(f"Applied distribution shift ({intensity}) to test data")
        
        # 8. Create shifted evaluation DataFrame
        if isinstance(X_train, pd.DataFrame):
            X_train_df = X_train
            feature_cols = X_train.columns
        else:
            X_train_df = pd.DataFrame(X_train)
            feature_cols = [f"X{i}" for i in range(X_train.shape[1])]
            
        if isinstance(X_test_shifted, pd.DataFrame):
            X_test_shifted_df = X_test_shifted
        else:
            X_test_shifted_df = pd.DataFrame(X_test_shifted, columns=feature_cols)
        
        # Combine into a single DataFrame
        shifted_df = pd.concat([
            X_train_df,
            X_test_shifted_df
        ])
        
        # Add target column
        y_combined = pd.Series(np.concatenate([y_train, y_test]))
        shifted_df[y_col] = y_combined.values
        
        # 9. Evaluate on shifted data
        shifted_performance = downstream_task_new(
            shifted_df, test_indices, train_indices, task_name, measure, state_num=42
        )
        info(f"Shifted performance: {shifted_performance:.4f}")
        
        # 10. Calculate metrics
        absolute_drop = baseline_performance - shifted_performance
        relative_drop = absolute_drop / baseline_performance if baseline_performance != 0 else 0
        
        # 11. Return results
        result = {
            'dataset': dataset,
            'model_variant': model_variant,
            'shift_type': shift_type,
            'intensity': intensity,
            'baseline_performance': baseline_performance,
            'shifted_performance': shifted_performance,
            'absolute_drop': absolute_drop,
            'relative_drop': relative_drop
        }
        
        info(f"Results: Baseline={baseline_performance:.4f}, Shifted={shifted_performance:.4f}, "
             f"Drop={absolute_drop:.4f} ({relative_drop*100:.1f}%)")
        
        return result
        
    except Exception as e:
        error(f"Error evaluating {model_variant} on {dataset} with {shift_type} ({intensity}): {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'dataset': dataset,
            'model_variant': model_variant,
            'shift_type': shift_type,
            'intensity': intensity,
            'baseline_performance': None,
            'shifted_performance': None,
            'absolute_drop': None,
            'relative_drop': None,
            'error': str(e)
        }

def create_robustness_comparison_report(results_df):
    """
    Generate a detailed comparison report of causal vs. statistical models
    across different datasets and shift types
    """
    # Filter out failed experiments
    valid_results = results_df.dropna(subset=['baseline_performance', 'shifted_performance'])
    
    # Create report file
    report_path = os.path.join(BASE_DIR, 'robustness_comparison_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CAUSAL VS. STATISTICAL MODEL ROBUSTNESS COMPARISON\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL SUMMARY:\n")
        f.write("-"*80 + "\n")
        
        # Calculate average performance drop by model type
        avg_drop_by_model = valid_results.groupby('model_variant')['relative_drop'].mean()
        f.write(f"Average relative performance drop:\n")
        f.write(f"  Causal model:      {avg_drop_by_model.get('causal', 0):.4f} ({avg_drop_by_model.get('causal', 0)*100:.1f}%)\n")
        f.write(f"  Statistical model: {avg_drop_by_model.get('statistical', 0):.4f} ({avg_drop_by_model.get('statistical', 0)*100:.1f}%)\n\n")
        
        # Compare by shift type
        f.write("COMPARISON BY SHIFT TYPE:\n")
        f.write("-"*80 + "\n")
        
        for shift_type in ['gaussian_noise', 'feature_mask', 'distribution_shift']:
            shift_data = valid_results[valid_results['shift_type'] == shift_type]
            
            if shift_data.empty:
                continue
                
            avg_by_model = shift_data.groupby('model_variant')['relative_drop'].mean()
            
            f.write(f"\n{shift_type.upper()}:\n")
            f.write(f"  Causal model:      {avg_by_model.get('causal', 0):.4f} ({avg_by_model.get('causal', 0)*100:.1f}%)\n")
            f.write(f"  Statistical model: {avg_by_model.get('statistical', 0):.4f} ({avg_by_model.get('statistical', 0)*100:.1f}%)\n")
            
            # Winner
            if 'causal' in avg_by_model.index and 'statistical' in avg_by_model.index:
                winner = "CAUSAL" if avg_by_model['causal'] < avg_by_model['statistical'] else "STATISTICAL"
                improvement = abs(avg_by_model['causal'] - avg_by_model['statistical']) / max(avg_by_model['statistical'], 0.001)
                f.write(f"  Winner: {winner} model (by {improvement*100:.1f}%)\n")
        
        # DETAILED BREAKDOWN BY DATASET AND SHIFT TYPE
        f.write("\n\n")
        f.write("="*80 + "\n")
        f.write("DETAILED DATASET-BY-DATASET COMPARISON\n")
        f.write("="*80 + "\n")
        
        # Get unique datasets
        datasets = valid_results['dataset'].unique()
        
        for dataset in sorted(datasets):
            dataset_results = valid_results[valid_results['dataset'] == dataset]
            
            f.write(f"\n\nDATASET: {dataset}\n")
            f.write("-"*80 + "\n")
            
            # Get baseline performances
            baseline_causal = dataset_results[dataset_results['model_variant'] == 'causal']['baseline_performance'].mean()
            baseline_statistical = dataset_results[dataset_results['model_variant'] == 'statistical']['baseline_performance'].mean()
            
            f.write(f"Baseline performances:\n")
            f.write(f"  Causal model:      {baseline_causal:.4f}\n")
            f.write(f"  Statistical model: {baseline_statistical:.4f}\n\n")
            
            # Compare by shift type for this dataset
            for shift_type in ['gaussian_noise', 'feature_mask', 'distribution_shift']:
                shift_data = dataset_results[dataset_results['shift_type'] == shift_type]
                
                if shift_data.empty:
                    continue
                    
                f.write(f"\n{shift_type.upper()}:\n")
                
                # Create comparison table
                f.write(f"{'Intensity':<15} {'Causal Drop':<15} {'Statistical Drop':<15} {'Winner':<10}\n")
                f.write("-"*55 + "\n")
                
                # Get unique intensities for this shift type
                intensities = shift_data['intensity'].unique()
                
                for intensity in sorted(intensities):
                    intensity_data = shift_data[shift_data['intensity'] == intensity]
                    
                    causal_row = intensity_data[intensity_data['model_variant'] == 'causal']
                    statistical_row = intensity_data[intensity_data['model_variant'] == 'statistical']
                    
                    if not causal_row.empty and not statistical_row.empty:
                        causal_drop = causal_row['relative_drop'].values[0]
                        statistical_drop = statistical_row['relative_drop'].values[0]
                        
                        winner = "CAUSAL" if causal_drop < statistical_drop else "STATISTICAL"
                        
                        f.write(f"{str(intensity):<15} {causal_drop*100:>5.1f}%{'':<9} {statistical_drop*100:>5.1f}%{'':<9} {winner:<10}\n")
        
        # Final summary
        f.write("\n\n")
        f.write("="*80 + "\n")
        f.write("FINAL VERDICT\n")
        f.write("="*80 + "\n")
        
        # Count wins by model type
        better_model_counts = {}
        for shift_type in ['gaussian_noise', 'feature_mask', 'distribution_shift']:
            for dataset in datasets:
                shift_dataset_data = valid_results[(valid_results['shift_type'] == shift_type) & 
                                                  (valid_results['dataset'] == dataset)]
                
                if len(shift_dataset_data) < 2:
                    continue
                    
                causal_avg = shift_dataset_data[shift_dataset_data['model_variant'] == 'causal']['relative_drop'].mean()
                statistical_avg = shift_dataset_data[shift_dataset_data['model_variant'] == 'statistical']['relative_drop'].mean()
                
                if not np.isnan(causal_avg) and not np.isnan(statistical_avg):
                    better = "causal" if causal_avg < statistical_avg else "statistical"
                    if shift_type not in better_model_counts:
                        better_model_counts[shift_type] = {"causal": 0, "statistical": 0}
                    better_model_counts[shift_type][better] += 1
        
        # Report wins
        f.write("\nDataset wins by model type and shift type:\n\n")
        for shift_type, counts in better_model_counts.items():
            total = counts["causal"] + counts["statistical"]
            if total > 0:
                f.write(f"{shift_type.upper()}:\n")
                f.write(f"  Causal model wins:      {counts['causal']} ({counts['causal']/total*100:.1f}%)\n")
                f.write(f"  Statistical model wins: {counts['statistical']} ({counts['statistical']/total*100:.1f}%)\n\n")
        
        # Overall winner
        total_causal_wins = sum(counts["causal"] for counts in better_model_counts.values())
        total_statistical_wins = sum(counts["statistical"] for counts in better_model_counts.values())
        total_comparisons = total_causal_wins + total_statistical_wins
        
        if total_comparisons > 0:
            f.write("\nOVERALL WINNER:\n")
            if total_causal_wins > total_statistical_wins:
                f.write(f"CAUSAL model wins overall, with {total_causal_wins}/{total_comparisons} wins ({total_causal_wins/total_comparisons*100:.1f}%)\n")
            elif total_statistical_wins > total_causal_wins:
                f.write(f"STATISTICAL model wins overall, with {total_statistical_wins}/{total_comparisons} wins ({total_statistical_wins/total_comparisons*100:.1f}%)\n")
            else:
                f.write(f"TIE between models, with each winning {total_causal_wins}/{total_comparisons} comparisons\n")
    
    info(f"Robustness comparison report saved to {report_path}")
    return report_path

def create_robustness_visualizations(results_df):
    """Create visualizations for robustness results"""
    # Filter out failed experiments
    valid_results = results_df.dropna(subset=['baseline_performance', 'shifted_performance'])
    
    # 1. Performance Drop Heatmap by Intensity
    plt.figure(figsize=(14, 8))
    for i, model in enumerate(['causal', 'statistical']):
        plt.subplot(1, 2, i+1)
        model_data = valid_results[valid_results['model_variant'] == model]
        
        if model_data.empty:
            continue
            
        # Pivot data for heatmap
        try:
            heatmap_data = model_data.pivot_table(
                index='dataset', 
                columns=['shift_type', 'intensity'], 
                values='relative_drop'
            )
            
            # Sort datasets by average robustness
            dataset_order = model_data.groupby('dataset')['relative_drop'].mean().sort_values().index
            heatmap_data = heatmap_data.reindex(dataset_order)
            
            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlOrRd", 
                       cbar_kws={'label': 'Relative Performance Drop'})
            plt.title(f"{model.capitalize()} Model: Performance Drop by Dataset and Shift Type")
            plt.tight_layout()
        except Exception as e:
            info(f"Error creating heatmap for {model}: {e}")
    
    plt.savefig(os.path.join(BASE_DIR, 'robustness_heatmap.png'), dpi=300, bbox_inches='tight')
    
    # 2. Bar chart comparing models across shift types
    try:
        plt.figure(figsize=(14, 6))
        
        # Calculate average drop by shift type and model
        avg_drop = valid_results.groupby(['shift_type', 'model_variant'])['relative_drop'].mean().reset_index()
        avg_drop_pivot = avg_drop.pivot(index='shift_type', columns='model_variant', values='relative_drop')
        
        if not avg_drop_pivot.empty:
            ax = avg_drop_pivot.plot(kind='bar', color=['royalblue', 'tomato'], figsize=(10, 6))
            plt.title('Average Performance Drop by Shift Type', fontsize=14)
            plt.ylabel('Relative Performance Drop', fontsize=12)
            plt.xlabel('Shift Type', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.legend(['Causal', 'Statistical'], title='Model Type')
            
            # Add data labels
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(os.path.join(BASE_DIR, 'robustness_by_shift_type.png'), dpi=300, bbox_inches='tight')
    except Exception as e:
        info(f"Error creating bar chart: {e}")
    
    # 3. Feature Type Impact Analysis (for feature_mask)
    feature_mask_data = valid_results[valid_results['shift_type'] == 'feature_mask']
    feature_mask_data = feature_mask_data[feature_mask_data['intensity'].isin(['direct_causes', 'indirect_causes', 'other'])]
    
    if not feature_mask_data.empty:
        try:
            plt.figure(figsize=(12, 6))
            
            # Plot drop by feature type and model variant
            feature_impact = feature_mask_data.pivot_table(
                index='intensity', 
                columns='model_variant', 
                values='relative_drop',
                aggfunc='mean'
            )
            
            ax = feature_impact.plot(kind='bar', color=['royalblue', 'tomato'], figsize=(10, 6))
            plt.title('Impact of Masking Different Feature Types', fontsize=14)
            plt.ylabel('Relative Performance Drop', fontsize=12)
            plt.xlabel('Feature Type', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.legend(['Causal', 'Statistical'], title='Model Type')
            
            # Add data labels
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(os.path.join(BASE_DIR, 'feature_type_impact.png'), dpi=300, bbox_inches='tight')
        except Exception as e:
            info(f"Error creating feature impact chart: {e}")
    
    # 4. Robustness Curves for Gaussian Noise
    gaussian_data = valid_results[valid_results['shift_type'] == 'gaussian_noise']
    
    if not gaussian_data.empty:
        try:
            plt.figure(figsize=(10, 6))
            for model in ['causal', 'statistical']:
                model_data = gaussian_data[gaussian_data['model_variant'] == model]
                
                if model_data.empty:
                    continue
                
                # Get average performance by intensity
                intensities = ['low', 'medium', 'high']
                perf_by_intensity = []
                
                for intensity in intensities:
                    intensity_data = model_data[model_data['intensity'] == intensity]
                    if not intensity_data.empty:
                        avg_perf = intensity_data['shifted_performance'].mean() / intensity_data['baseline_performance'].mean()
                        perf_by_intensity.append(avg_perf)
                
                if perf_by_intensity:
                    plt.plot(intensities[:len(perf_by_intensity)], perf_by_intensity, 
                            'o-', linewidth=2, markersize=8, 
                            label=f"{model.capitalize()} Model")
            
            plt.title('Robustness to Gaussian Noise', fontsize=14)
            plt.ylabel('Average Performance Ratio (Shifted/Baseline)', fontsize=12)
            plt.xlabel('Noise Intensity', fontsize=12)
            plt.grid(linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(BASE_DIR, 'noise_robustness_curve.png'), dpi=300, bbox_inches='tight')
        except Exception as e:
            info(f"Error creating robustness curve: {e}")

def run_robustness_experiment(datasets=None, model_variants=None, shift_types=None):
    """Run comprehensive robustness analysis"""
    results = []
    
    # Define experiment parameters
    if datasets is None:
        datasets = [
            "openml_586", "openml_589", "openml_607", 
            "openml_616", "openml_618", "openml_620",
            "german_credit", "wine_red", "ionosphere"
        ]
    
    if model_variants is None:
        model_variants = ['causal', 'statistical']
    
    if shift_types is None:
        shift_types = ['gaussian_noise', 'feature_mask', 'distribution_shift']
    
    intensities = {
        'gaussian_noise': ['low', 'medium', 'high'],
        'feature_mask': ['low', 'medium', 'high', 'direct_causes', 'indirect_causes', 'other'],
        'distribution_shift': ['low', 'medium', 'high']
    }
    
    info(f"Starting robustness analysis with: {len(datasets)} datasets, {len(model_variants)} model variants, {len(shift_types)} shift types")
    
    # Run experiments
    for dataset in datasets:
        info(f"Starting robustness analysis for dataset: {dataset}")
        
        for model_variant in model_variants:
            for shift_type in shift_types:
                for intensity in intensities[shift_type]:
                    # Skip feature-specific tests if not applicable
                    if shift_type == 'feature_mask' and intensity in ['direct_causes', 'indirect_causes', 'other']:
                        # For feature-type specific masking, first check if causal map exists
                        try:
                            # Try to load the causal map to confirm it exists
                            load_causal_map(dataset, base_path)
                        except FileNotFoundError:
                            info(f"Skipping {intensity} test for {dataset} - no causal map available")
                            continue
                    
                    # Run evaluation
                    result = evaluate_robustness(dataset, model_variant, shift_type, intensity)
                    results.append(result)
                    
                    # Save intermediate results
                    pd.DataFrame(results).to_csv(
                        os.path.join(BASE_DIR, 'robustness_results_partial.csv'), 
                        index=False
                    )
    
    # Save final results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(BASE_DIR, 'robustness_results_final.csv'), index=False)
    
    # Create visualizations
    # create_robustness_visualizations(results_df)
    # Create comparison report
    create_robustness_comparison_report(results_df)
    
    return results_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Robustness Analysis for Causal Feature Engineering')
    parser.add_argument('--datasets', nargs='+', default=['openml_586', 'openml_589','openml_607', 'openml_616', 'openml_618', 'openml_620',
                                                            'wine_red', 'wine_white', 'svmguide3', 'spam_base', 'housing_boston'], help='List of datasets to analyze')
    parser.add_argument('--models', nargs='+', default=['causal', 'statistical'], 
                      choices=['causal', 'statistical'], help='Model variants to test')
    parser.add_argument('--shift-types', nargs='+', default=['distribution_shift'], 
                      choices=['gaussian_noise', 'feature_mask', 'distribution_shift'],
                      help='Types of distribution shifts to test')
    args = parser.parse_args()
    
    info("Starting robustness analysis experiment")
    results = run_robustness_experiment(
        datasets=args.datasets,
        model_variants=args.models,
        shift_types=args.shift_types
    )
    info("Robustness analysis complete")