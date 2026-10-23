import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
from cafe_main.cafe import BASE_DIR, load_causal_map
from matplotlib.gridspec import GridSpec
from cafe_main.task_mapping import task_dict
import matplotlib as mpl
from datetime import datetime

# Set AAAI publication standards
mpl.rcParams.update({
    "font.family": "Times New Roman", 
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
})

def load_model_results(dataset, model_variant):
    """Load engineered features from previous experiments"""
    
    # Try multiple possible directory paths
    potential_dirs = []
    if model_variant == 'causal':
        potential_dirs = [
            os.path.join(BASE_DIR, 'tmp', f'{dataset}_enhanced_causal'),
            os.path.join(BASE_DIR, 'tmp', f'{dataset}_causal'),
            os.path.join(os.path.dirname(BASE_DIR), 'tmp', f'{dataset}_enhanced_causal'),
            os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), 'tmp', f'{dataset}_enhanced_causal')
        ]
    else:
        potential_dirs = [
            os.path.join(BASE_DIR, 'tmp', f'{dataset}'),
            os.path.join(os.path.dirname(BASE_DIR), 'tmp', f'{dataset}'),
            os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), 'tmp', f'{dataset}')
        ]
    
    # Check each potential directory
    for results_dir in potential_dirs:
        if os.path.exists(results_dir):
            print(f"Found results directory: {results_dir}")
            
            # Find the best performing result file
            best_file = None
            best_performance = -float('inf')
            
            try:
                for file in os.listdir(results_dir):
                    if file.endswith('.csv') and not file.endswith('_sorted.csv') and not file.endswith('_results.csv'):
                        try:
                          
                            parts = file.split('_')
                            if len(parts) >= 2:
                                try:
                                    performance = float(parts[-2].replace('.csv', ''))
                                    if performance > best_performance:
                                        best_performance = performance
                                        best_file = file
                                except ValueError:
                                    continue
                        except:
                            continue
                
                if best_file is None:
                    # Try to find files with 'best' in the name
                    for file in os.listdir(results_dir):
                        if 'best' in file and file.endswith('.csv'):
                            best_file = file
                            # Try to extract performance from filename
                            try:
                                parts = file.split('_')
                                for part in parts:
                                    try:
                                        performance = float(part)
                                        best_performance = performance
                                        break
                                    except ValueError:
                                        continue
                            except:
                                best_performance = 0.0
                            break
                
                if best_file:
                    print(f"Using file: {best_file} with performance: {best_performance}")
                    model_df = pd.read_csv(os.path.join(results_dir, best_file))
                    return model_df, best_performance
            except Exception as e:
                print(f"Error processing directory {results_dir}: {e}")
                continue
    
    # If we reach here, no valid files were found
    raise FileNotFoundError(f"No result files found for {dataset} with {model_variant} model")

def train_models(dataset, base_path, task_type='reg'):
    """Train both causal and statistical models for SHAP comparison"""
    
    # Load original dataset
    data_path = os.path.join(base_path, f'{dataset}.hdf')
    
    try:
        df = pd.read_hdf(data_path)
    except:
        # Try CSV if HDF fails
        try:
            data_path = os.path.join(base_path, f'{dataset}.csv')
            df = pd.read_csv(data_path)
        except:
            raise FileNotFoundError(f"Dataset {dataset} not found at {data_path}")
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    feature_names = X.columns
    
    # Load causal knowledge
    try:
        causal_map = load_causal_map(dataset, base_path)
        direct_causes = [f for f in feature_names if causal_map.get(f, {}).get('group') == 'direct']
        print(f"Found {len(direct_causes)} direct causes")
    except:
        direct_causes = []
        causal_map = {}
        print("No causal map found, proceeding without causal knowledge")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load engineered features
    try:
        causal_df, causal_perf = load_model_results(dataset, 'causal')
        statistical_df, stat_perf = load_model_results(dataset, 'statistical')
        print(f"Loaded model results: Causal ({causal_perf:.4f}), Statistical ({stat_perf:.4f})")
    except Exception as e:
        print(f"Error loading model results: {e}")
        # Use original features if engineered features aren't available
        causal_df = statistical_df = df.copy()
    
    # Ensure target column exists
    y_col = df.columns[-1]
    
    # For causal model
    if y_col in causal_df.columns:
        X_causal = causal_df.drop(columns=[y_col])
        y_causal = causal_df[y_col]
    else:
        X_causal = causal_df
        y_causal = y
        
    # For statistical model
    if y_col in statistical_df.columns:
        X_statistical = statistical_df.drop(columns=[y_col])
        y_statistical = statistical_df[y_col]
    else:
        X_statistical = statistical_df
        y_statistical = y
        
    # Train models
    if task_type == 'reg':
        causal_model = RandomForestRegressor(n_estimators=100, random_state=42)
        statistical_model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        causal_model = RandomForestClassifier(n_estimators=100, random_state=42)
        statistical_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Use the same train/test split for both models
    causal_train_indices = np.arange(len(X_causal))[:int(0.8 * len(X_causal))]
    causal_test_indices = np.arange(len(X_causal))[int(0.8 * len(X_causal)):]
    
    statistical_train_indices = np.arange(len(X_statistical))[:int(0.8 * len(X_statistical))]
    statistical_test_indices = np.arange(len(X_statistical))[int(0.8 * len(X_statistical)):]
    
    # Train models
    causal_model.fit(X_causal.iloc[causal_train_indices], y_causal.iloc[causal_train_indices])
    statistical_model.fit(X_statistical.iloc[statistical_train_indices], y_statistical.iloc[statistical_train_indices])
    
    return {
        'causal_model': causal_model,
        'statistical_model': statistical_model,
        'X_causal': X_causal,
        'X_statistical': X_statistical,
        'y_causal': y_causal,
        'y_statistical': y_statistical,
        'causal_train_indices': causal_train_indices,
        'causal_test_indices': causal_test_indices,
        'statistical_train_indices': statistical_train_indices,
        'statistical_test_indices': statistical_test_indices,
        'causal_map': causal_map,
        'task_type': task_type
    }

def generate_shap_values(model_data):
    """Generate SHAP values for both causal and statistical models"""
    
    # Unpack data
    causal_model = model_data['causal_model']
    statistical_model = model_data['statistical_model']
    X_causal = model_data['X_causal']
    X_statistical = model_data['X_statistical']
    causal_test_indices = model_data['causal_test_indices']
    statistical_test_indices = model_data['statistical_test_indices']
    
    # Use test data for SHAP analysis
    X_causal_test = X_causal.iloc[causal_test_indices]
    X_statistical_test = X_statistical.iloc[statistical_test_indices]
    
    # Create SHAP explainers
    if hasattr(causal_model, 'predict_proba'):
        # For classifiers
        causal_explainer = shap.TreeExplainer(causal_model)
        statistical_explainer = shap.TreeExplainer(statistical_model)
        
        causal_shap = causal_explainer.shap_values(X_causal_test)
        statistical_shap = statistical_explainer.shap_values(X_statistical_test)
        
        # For multi-class, take the positive class
        if isinstance(causal_shap, list):
            causal_shap = causal_shap[1]  # Positive class
        if isinstance(statistical_shap, list):
            statistical_shap = statistical_shap[1]  # Positive class
    else:
        # For regressors
        causal_explainer = shap.TreeExplainer(causal_model)
        statistical_explainer = shap.TreeExplainer(statistical_model)
        
        causal_shap = causal_explainer.shap_values(X_causal_test)
        statistical_shap = statistical_explainer.shap_values(X_statistical_test)
    
    # Save explainers and SHAP values
    model_data['causal_explainer'] = causal_explainer
    model_data['statistical_explainer'] = statistical_explainer
    model_data['causal_shap'] = causal_shap
    model_data['statistical_shap'] = statistical_shap
    model_data['X_causal_test'] = X_causal_test
    model_data['X_statistical_test'] = X_statistical_test
    
    # Calculate expected values
    model_data['causal_expected_value'] = causal_explainer.expected_value
    model_data['statistical_expected_value'] = statistical_explainer.expected_value
    
    return model_data

def analyze_causal_alignment(model_data):
    """Analyze how well SHAP explanations align with causal knowledge"""
    
    causal_map = model_data['causal_map']
    X_causal = model_data['X_causal']
    X_statistical = model_data['X_statistical']
    causal_shap = model_data['causal_shap']
    statistical_shap = model_data['statistical_shap']
    
    # Map features to causal groups
    causal_features = list(X_causal.columns)
    statistical_features = list(X_statistical.columns)
    
    # Categorize features by causal group
    causal_group_map = {}
    for f in set(causal_features + statistical_features):
        group = causal_map.get(f, {}).get('group', 'other')
        causal_group_map[f] = group
    
    # Calculate mean absolute SHAP values for each feature
    causal_importance = np.abs(causal_shap).mean(axis=0)
    statistical_importance = np.abs(statistical_shap).mean(axis=0)
    
    # Map feature importances to causal groups
    causal_group_importances = {'direct': [], 'indirect': [], 'other': []}
    statistical_group_importances = {'direct': [], 'indirect': [], 'other': []}
    
    # Analyze causal model
    for i, feature in enumerate(causal_features):
        group = causal_group_map.get(feature, 'other')
        causal_group_importances[group].append(causal_importance[i])
    
    # Analyze statistical model
    for i, feature in enumerate(statistical_features):
        group = causal_group_map.get(feature, 'other')
        statistical_group_importances[group].append(statistical_importance[i])
    
    # Calculate mean importance by group
    causal_group_means = {k: np.mean(v) if v else 0 for k, v in causal_group_importances.items()}
    statistical_group_means = {k: np.mean(v) if v else 0 for k, v in statistical_group_importances.items()}
    
    # Calculate direct vs. indirect/other ratio (higher is better for causal understanding)
    if causal_group_means['indirect'] + causal_group_means['other'] > 0:
        causal_direct_ratio = causal_group_means['direct'] / (causal_group_means['indirect'] + causal_group_means['other'])
    else:
        causal_direct_ratio = float('inf')
        
    if statistical_group_means['indirect'] + statistical_group_means['other'] > 0:
        statistical_direct_ratio = statistical_group_means['direct'] / (statistical_group_means['indirect'] + statistical_group_means['other'])
    else:
        statistical_direct_ratio = float('inf')
    
    # When creating feature importances list, ensure values are scalar
    causal_feature_importances = []
    for i, feature in enumerate(causal_features):
        value = causal_importance[i]
        if isinstance(value, np.ndarray):
            value = float(np.mean(value))  # Convert array to scalar
        causal_feature_importances.append((feature, value))
    
    statistical_feature_importances = []
    for i, feature in enumerate(statistical_features):
        value = statistical_importance[i]
        if isinstance(value, np.ndarray):
            value = float(np.mean(value))  # Convert array to scalar
        statistical_feature_importances.append((feature, value))
    
    results = {
        'causal_group_means': causal_group_means,
        'statistical_group_means': statistical_group_means,
        'causal_direct_ratio': causal_direct_ratio,
        'statistical_direct_ratio': statistical_direct_ratio,
        'causal_group_importances': causal_group_importances,
        'statistical_group_importances': statistical_group_importances,
        'causal_feature_importances': causal_feature_importances,
        'statistical_feature_importances': statistical_feature_importances
    }
    
    return results

def analyze_explanation_stability(model_data, n_samples=100, noise_level=0.1):
    """Analyze how stable explanations are under small input perturbations"""
    
    X_causal_test = model_data['X_causal_test']
    X_statistical_test = model_data['X_statistical_test']
    causal_explainer = model_data['causal_explainer']
    statistical_explainer = model_data['statistical_explainer']
    
    # Select a few test instances for stability analysis
    if len(X_causal_test) > 5 and len(X_statistical_test) > 5:
        causal_test_indices = np.random.choice(len(X_causal_test), size=5, replace=False)
        statistical_test_indices = np.random.choice(len(X_statistical_test), size=5, replace=False)
    else:
        causal_test_indices = np.arange(min(5, len(X_causal_test)))
        statistical_test_indices = np.arange(min(5, len(X_statistical_test)))
    
    stability_results = {'causal': [], 'statistical': []}
    
    # Causal model stability
    for idx in causal_test_indices:
        instance = X_causal_test.iloc[idx:idx+1]
        
        # Get original SHAP values
        original_shap = causal_explainer.shap_values(instance)
        if isinstance(original_shap, list):
            original_shap = original_shap[1]  # Take positive class for classification
        
        # Create perturbed versions
        variations = []
        
        for _ in range(n_samples):
            # Create noise (proportional to feature values)
            noise = np.random.normal(0, noise_level, instance.shape) * instance.values
            
            # Apply noise
            perturbed = pd.DataFrame(
                instance.values + noise,
                columns=instance.columns
            )
            
            # Get SHAP values for perturbed instance
            perturbed_shap = causal_explainer.shap_values(perturbed)
            if isinstance(perturbed_shap, list):
                perturbed_shap = perturbed_shap[1]
            
            # Calculate variation
            variations.append(np.abs(perturbed_shap - original_shap).mean())
        
        # Store stability metric
        stability_results['causal'].append(np.mean(variations))
    
    # Statistical model stability
    for idx in statistical_test_indices:
        instance = X_statistical_test.iloc[idx:idx+1]
        
        # Get original SHAP values
        original_shap = statistical_explainer.shap_values(instance)
        if isinstance(original_shap, list):
            original_shap = original_shap[1]
        
        # Create perturbed versions
        variations = []
        
        for _ in range(n_samples):
            # Create noise
            noise = np.random.normal(0, noise_level, instance.shape) * instance.values
            
            # Apply noise
            perturbed = pd.DataFrame(
                instance.values + noise,
                columns=instance.columns
            )
            
            # Get SHAP values for perturbed instance
            perturbed_shap = statistical_explainer.shap_values(perturbed)
            if isinstance(perturbed_shap, list):
                perturbed_shap = perturbed_shap[1]
            
            # Calculate variation
            variations.append(np.abs(perturbed_shap - original_shap).mean())
        
        # Store stability metric
        stability_results['statistical'].append(np.mean(variations))
    
    # Calculate overall stability (lower means more stable)
    causal_stability = np.mean(stability_results['causal'])
    statistical_stability = np.mean(stability_results['statistical'])
    
    return {
        'causal_stability': causal_stability,
        'statistical_stability': statistical_stability,
        'per_instance': stability_results
    }

def create_interpretability_visualizations(dataset, model_data, alignment_results, stability_results_by_noise, output_dir):
    """Create publication-quality visualizations for the AAAI paper"""
    
    os.makedirs(output_dir, exist_ok=True)
    dataset_dir = os.path.join(output_dir, dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Figure 1: SHAP Summary Plot Comparison
    create_summary_plot_comparison(dataset, model_data, dataset_dir)
    
    # Figure 2: Group-level Feature Importance Comparison
    create_group_importance_plot(dataset, alignment_results, dataset_dir)
    
    # Figure 3: Explanation Stability Analysis across noise levels
    create_stability_plot(dataset, stability_results_by_noise, dataset_dir)
    
    # Figure 4: Comprehensive Multi-panel Figure for AAAI
    create_comprehensive_figure(dataset, model_data, alignment_results, stability_results_by_noise, dataset_dir)

def create_summary_plot_comparison(dataset, model_data, output_dir):
    """Create side-by-side SHAP summary plots"""
    
    fig = plt.figure(figsize=(7, 5))
    
    # Causal model summary plot
    plt.subplot(1, 2, 1)
    shap.summary_plot(
        model_data['causal_shap'], 
        model_data['X_causal_test'],
        plot_type="bar", 
        show=False,
        max_display=10
    )
    plt.title(f"Causal Model", fontweight='bold', fontsize=10)
    
    # Statistical model summary plot
    plt.subplot(1, 2, 2)
    shap.summary_plot(
        model_data['statistical_shap'],
        model_data['X_statistical_test'],
        plot_type="bar",
        show=False,
        max_display=10
    )
    plt.title(f"Statistical Model", fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset}_shap_summary_comparison.pdf"), bbox_inches='tight')
    plt.close()

def create_group_importance_plot(dataset, alignment_results, output_dir):
    """Create causal group importance comparison plot"""
    
    causal_means = alignment_results['causal_group_means']
    statistical_means = alignment_results['statistical_group_means']
    
    groups = ['direct', 'indirect', 'other']
    causal_values = [causal_means.get(g, 0) for g in groups]
    statistical_values = [statistical_means.get(g, 0) for g in groups]
    
    fig, ax = plt.subplots(figsize=(4, 3))
    
    x = np.arange(len(groups))
    width = 0.35
    
    ax.bar(x - width/2, causal_values, width, label='Causal Model', color='#1f77b4', edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, statistical_values, width, label='Statistical Model', color='#ff7f0e', edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for i, v in enumerate(causal_values):
        ax.text(i - width/2, v + 0.01, f"{v:.3f}", ha='center', fontsize=7)
    for i, v in enumerate(statistical_values):
        ax.text(i + width/2, v + 0.01, f"{v:.3f}", ha='center', fontsize=7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Direct Causes', 'Indirect Causes', 'Other Features'])
    ax.set_ylabel('Mean |SHAP Value|')
    ax.set_title(f'Feature Importance by Causal Group: {dataset}')
    ax.legend()
    
    # Add direct ratio information
    plt.figtext(0.5, 0.01, 
                f"Direct-to-Other Ratio: Causal={alignment_results['causal_direct_ratio']:.2f}, " + 
                f"Statistical={alignment_results['statistical_direct_ratio']:.2f}",
                ha='center', fontsize=8)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(output_dir, f"{dataset}_causal_group_importance.pdf"), bbox_inches='tight')
    plt.close()

def create_stability_plot(dataset, stability_results_by_noise, output_dir):
    """Create explanation stability comparison plot across noise levels"""
    
    noise_levels = list(stability_results_by_noise.keys())
    causal_stabilities = [stability_results_by_noise[n]['causal_stability'] for n in noise_levels]
    statistical_stabilities = [stability_results_by_noise[n]['statistical_stability'] for n in noise_levels]
    
    # Calculate stability improvements
    stability_improvements = [(s - c) / s * 100 if s > 0 else 0 
                             for c, s in zip(causal_stabilities, statistical_stabilities)]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot line chart showing stability across noise levels
    ax.plot(noise_levels, causal_stabilities, 'o-', color='#1f77b4', 
            linewidth=2, label='Causal Model', markersize=8)
    ax.plot(noise_levels, statistical_stabilities, 's-', color='#ff7f0e', 
            linewidth=2, label='Statistical Model', markersize=8)
    
    # Add value labels
    for i, (noise, causal, statistical) in enumerate(zip(noise_levels, causal_stabilities, statistical_stabilities)):
        ax.text(noise, causal + 0.0002, f"{causal:.5f}", ha='center', va='bottom', fontsize=8)
        ax.text(noise, statistical + 0.0002, f"{statistical:.5f}", ha='center', va='bottom', fontsize=8)
        
        # Add improvement annotation
        ax.annotate(f"{stability_improvements[i]:.1f}%", 
                   xy=(noise, (causal + statistical)/2),
                   xytext=(noise + 0.02, (causal + statistical)/2 + 0.0005),
                   arrowprops=dict(arrowstyle='->', color='green', linewidth=1),
                   ha='left', va='center', fontsize=8, fontweight='bold', color='green')
    
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Mean SHAP Variation (Lower is Better)')
    ax.set_title(f'Explanation Stability vs. Noise Level: {dataset}')
    ax.legend()
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset}_explanation_stability_by_noise.pdf"), bbox_inches='tight')
    plt.close()
    
    # Create bar chart version for the default noise level (first one)
    if noise_levels:
        default_noise = noise_levels[0]
        stability_results = stability_results_by_noise[default_noise]
        
        fig, ax = plt.subplots(figsize=(4, 3))
        
        models = ['Causal', 'Statistical']
        stability_values = [stability_results['causal_stability'], stability_results['statistical_stability']]
        
        bars = ax.bar(models, stability_values, color=['#1f77b4', '#ff7f0e'], edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.4f}', ha='center', fontsize=9)
        
        ax.set_ylabel('Mean SHAP Variation Under Noise')
        ax.set_title(f'Explanation Stability (Noise={default_noise}): {dataset}\n(Lower is Better)')
        
        # Add stability improvement percentage
        statistical_stability = stability_results['statistical_stability']
        causal_stability = stability_results['causal_stability']
        if statistical_stability > 0:
            improvement = (statistical_stability - causal_stability) / statistical_stability * 100
            plt.figtext(0.5, 0.01, f"Causal model explanations are {improvement:.1f}% more stable",
                      ha='center', fontsize=8)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(os.path.join(output_dir, f"{dataset}_explanation_stability.pdf"), bbox_inches='tight')
        plt.close()

def create_feature_ranking_panel(ax, model_data, alignment_results):
    """Create feature ranking alignment panel"""
    
    causal_feature_importances = alignment_results.get('causal_feature_importances', [])
    statistical_feature_importances = alignment_results.get('statistical_feature_importances', [])
    causal_map = model_data['causal_map']
    
    # Convert numpy arrays to floats for sorting
    causal_feature_importances_fixed = []
    for feature, importance in causal_feature_importances:
        # If importance is a numpy array, take its mean
        if isinstance(importance, np.ndarray):
            importance = float(np.mean(np.abs(importance)))
        causal_feature_importances_fixed.append((feature, float(importance)))
    
    statistical_feature_importances_fixed = []
    for feature, importance in statistical_feature_importances:
        # If importance is a numpy array, take its mean
        if isinstance(importance, np.ndarray):
            importance = float(np.mean(np.abs(importance)))
        statistical_feature_importances_fixed.append((feature, float(importance)))
    
    # Sort by importance
    causal_feature_importances = sorted(causal_feature_importances_fixed, key=lambda x: x[1], reverse=True)
    statistical_feature_importances = sorted(statistical_feature_importances_fixed, key=lambda x: x[1], reverse=True)
    
    # Rest of the function remains the same...
    # Get top features
    causal_top_features = causal_feature_importances[:5]
    statistical_top_features = statistical_feature_importances[:5]
    
    # Calculate alignment with causality
    causal_direct_count = sum(1 for f, _ in causal_top_features 
                             if causal_map.get(f, {}).get('group') == 'direct')
    
    statistical_direct_count = sum(1 for f, _ in statistical_top_features 
                                 if causal_map.get(f, {}).get('group') == 'direct')
    
    
    causal_alignment = causal_direct_count / len(causal_top_features) * 100 if causal_top_features else 0
    statistical_alignment = statistical_direct_count / len(statistical_top_features) * 100 if statistical_top_features else 0
    
    # Create table
    table_data = [['Model', 'Top Features', 'Direct Cause %']]
    
    # Create table rows for causal model
    causal_features_text = '\n'.join([
        f"{i+1}. {f} ({'⭐' if causal_map.get(f, {}).get('group') == 'direct' else '○'})"
        for i, (f, _) in enumerate(causal_top_features)
    ])
    table_data.append(['Causal', causal_features_text, f"{causal_alignment:.1f}%"])
    
    # Create table rows for statistical model
    statistical_features_text = '\n'.join([
        f"{i+1}. {f} ({'⭐' if causal_map.get(f, {}).get('group') == 'direct' else '○'})"
        for i, (f, _) in enumerate(statistical_top_features)
    ])
    table_data.append(['Statistical', statistical_features_text, f"{statistical_alignment:.1f}%"])
    
    # Create table
    ax.axis('off')
    table = ax.table(
        cellText=table_data,
        cellLoc='left',
        loc='center',
        colWidths=[0.2, 0.6, 0.2]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    
    # Style the header row
    for j in range(len(table_data[0])):
        table[(0, j)].set_facecolor('#CCCCCC')
        table[(0, j)].set_text_props(weight='bold')
    
    # Style the data rows
    for i in range(1, len(table_data)):
        # Highlight causal model row if it has better alignment
        if i == 1 and causal_alignment > statistical_alignment:
            for j in range(len(table_data[0])):
                table[(i, j)].set_facecolor('#D8F0D8')
    
    # Add a legend
    ax.text(0.5, -0.02, "⭐ = Direct Cause, ○ = Indirect/Other", 
            ha='center', va='top', transform=ax.transAxes, fontsize=7,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3'))

def create_stability_panel(ax, stability_results_by_noise):
    """Create stability analysis panel with multiple noise levels"""
    
    noise_levels = sorted(list(stability_results_by_noise.keys()))
    causal_stabilities = [stability_results_by_noise[n]['causal_stability'] for n in noise_levels]
    statistical_stabilities = [stability_results_by_noise[n]['statistical_stability'] for n in noise_levels]
    
    # Plot lines showing stability vs noise level
    ax.plot(noise_levels, causal_stabilities, 'o-', color='#1f77b4', 
            linewidth=1.5, label='Causal', markersize=5)
    ax.plot(noise_levels, statistical_stabilities, 's-', color='#ff7f0e', 
            linewidth=1.5, label='Statistical', markersize=5)
    
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Mean SHAP Variation\n(Lower is Better)')
    ax.legend(fontsize=7)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add annotation for average improvement
    improvements = [(s - c) / s * 100 if s > 0 else 0 
                   for c, s in zip(causal_stabilities, statistical_stabilities)]
    avg_improvement = np.mean(improvements)
    
    ax.text(0.03, 0.03, f"Avg. Improvement: {avg_improvement:.1f}%", 
            transform=ax.transAxes, fontsize=7,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.2'))

def create_instance_comparison(ax, model_data):
    """Create instance-level explanation comparison"""
    
    try:
        # Get explainers and test data
        causal_explainer = model_data['causal_explainer']
        statistical_explainer = model_data['statistical_explainer']
        X_causal_test = model_data['X_causal_test']
        X_statistical_test = model_data['X_statistical_test']
        causal_map = model_data['causal_map']
        
        # Select an instance (same index for both models)
        instance_idx = 0
        
        # Get causal explanation
        causal_instance = X_causal_test.iloc[instance_idx:instance_idx+1]
        causal_shap_values = causal_explainer.shap_values(causal_instance)
        if isinstance(causal_shap_values, list):
            causal_shap_values = causal_shap_values[1]  # Take positive class for classification
            
        # Get statistical explanation
        statistical_instance = X_statistical_test.iloc[instance_idx:instance_idx+1]
        statistical_shap_values = statistical_explainer.shap_values(statistical_instance)
        if isinstance(statistical_shap_values, list):
            statistical_shap_values = statistical_shap_values[1]
        
        # Get top features for each model
        causal_features = list(X_causal_test.columns)
        statistical_features = list(X_statistical_test.columns)
        
        causal_importances = np.abs(causal_shap_values[0])
        statistical_importances = np.abs(statistical_shap_values[0])
        
        # Get top 5 features for each
        causal_top_indices = np.argsort(causal_importances)[-5:][::-1]
        statistical_top_indices = np.argsort(statistical_importances)[-5:][::-1]
        
        causal_top_features = [(causal_features[i], causal_shap_values[0][i]) for i in causal_top_indices]
        statistical_top_features = [(statistical_features[i], statistical_shap_values[0][i]) for i in statistical_top_indices]
        
        # Create split bar chart
        feature_names = []
        causal_values = []
        statistical_values = []
        
        # Combine all top feature names
        all_top_features = set([f for f, _ in causal_top_features] + [f for f, _ in statistical_top_features])
        
        for f in all_top_features:
            feature_names.append(f)
            
            # Find value in causal model (0 if not present)
            causal_value = 0
            for feature, value in causal_top_features:
                if feature == f:
                    causal_value = value
                    break
            causal_values.append(causal_value)
            
            # Find value in statistical model (0 if not present)
            statistical_value = 0
            for feature, value in statistical_top_features:
                if feature == f:
                    statistical_value = value
                    break
            statistical_values.append(statistical_value)
        
        # Sort by combined absolute importance
        combined_importance = [abs(c) + abs(s) for c, s in zip(causal_values, statistical_values)]
        sorted_indices = np.argsort(combined_importance)
        
        feature_names = [feature_names[i] for i in sorted_indices]
        causal_values = [causal_values[i] for i in sorted_indices]
        statistical_values = [statistical_values[i] for i in sorted_indices]
        
        # Plot bars side by side
        y_pos = np.arange(len(feature_names))
        
        # Causal bars
        causal_bars = ax.barh(y_pos - 0.2, causal_values, height=0.4, color='#1f77b4', 
                              edgecolor='black', linewidth=0.5, label='Causal Model')
        
        # Statistical bars
        statistical_bars = ax.barh(y_pos + 0.2, statistical_values, height=0.4, color='#ff7f0e',
                                  edgecolor='black', linewidth=0.5, label='Statistical Model')
        
        # Feature labels with causal indicators
        feature_labels = []
        for f in feature_names:
            group = causal_map.get(f, {}).get('group', 'other')
            if group == 'direct':
                feature_labels.append(f"{f} ⭐")
            else:
                feature_labels.append(f"{f}")
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_labels, fontsize=7)
        ax.set_xlabel('SHAP value (impact on prediction)')
        ax.legend(fontsize=7)
        
        # Add prediction information
        causal_expected = model_data['causal_expected_value']
        if isinstance(causal_expected, list):
            causal_expected = causal_expected[1]
            
        statistical_expected = model_data['statistical_expected_value']
        if isinstance(statistical_expected, list):
            statistical_expected = statistical_expected[1]
            
        info_text = (
            f"Base values:\n"
            f"Causal: {causal_expected:.3f}\n"
            f"Statistical: {statistical_expected:.3f}"
        )
        
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=7,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3'))
        
    except Exception as e:
        # Fallback if instance comparison fails
        ax.text(0.5, 0.5, f"Instance comparison not available\n{str(e)}", 
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

def create_comprehensive_figure(dataset, model_data, alignment_results, stability_results_by_noise, output_dir):
    """Create a comprehensive multi-panel figure for AAAI paper"""
    
    # Set up the figure with a layout suitable for AAAI two-column format
    fig = plt.figure(figsize=(7.3, 6))
    gs = GridSpec(2, 3, width_ratios=[1.2, 1, 1], height_ratios=[1.2, 1])
    
    # Panel A: Feature importance by causal group
    ax1 = fig.add_subplot(gs[0, :2])
    create_group_importance_panel(ax1, alignment_results)
    ax1.set_title(f"A) Feature Importance by Causal Group", fontweight="bold")
    
    # Panel B: Top feature alignment with causality
    ax2 = fig.add_subplot(gs[0, 2])
    create_feature_ranking_panel(ax2, model_data, alignment_results)
    ax2.set_title(f"B) Feature Ranking", fontweight="bold")
    
    # Panel C: Explanation stability across noise levels
    ax3 = fig.add_subplot(gs[1, 0])
    create_stability_panel(ax3, stability_results_by_noise)
    ax3.set_title(f"C) Explanation Stability", fontweight="bold")
    
    # Panel D: Sample instance explanation comparison
    ax4 = fig.add_subplot(gs[1, 1:])
    create_instance_comparison(ax4, model_data)
    ax4.set_title(f"D) Sample Instance Explanation", fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset}_interpretability_comprehensive.pdf"), bbox_inches='tight')
    plt.close()

def create_group_importance_panel(ax, alignment_results):
    """Create causal group importance panel"""
    
    causal_means = alignment_results['causal_group_means']
    statistical_means = alignment_results['statistical_group_means']
    
    groups = ['direct', 'indirect', 'other']
    causal_values = [causal_means.get(g, 0) for g in groups]
    statistical_values = [statistical_means.get(g, 0) for g in groups]
    
    x = np.arange(len(groups))
    width = 0.35
    
    ax.bar(x - width/2, causal_values, width, label='Causal Model', color='#1f77b4', edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, statistical_values, width, label='Statistical Model', color='#ff7f0e', edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for i, v in enumerate(causal_values):
        ax.text(i - width/2, v + 0.01, f"{v:.3f}", ha='center', fontsize=7)
    for i, v in enumerate(statistical_values):
        ax.text(i + width/2, v + 0.01, f"{v:.3f}", ha='center', fontsize=7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Direct\nCauses', 'Indirect\nCauses', 'Other\nFeatures'])
    ax.set_ylabel('Mean |SHAP Value|')
    ax.legend(fontsize=7, loc='upper right')

def create_cross_dataset_summary(results, output_dir):
    """Create a summary of interpretability results across datasets"""
    
    # Extract key metrics
    datasets = list(results.keys())
    noise_levels = list(results[datasets[0]]['stability_by_noise'].keys()) if datasets else []
    
    # Create a multi-panel figure for each noise level
    for noise_level in noise_levels:
        fig, ax = plt.subplots(figsize=(7, 4))
        
        # Collect stability metrics for this noise level
        causal_stabilities = [results[d]['stability_by_noise'][noise_level]['causal_stability'] 
                             for d in datasets]
        statistical_stabilities = [results[d]['stability_by_noise'][noise_level]['statistical_stability'] 
                                  for d in datasets]
        
        # Calculate improvement percentages
        stability_improvements = [(s - c) / (s if s > 0 else 1) * 100 
                                 for c, s in zip(causal_stabilities, statistical_stabilities)]
        
        # Panel: Explanation Stability
        x = np.arange(len(datasets))
        width = 0.35
        
        ax.bar(x - width/2, causal_stabilities, width, label='Causal Model', 
               color='#1f77b4', edgecolor='black', linewidth=0.5)
        ax.bar(x + width/2, statistical_stabilities, width, label='Statistical Model', 
               color='#ff7f0e', edgecolor='black', linewidth=0.5)
        
        # Add improvement annotations
        for i, (c, s, imp) in enumerate(zip(causal_stabilities, statistical_stabilities, stability_improvements)):
            # Add improvement percentage
            ax.annotate(f"{imp:.1f}%", 
                       xy=(i, (c + s)/2),
                       xytext=(i, (c + s)/2 + max(statistical_stabilities)/10),
                       arrowprops=dict(arrowstyle='->', color='green', linewidth=1),
                       ha='center', va='bottom', fontsize=8, fontweight='bold', color='green')
        
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.set_ylabel('Explanation Variation (Lower is Better)')
        ax.set_title(f'Explanation Stability Across Datasets (Noise Level = {noise_level})')
        ax.legend(fontsize=8)
        
        # Add summary text
        plt.figtext(0.5, 0.01, 
                    f"Average Stability Improvement: {np.mean(stability_improvements):.1f}%",
                    ha='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(os.path.join(output_dir, f"cross_dataset_stability_noise_{noise_level}.pdf"), 
                    bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f"cross_dataset_stability_noise_{noise_level}.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create a summary plot showing stability improvement vs noise level
    if datasets and noise_levels:
        fig, ax = plt.subplots(figsize=(7, 4))
        
        # For each dataset, plot improvement vs noise level
        for dataset in datasets:
            improvements = []
            for noise in noise_levels:
                c_stab = results[dataset]['stability_by_noise'][noise]['causal_stability']
                s_stab = results[dataset]['stability_by_noise'][noise]['statistical_stability']
                imp = (s_stab - c_stab) / (s_stab if s_stab > 0 else 1) * 100
                improvements.append(imp)
            
            ax.plot(noise_levels, improvements, 'o-', linewidth=1.5, label=dataset, markersize=6)
        
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('Stability Improvement (%)')
        ax.set_title('Causal Model Stability Advantage vs. Noise Level')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "stability_improvement_vs_noise.pdf"), bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, "stability_improvement_vs_noise.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save numeric results
    summary_data = []
    
    for dataset in datasets:
        for noise_level in noise_levels:
            stability_result = results[dataset]['stability_by_noise'][noise_level]
            causal_stab = stability_result['causal_stability']
            stat_stab = stability_result['statistical_stability']
            
            # Calculate stability improvement
            if stat_stab > 0:
                stab_improvement = (stat_stab - causal_stab) / stat_stab * 100
            else:
                stab_improvement = 0.0
                
            summary_data.append({
                'Dataset': dataset,
                'Noise_Level': noise_level,
                'Causal_Stability': causal_stab,
                'Statistical_Stability': stat_stab,
                'Stability_Improvement_%': stab_improvement
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(output_dir, "interpretability_summary.csv"), index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Datasets analyzed: {len(datasets)}")
    
    # Calculate average improvement across noise levels
    if datasets and noise_levels:
        avg_improvements_by_noise = {}
        for noise in noise_levels:
            improvements = []
            for dataset in datasets:
                c_stab = results[dataset]['stability_by_noise'][noise]['causal_stability']
                s_stab = results[dataset]['stability_by_noise'][noise]['statistical_stability']
                if s_stab > 0:
                    imp = (s_stab - c_stab) / s_stab * 100
                    improvements.append(imp)
            
            if improvements:
                avg_improvements_by_noise[noise] = np.mean(improvements)
        
        for noise, avg_imp in avg_improvements_by_noise.items():
            print(f"Average explanation stability improvement at noise level {noise}: {avg_imp:.1f}%")
    
    print("="*80)

def run_interpretability_experiment(datasets, base_path, output_base_dir, n_samples=100, noise_levels=[0.1]):
    """Run the full interpretability experiment across multiple datasets and noise levels"""
    
    results = {}
    
    for dataset in datasets:
        print(f"\n{'='*50}\nAnalyzing dataset: {dataset}\n{'='*50}")
        
        # Create output directory for this dataset
        output_dir = os.path.join(output_base_dir, dataset)
        os.makedirs(output_dir, exist_ok=True)
        
        
        task_type = task_dict.get(dataset, 'reg')  # Default to regression
        
        # Train models
        try:
            print(f"Training models for {dataset}...")
            model_data = train_models(dataset, base_path, task_type)
            
            # Generate SHAP values
            print("Generating SHAP explanations...")
            model_data = generate_shap_values(model_data)
            
            # Analyze causal alignment
            print("Analyzing alignment with causal knowledge...")
            alignment_results = analyze_causal_alignment(model_data)
            
            # Analyze explanation stability across multiple noise levels
            print("Analyzing explanation stability across noise levels...")
            stability_results_by_noise = {}
            
            for noise_level in noise_levels:
                print(f"  - Analyzing with noise level {noise_level}...")
                stability_results = analyze_explanation_stability(
                    model_data, 
                    n_samples=n_samples, 
                    noise_level=noise_level
                )
                stability_results_by_noise[noise_level] = stability_results
            
            # Create visualizations
            print("Creating visualizations...")
            create_interpretability_visualizations(
                dataset, 
                model_data, 
                alignment_results, 
                stability_results_by_noise, 
                output_dir
            )
            
            # Store results
            results[dataset] = {
                'alignment': alignment_results,
                'stability_by_noise': stability_results_by_noise
            }
            
            print(f"Completed analysis for {dataset}")
            
        except Exception as e:
            print(f"Error processing dataset {dataset}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create summary report across datasets
    create_cross_dataset_summary(results, output_base_dir)
    
    return results

# Main execution
if __name__ == "__main__":
    import argparse
    import os
    from datetime import datetime
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='SHAP Interpretability Analysis for Causal Feature Engineering')
    parser.add_argument('--datasets', nargs='+', 
                        default=['wine_white', 'amazon_employee', 'openml_586', 'openml_616'],
                        help='List of datasets to analyze')
    parser.add_argument('--output-dir', type=str, 
                        default=os.path.join(BASE_DIR, 'interpretability_results', f'interpretability_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'),
                        help='Output directory for results and visualizations')
    parser.add_argument('--data-dir', type=str, 
                        default=os.path.join(BASE_DIR, 'Data', 'GRFG_data'),
                        help='Directory containing the datasets')
    parser.add_argument('--n-samples', type=int, default=100,
                        help='Number of samples for stability analysis')
    parser.add_argument('--noise-level', type=float, nargs='+', default=[0.1, 0.3, 0.5, 0.7],
                        help='Noise levels for stability analysis')
    
    args = parser.parse_args()
    
    # Ensure the base path exists
    if not os.path.exists(args.data_dir):
        args.data_dir = os.path.join(os.path.dirname(BASE_DIR), 'Data', 'GRFG_data')
        if not os.path.exists(args.data_dir):
            raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log experiment settings
    print(f"Starting interpretability experiment with:")
    print(f"  Datasets: {args.datasets}")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Stability samples: {args.n_samples}")
    print(f"  Noise levels: {args.noise_level}")
    
    # Run experiment with multiple noise levels
    results = run_interpretability_experiment(
        datasets=args.datasets,
        base_path=args.data_dir,
        output_base_dir=args.output_dir,
        n_samples=args.n_samples,
        noise_levels=args.noise_level
    )
    
    print(f"Interpretability analysis complete. Results saved to: {args.output_dir}")