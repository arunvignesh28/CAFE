import os
import sys
import json
import psutil
import wandb
import numpy as np
import pandas as pd
import time
import math
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_regression, VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, PowerTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import torch
from sklearn.manifold import TSNE
from logger import *
from Operation import add_unary, operation_set, O1, O2, sep_token
os.environ['NUMEXPR_MAX_THREADS'] = '32'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
torch.set_num_threads(8)
info(torch.get_num_threads())
info(torch.__config__.parallel_info())
import warnings
from nni.utils import merge_parameter
import nni
import matplotlib.pyplot as plt
torch.manual_seed(0)
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')
from DQN import DQN1, DQN2, DDQN1, DDQN2, DuelingDQN1, DuelingDDQN2, DuelingDQN2, DuelingDDQN1
from tools import *
from task_mapping import task_dict, task_type, base_path, task_measure, state_rep, support_rl_method
import argparse
import pickle
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from tools import estimate_causal_effect
from pgmpy.estimators import PC
from functools import lru_cache
import networkx as nx

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def init_param():
    parser = argparse.ArgumentParser(description='Enhanced Causal-GRFG Experiment')
    parser.add_argument('--file-name', type=str, default='german_credit', help='data name')
    parser.add_argument('--log-level', type=str, default='info', help='log level')
    parser.add_argument('--task', type=str, default='reg', help='ng/cls/reg/det/rank')
    parser.add_argument('--episodes', type=int, default=10, help='episodes for training')
    parser.add_argument('--steps', type=int, default=5, help='steps for each episode')
    parser.add_argument('--memory', type=int, default=32, help='memory capacity')
    parser.add_argument('--eps_start', type=float, default=0.95, help='eps start')
    parser.add_argument('--eps_end', type=float, default=0.2, help='eps end')
    parser.add_argument('--eps_decay', type=int, default=200, help='eps decay')
    parser.add_argument('--a', type=float, default=1, help='a')
    parser.add_argument('--b', type=float, default=1, help='b')
    parser.add_argument('--c', type=float, default=1, help='c')
    parser.add_argument('--rl-method', type=str, default='dqn', help='used reinforcement methods')
    parser.add_argument('--state-method', type=str, default='gcn', help='state representation method')
    parser.add_argument('--default-cuda', type=int, default=1, help='the using cuda')
    parser.add_argument('--strategy-mode', type=str, default='hybrid', help='hybrid/causal_only/traditional', 
                       choices=['hybrid', 'causal_only', 'traditional'])
    parser.add_argument('--max-per-cluster', type=int, default=25, help='max features per causal cluster')
    parser.add_argument('--max-binary-combinations', type=int, default=50, help='max binary feature combinations')
    args, _ = parser.parse_known_args()
    return args

def init_feature_cache(feature_names, causal_map):
    """Pre-compute and cache causal group assignments"""
    cluster_dict = {'direct': 0, 'indirect': 1, 'other': 2}
    feature_group_cache = {}
    feature_cluster_cache = {}
    
    for f in feature_names:
        group = causal_map.get(f, {}).get('group', 'other') 
        feature_group_cache[f] = group
        feature_cluster_cache[f] = cluster_dict.get(group, 2)
    
    return feature_group_cache, feature_cluster_cache

def cluster_features_causal_enhanced(X, feature_names, causal_map, feature_cluster_cache, max_per_cluster=25):
    """
    Enhanced clustering that combines causal knowledge with statistical methods
    """
    
    # Step 1: Primary causal clustering
    clusters = np.zeros(len(feature_names), dtype=np.int32)
    for i, f in enumerate(feature_names):
        if f not in feature_cluster_cache:
            group = causal_map.get(f, {}).get('group', 'other')
            feature_cluster_cache[f] = {'direct': 0, 'indirect': 1, 'other': 2}.get(group, 2)
        clusters[i] = feature_cluster_cache[f]
    
    # Step 2: Sub-clustering within large causal groups
    feature_cluster_dict = {}
    target = X[:, -1] if X.shape[1] > len(feature_names) else None
    
    for cluster_id in np.unique(clusters):
        indices = np.where(clusters == cluster_id)[0]
        
        if len(indices) <= max_per_cluster:
            feature_cluster_dict[cluster_id] = indices.tolist()
        else:
            info(f"Sub-clustering causal cluster {cluster_id}: {len(indices)} features")
            
            X_cluster = X[:, indices]
            
            # Method 1: Mutual Information based selection
            if target is not None:
                try:
                    mi_scores = mutual_info_regression(X_cluster, target)
                    top_indices = np.argsort(mi_scores)[-max_per_cluster:]
                    feature_cluster_dict[cluster_id] = indices[top_indices].tolist()
                    info(f"Selected {len(top_indices)} features from cluster {cluster_id} using MI")
                    continue
                except Exception as e:
                    info(f"MI selection failed: {e}")
            
            # Method 2: Correlation-based sub-clustering (fallback)
            try:
                corr_matrix = np.corrcoef(X_cluster.T)
                selected_indices = []
                for i in range(len(indices)):
                    if len(selected_indices) >= max_per_cluster:
                        break
                    
                    is_unique = True
                    for j in selected_indices:
                        if abs(corr_matrix[i, j]) > 0.9:
                            is_unique = False
                            break
                    
                    if is_unique:
                        selected_indices.append(i)
                
                if len(selected_indices) > 0:
                    feature_cluster_dict[cluster_id] = indices[selected_indices].tolist()
                    info(f"Selected {len(selected_indices)} features from cluster {cluster_id} using correlation filtering")
                else:
                    np.random.seed(42 + cluster_id)
                    selected = np.random.choice(indices, min(max_per_cluster, len(indices)), replace=False)
                    feature_cluster_dict[cluster_id] = selected.tolist()
                    
            except Exception as e:
                info(f"Correlation clustering failed: {e}, using random selection")
                np.random.seed(42 + cluster_id)
                selected = np.random.choice(indices, min(max_per_cluster, len(indices)), replace=False)
                feature_cluster_dict[cluster_id] = selected.tolist()
    
    return feature_cluster_dict

def adaptive_epsilon_decay(steps_done, performance_history, eps_start, eps_end, eps_decay):
    """Adaptive epsilon that adjusts based on performance trends"""
    base_eps = eps_end + (eps_start - eps_end) * math.exp(-1.0 * steps_done / eps_decay)
    
    if len(performance_history) >= 10:
        recent_avg = np.mean(performance_history[-5:])
        older_avg = np.mean(performance_history[-10:-5])
        
        if recent_avg <= older_avg:
            exploration_boost = 0.1
            info(f"Performance stagnating, boosting exploration by {exploration_boost}")
        else:
            exploration_boost = 0.0
    else:
        exploration_boost = 0.0
    
    return min(0.95, base_eps + exploration_boost)

def hybrid_causal_feature_selection(op, feature_cluster_dict, Dg, feature_names, 
                                   causal_map, feature_group_cache, eps_threshold,
                                   dqn_cluster1, dqn_cluster2, operation_emb, 
                                   STATE_METHOD, DEVICE, step_count, strategy_mode='hybrid'):
    """
    Hybrid approach: Prioritize causal combinations, maintain top-k traditional exploration
    """
    
    if strategy_mode == 'causal_only':
        strategy_weights = {'causal_hierarchical': 1.0}
    elif strategy_mode == 'traditional':
        strategy_weights = {'traditional_topk': 0.8, 'random_exploration': 0.2}
    else:  # hybrid
        if step_count < 5:  # Early exploration
            strategy_weights = {
                'causal_hierarchical': 0.5,
                'traditional_topk': 0.3, 
                'random_exploration': 0.2
            }
        else:  # Later exploitation
            strategy_weights = {
                'causal_hierarchical': 0.7,
                'traditional_topk': 0.2,
                'random_exploration': 0.1
            }
    
    strategy = np.random.choice(
        list(strategy_weights.keys()), 
        p=list(strategy_weights.values())
    )
    
    info(f"Selected strategy: {strategy} for operation: {op}")
    
    if strategy == 'causal_hierarchical':
        return causal_hierarchical_selection(op, feature_cluster_dict, Dg, feature_names, 
                                           causal_map, feature_group_cache, eps_threshold,
                                           dqn_cluster1, dqn_cluster2, operation_emb, 
                                           STATE_METHOD, DEVICE), strategy
    elif strategy == 'traditional_topk':
        return traditional_topk_selection(op, feature_cluster_dict, Dg, feature_names,
                                        eps_threshold, dqn_cluster1, dqn_cluster2, 
                                        operation_emb, STATE_METHOD, DEVICE), strategy
    else:  # random_exploration
        return random_exploration_selection(op, feature_cluster_dict, Dg, feature_names,
                                          eps_threshold, dqn_cluster1, dqn_cluster2,
                                          operation_emb, STATE_METHOD, DEVICE), strategy

def causal_hierarchical_selection(op, feature_cluster_dict, Dg, feature_names, 
                                 causal_map, feature_group_cache, eps_threshold,
                                 dqn_cluster1, dqn_cluster2, operation_emb, 
                                 STATE_METHOD, DEVICE):
    """
    Causal-hierarchical strategy with intelligent restrictions:
    - Unary: Only direct causes and other (exclude indirect)
    - Binary: Prioritize direct × indirect, direct × direct
    """
    
    if op in O1:  # Unary operations
        # RESTRICTION: No unary operations on indirect causes
        direct_indices = feature_cluster_dict.get(0, [])
        other_indices = feature_cluster_dict.get(2, [])
        
        # Combine direct + other (exclude indirect for unary)
        allowed_indices = direct_indices + other_indices
        
        if len(allowed_indices) == 0:
            info("No direct or other causes available for unary operation")
            return None
        
        # Prioritize direct causes but allow others
        if len(direct_indices) > 0:
            n_direct = min(len(direct_indices), int(0.8 * 20))  # Max 16 direct
            n_other = min(len(other_indices), 20 - n_direct)    # Fill remaining with others
            selected_indices = direct_indices[:n_direct] + other_indices[:n_other]
        else:
            selected_indices = allowed_indices[:20]
        
        info(f"Unary '{op}' - Selected {len(selected_indices)} features (excluding indirect causes)")
        
        return select_from_specific_indices(selected_indices, Dg, feature_names, 
                                          eps_threshold, dqn_cluster1, STATE_METHOD, DEVICE)
        
    elif op in O2:  # Binary operations
        # PRIORITIZE: Direct × Indirect, Direct × Direct combinations
        direct_indices = feature_cluster_dict.get(0, [])
        indirect_indices = feature_cluster_dict.get(1, [])
        other_indices = feature_cluster_dict.get(2, [])
        
        # Strategy for binary operations
        if len(direct_indices) > 0 and len(indirect_indices) > 0:
            # Primary: Direct × Indirect
            cluster1_indices = direct_indices[:15]
            cluster2_indices = indirect_indices[:10]
            combination_type = "direct_indirect"
        elif len(direct_indices) >= 2:
            # Secondary: Direct × Direct  
            mid = len(direct_indices) // 2
            cluster1_indices = direct_indices[:mid]
            cluster2_indices = direct_indices[mid:]
            combination_type = "direct_direct"
        else:
            # Fallback: Any available features
            all_indices = direct_indices + indirect_indices + other_indices
            if len(all_indices) < 2:
                return None
            mid = len(all_indices) // 2
            cluster1_indices = all_indices[:mid]
            cluster2_indices = all_indices[mid:]
            combination_type = "fallback"
        
        info(f"Binary '{op}' - {combination_type}: {len(cluster1_indices)} × {len(cluster2_indices)}")
        
        return select_binary_from_indices(cluster1_indices, cluster2_indices, Dg, feature_names,
                                        op, operation_emb, eps_threshold, dqn_cluster1, 
                                        dqn_cluster2, STATE_METHOD, DEVICE)

def traditional_topk_selection(op, feature_cluster_dict, Dg, feature_names,
                              eps_threshold, dqn_cluster1, dqn_cluster2, 
                              operation_emb, STATE_METHOD, DEVICE):
    """
    Traditional approach but with top-k selection for efficiency
    """
    
    try:
        target = Dg.iloc[:, -1]
        all_features = Dg.iloc[:, :-1]
        
        mi_scores = mutual_info_regression(all_features, target)
        top_k = min(30, len(mi_scores))
        top_indices = np.argsort(mi_scores)[-top_k:]
        
        info(f"Traditional top-k selection: {len(top_indices)} features selected")
        
        if op in O1:
            return select_from_specific_indices(top_indices, Dg, feature_names,
                                              eps_threshold, dqn_cluster1, STATE_METHOD, DEVICE)
        elif op in O2:
            mid = len(top_indices) // 2
            cluster1_indices = top_indices[:mid]
            cluster2_indices = top_indices[mid:]
            
            return select_binary_from_indices(cluster1_indices, cluster2_indices, Dg, feature_names,
                                            op, operation_emb, eps_threshold, dqn_cluster1,
                                            dqn_cluster2, STATE_METHOD, DEVICE)
    except Exception as e:
        info(f"Traditional top-k selection failed: {e}, using random selection")
        return random_exploration_selection(op, feature_cluster_dict, Dg, feature_names,
                                          eps_threshold, dqn_cluster1, dqn_cluster2,
                                          operation_emb, STATE_METHOD, DEVICE)

def random_exploration_selection(op, feature_cluster_dict, Dg, feature_names,
                               eps_threshold, dqn_cluster1, dqn_cluster2,
                               operation_emb, STATE_METHOD, DEVICE):
    """Random exploration for discovery of unexpected patterns"""
    
    all_indices = []
    for cluster_id, indices in feature_cluster_dict.items():
        all_indices.extend(indices)
    
    if len(all_indices) == 0:
        return None
    
    if op in O1:
        selected_indices = np.random.choice(all_indices, min(20, len(all_indices)), replace=False)
        return select_from_specific_indices(selected_indices, Dg, feature_names,
                                          eps_threshold, dqn_cluster1, STATE_METHOD, DEVICE)
    elif op in O2:
        np.random.shuffle(all_indices)
        mid = len(all_indices) // 2
        cluster1_indices = all_indices[:mid]
        cluster2_indices = all_indices[mid:]
        
        return select_binary_from_indices(cluster1_indices, cluster2_indices, Dg, feature_names,
                                        op, operation_emb, eps_threshold, dqn_cluster1,
                                        dqn_cluster2, STATE_METHOD, DEVICE)

def select_from_specific_indices(indices, Dg, feature_names, eps_threshold, dqn_cluster1, STATE_METHOD, DEVICE):
    """Helper function to select features from specific indices"""
    # Create a restricted cluster dict
    restricted_dict = {0: indices}
    return select_meta_cluster1(restricted_dict, Dg.values[:, :-1], feature_names, 
                               eps_threshold, dqn_cluster1, STATE_METHOD, DEVICE)

def select_binary_from_indices(cluster1_indices, cluster2_indices, Dg, feature_names, op,
                              operation_emb, eps_threshold, dqn_cluster1, dqn_cluster2, 
                              STATE_METHOD, DEVICE):
    """Helper function for binary feature selection"""
    
    # Select from first cluster
    restricted_dict1 = {0: cluster1_indices}
    action_emb_c1, state_emb_c1, f_cluster1, f_names1 = select_meta_cluster1(
        restricted_dict1, Dg.values[:, :-1], feature_names, eps_threshold, 
        dqn_cluster1, STATE_METHOD, DEVICE
    )
    
    # Select from second cluster
    restricted_dict2 = {0: cluster2_indices}
    op_emb = operation_emb[op]
    action_emb_c2, state_emb_c2, f_cluster2, f_names2 = select_meta_cluster2(
        restricted_dict2, Dg.values[:, :-1], feature_names, f_cluster1, op_emb,
        eps_threshold, dqn_cluster2, STATE_METHOD, DEVICE
    )
    
    return (action_emb_c1, state_emb_c1, f_cluster1, f_names1,
            action_emb_c2, state_emb_c2, f_cluster2, f_names2)

def calculate_hybrid_reward(new_per, old_per, feature_names, operation, strategy_used,
                           causal_map, feature_group_cache, action_history, num_new_features):
    """
    Enhanced reward function that gives bonuses based on strategy and causal adherence
    """
    
    # Base performance reward
    performance_reward = new_per - old_per
    
    # Get causal information
    causal_bonus, direct_ratio, indirect_ratio = get_causal_bonus_optimized(
        feature_names, causal_map, feature_group_cache
    )
    
    # Feature efficiency metrics
    feature_efficiency = performance_reward / max(1, num_new_features) if performance_reward > 0 else performance_reward
    
    # Penalize excessive feature generation
    if num_new_features > 100:
        feature_penalty = -0.01 * (num_new_features - 100) / 100
    else:
        feature_penalty = 0
    
    # Strategy-specific bonuses
    strategy_bonus = 0.0
    
    if strategy_used == 'causal_hierarchical':
        if operation in O1:
            if direct_ratio > 0.7:  # 70% direct causes
                strategy_bonus = 0.03
                info("Strategy bonus: Unary operation on direct causes")
        elif operation in O2:
            if direct_ratio > 0.3 and indirect_ratio > 0.3:
                strategy_bonus = 0.025
                info("Strategy bonus: Good direct-indirect binary combination")
            elif direct_ratio > 0.8:
                strategy_bonus = 0.02
                info("Strategy bonus: Direct-direct binary combination")
    
    elif strategy_used == 'traditional_topk':
        if performance_reward > 0:
            strategy_bonus = 0.01
            info("Strategy bonus: Traditional top-k improvement")
    
    # Calculate diversity and exploration bonuses
    if len(action_history) >= 5:
        recent_ops = action_history[-5:]
        diversity_ratio = len(set(recent_ops)) / len(recent_ops)
        diversity_bonus = 0.001 * diversity_ratio
    else:
        diversity_bonus = 0
    
    exploration_bonus = 0.001 * (direct_ratio + indirect_ratio)
    baseline_reward = 0.001
    
    # Combine rewards
    if performance_reward >= 0:
        total_reward = (performance_reward * (1 + causal_bonus + strategy_bonus) + 
                       exploration_bonus + diversity_bonus + feature_penalty)
    else:
        penalty_multiplier = 1 + (num_new_features / 100)
        penalty_reduction = strategy_bonus
        total_reward = (performance_reward * (penalty_multiplier * 0.1 - penalty_reduction) + 
                       exploration_bonus + baseline_reward + diversity_bonus + feature_penalty)
    
    # Anti-repetition penalty
    if operation == '/' and len(action_history) >= 3 and all(x == '/' for x in action_history[-3:]):
        total_reward -= 0.05
        info("Penalizing repeated division failures")
    
    return total_reward, {
        'performance_reward': performance_reward,
        'causal_bonus': causal_bonus,
        'strategy_bonus': strategy_bonus,
        'direct_ratio': direct_ratio,
        'indirect_ratio': indirect_ratio,
        'diversity_bonus': diversity_bonus,
        'exploration_bonus': exploration_bonus,
        'feature_efficiency': feature_efficiency,
        'feature_penalty': feature_penalty
    }

def intelligent_feature_pruning(Dg, feature_limit=800):
    """More aggressive and intelligent feature pruning"""
    
    if Dg.shape[1] <= feature_limit:
        return Dg
    
    info(f"Intelligent feature pruning: {Dg.shape[1]} -> {feature_limit}")
    
    target = Dg.iloc[:, -1]
    features = Dg.iloc[:, :-1]
    
    try:
        # Method 1: Mutual information
        mi_scores = mutual_info_regression(features, target)
        
        # Method 2: Variance threshold
        var_selector = VarianceThreshold(threshold=0.01)
        high_var_features = var_selector.fit_transform(features)
        high_var_mask = var_selector.get_support()
        
        # Combine criteria: high MI AND high variance
        combined_scores = mi_scores.copy()
        combined_scores[~high_var_mask] = 0
        
        # Select top features
        top_indices = np.argsort(combined_scores)[-feature_limit:]
        selected_features = features.iloc[:, top_indices]
        
        # Reconstruct dataframe
        Dg_pruned = pd.concat([selected_features, target], axis=1)
        
        info(f"Features after intelligent pruning: {Dg_pruned.shape[1]-1}")
        info(f"Removed {len(features.columns) - len(selected_features.columns)} features")
        
        return Dg_pruned
        
    except Exception as e:
        info(f"Intelligent pruning failed: {e}, using basic SelectKBest")
        selector = SelectKBest(mutual_info_regression, k=feature_limit)
        X_selected = selector.fit_transform(features, target)
        selected_mask = selector.get_support()
        selected_columns = features.columns[selected_mask]
        return pd.concat([pd.DataFrame(X_selected, columns=selected_columns), target], axis=1)

def load_causal_map(file_name, data_dir):
    """Load causal map from JSON file"""
    if file_name.endswith('.csv'):
        file_name = file_name[:-4]
    parts = file_name.split('_', 1)
    # target = parts[1] if len(parts) > 1 else file_name
    target = file_name
    
    map_dir = os.path.join(data_dir, 'causal_maps')
    map_path_underscore = os.path.join(map_dir, f'causal_map_{target}.json')
    map_path_dash = os.path.join(map_dir, f'causal_map_{target.replace("_", "-")}.json')
    
    if os.path.exists(map_path_underscore):
        with open(map_path_underscore, 'r') as f:
            return json.load(f)
    elif os.path.exists(map_path_dash):
        with open(map_path_dash, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Causal map not found: {map_path_underscore} or {map_path_dash}")

def get_causal_bonus_optimized(feature_names, causal_map, feature_group_cache):
    """Optimized causal bonus calculation with caching"""
    direct_count = indirect_count = 0
    total_count = len(feature_names)
    
    for f in feature_names:
        if f not in feature_group_cache:
            group = causal_map.get(f, {}).get('group', 'other')
            feature_group_cache[f] = group
        
        group = feature_group_cache[f]
        if group == 'direct':
            direct_count += 1
        elif group == 'indirect':
            indirect_count += 1
    
    direct_ratio = direct_count / total_count if total_count > 0 else 0
    indirect_ratio = indirect_count / total_count if total_count > 0 else 0
    causal_bonus = (direct_ratio * 0.8) + (indirect_ratio * 0.2)
    
    return causal_bonus, direct_ratio, indirect_ratio

def model_train(param, nni):
    """Main training function with all enhancements and bug fixes"""
    
    DEVICE = param['default_cuda']
    STATE_METHOD = param['state_method']
    assert STATE_METHOD in state_rep
    use_nni = nni is not None
    current_trial_name = nni.get_trial_id() if use_nni else 'local'
    D_OPT_PATH = os.path.join(BASE_DIR, 'tmp', param['file_name']+'_enhanced_causal')
    os.makedirs(D_OPT_PATH, exist_ok=True)
    info('opt path is {}'.format(D_OPT_PATH))

    acc_list, dis_list, per_list, rew_list = [], [], [], []
    always_best, all_results, all_feature = [], [], []

    # 1. Read and clean data
    data_path = base_path + param['file_name'] + '.hdf'
    info('read the data from {}'.format(data_path))
    Dg = pd.read_hdf(data_path)
    y = Dg.iloc[:, -1]
    X = Dg.values[:, :-1]
    X_scaled = X[:, ~(X == 0).all(axis=0)]
    valid_indices = ~y.isna()
    X_valid = X_scaled[valid_indices, :]
    y_valid = y[valid_indices].values

    # 2. Enhanced causal feature clustering
    causal_map = load_causal_map(param['file_name'], base_path)
    feature_group_cache, feature_cluster_cache = init_feature_cache(list(Dg.columns[:-1]), causal_map)
    
    feature_cluster_dict = cluster_features_causal_enhanced(
        X, list(Dg.columns[:-1]), causal_map, feature_cluster_cache,
        max_per_cluster=param.get('max_per_cluster', 25)
    )

    # 3. Train/test split
    train_indices, test_indices = train_test_split(
        np.arange(X_valid.shape[0]), test_size=0.2, random_state=42, shuffle=True
    )

    # 4. Prepare DataFrame for RL
    X_train, X_val = X_valid[train_indices], X_valid[test_indices]
    y_train, y_val = y_valid[train_indices], y_valid[test_indices]
    Dg = pd.concat([pd.DataFrame(X_valid), pd.DataFrame(y_valid)], axis=1)
    
    # 5. Initialize task parameters
    assert param['rl_method'] in support_rl_method
    task_name = task_dict[param['file_name']] if param['task'] == 'ng' else param['task']
    measure = task_measure[task_name]
    info(f'Enhanced causal task: {task_name} on dataset {param["file_name"]} using {param.get("strategy_mode", "hybrid")} strategy')
    
    # 6. Baseline performance
    zero_count = (Dg == 0).sum().sum()
    total_elements = Dg.size
    zero_ratio = zero_count / total_elements
    old_per = downstream_task_new(Dg, test_indices, train_indices, task_name, measure, 22)
    base_per = old_per
    per_list.append(old_per)
    info('Baseline performance: {}'.format(old_per))

    # 7. Feature initialization
    feature_names = list(Dg.columns)
    X = Dg.values[:, :-1]
    y = Dg.values[:, -1]
    Dg = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)
    Dg.columns = [str(i + len(operation_set) + 5) for i in range(len(feature_names))]
    feature_names = Dg.columns

    # 8. RL agent setup
    one_hot_op = pd.get_dummies(operation_set)
    operation_emb = defaultdict()
    for item in one_hot_op.columns:
        operation_emb[item] = one_hot_op[item].values

    EPISODES = param['episodes']
    STEPS = param['steps']
    STATE_DIM = X.shape[0]
    ACTION_DIM = X.shape[0]
    MEMORY_CAPACITY = max(param['memory'], 32)
    OP_DIM = len(operation_set)
    FEATURE_LIMIT = 800
    N_ACTIONS = len(operation_set)
    MAX_BINARY_COMBINATIONS = param.get('max_binary_combinations', 50)
    
    # Initialize RL agents
    if param['rl_method'] == 'dqn':
        dqn_cluster1 = DQN1(STATE_DIM=STATE_DIM, ACTION_DIM=ACTION_DIM, MEMORY_CAPACITY=MEMORY_CAPACITY)
        dqn_operation = DQN2(N_STATES=STATE_DIM, N_ACTIONS=N_ACTIONS, MEMORY_CAPACITY=MEMORY_CAPACITY)
        dqn_cluster2 = DQN1(STATE_DIM=STATE_DIM + OP_DIM, ACTION_DIM=ACTION_DIM, MEMORY_CAPACITY=MEMORY_CAPACITY)
    elif param['rl_method'] == 'ddqn':
        dqn_cluster1 = DDQN1(STATE_DIM=STATE_DIM, ACTION_DIM=ACTION_DIM, MEMORY_CAPACITY=MEMORY_CAPACITY)
        dqn_operation = DDQN2(N_STATES=STATE_DIM, N_ACTIONS=N_ACTIONS, MEMORY_CAPACITY=MEMORY_CAPACITY)
        dqn_cluster2 = DDQN1(STATE_DIM=STATE_DIM + OP_DIM, ACTION_DIM=ACTION_DIM, MEMORY_CAPACITY=MEMORY_CAPACITY)
    elif param['rl_method'] == 'dueling_dqn':
        dqn_cluster1 = DuelingDQN1(STATE_DIM=STATE_DIM, ACTION_DIM=ACTION_DIM, MEMORY_CAPACITY=MEMORY_CAPACITY)
        dqn_operation = DuelingDQN2(N_STATES=STATE_DIM, N_ACTIONS=N_ACTIONS, MEMORY_CAPACITY=MEMORY_CAPACITY)
        dqn_cluster2 = DuelingDQN1(STATE_DIM=STATE_DIM + OP_DIM, ACTION_DIM=ACTION_DIM, MEMORY_CAPACITY=MEMORY_CAPACITY)
    elif param['rl_method'] == 'dueling_ddqn':
        dqn_cluster1 = DuelingDDQN1(STATE_DIM=STATE_DIM, ACTION_DIM=ACTION_DIM, MEMORY_CAPACITY=MEMORY_CAPACITY)
        dqn_operation = DuelingDDQN2(N_STATES=STATE_DIM, N_ACTIONS=N_ACTIONS, MEMORY_CAPACITY=MEMORY_CAPACITY)
        dqn_cluster2 = DuelingDDQN1(STATE_DIM=STATE_DIM + OP_DIM, ACTION_DIM=ACTION_DIM, MEMORY_CAPACITY=MEMORY_CAPACITY)

    # 9. Training initialization
    best_per = old_per
    D_OPT = Dg.copy()
    steps_done = 0
    EPS_START = param['eps_start']
    EPS_END = param['eps_end']
    EPS_DECAY = max(param['eps_decay'], 200)
    
    action_history = []
    strategy_performance = ['baseline']  # FIX: Initialize with baseline strategy
    
    info(f'Enhanced training starting with strategy mode: {param.get("strategy_mode", "hybrid")}')
    training_start_time = time.time()
    episode = 0
    global_step = 0
    
    # 10. Main training loop
    while episode < EPISODES:
        step = 0
        Dg = D_OPT.copy()
        
        while step < STEPS:
            step_start_time = time.time()
            steps_done += 1
            
            # Enhanced epsilon with adaptive decay
            eps_threshold = adaptive_epsilon_decay(steps_done, per_list, EPS_START, EPS_END, EPS_DECAY)
            
            feature_names = list(Dg.columns[:-1])

            # Enhanced clustering with controlled size
            feature_cluster_dict = cluster_features_causal_enhanced(
                Dg.values[:, :-1], feature_names, causal_map, feature_cluster_cache,
                max_per_cluster=param.get('max_per_cluster', 25)
            )

            # FIX: Operation selection BEFORE feature selection
            # First, select operation using a temporary cluster
            temp_indices = []
            for cluster_id, indices in feature_cluster_dict.items():
                temp_indices.extend(indices[:5])  # Take first 5 from each cluster
            
            if len(temp_indices) == 0:
                info("No features available for operation selection, skipping step")
                step += 1
                global_step += 1
                continue
                
            temp_cluster = Dg.values[:, temp_indices[:min(10, len(temp_indices))]]
            
            try:
                state_emb_op, op, op_index = select_operation(temp_cluster, operation_set, dqn_operation, steps_done, STATE_METHOD, DEVICE)
                info(f'Selected operation: {op}')
            except Exception as e:
                info(f"Operation selection failed: {e}, skipping step")
                step += 1
                global_step += 1
                continue

            # Now get the actual feature selection for this operation
            try:
                selection_result, strategy_used = hybrid_causal_feature_selection(
                    op, feature_cluster_dict, Dg, feature_names, causal_map, feature_group_cache,
                    eps_threshold, dqn_cluster1, dqn_cluster2, operation_emb, 
                    STATE_METHOD, DEVICE, step, param.get('strategy_mode', 'hybrid')
                )
                
                if selection_result is None:
                    info(f"Feature selection failed for operation {op}, skipping step")
                    # FIX: Add failed strategy to tracking
                    strategy_performance.append('failed')
                    per_list.append(old_per)  # Keep same performance
                    step += 1
                    global_step += 1
                    continue
                    
            except Exception as e:
                info(f"Enhanced feature selection failed: {e}, skipping step")
                strategy_performance.append('error')
                per_list.append(old_per)
                step += 1
                global_step += 1
                continue

            info(f'Step {step}, Operation: {op}, Strategy: {strategy_used}')

            # Track action diversity
            action_history.append(op)
            if len(action_history) > 10:
                action_history = action_history[-10:]

            # Calculate diversity bonus
            if len(action_history) >= 5:
                recent_ops = action_history[-5:]
                diversity_ratio = len(set(recent_ops)) / len(recent_ops)
                diversity_bonus = 0.001 * diversity_ratio
            else:
                diversity_bonus = 0

            # Feature generation with enhanced binary limitation
            try:
                if op in O1:  # Unary operations
                    action_emb_c1, state_emb_c1, f_cluster1, f_names1 = selection_result
                    action_emb_c2 = state_emb_c2 = f_cluster2 = f_names2 = None
                    
                    op_sign = justify_operation_type(op)
                    f_new, f_new_name = [], []
                    
                    for i in range(f_cluster1.shape[1]):
                        try:
                            if op == 'sqrt' and np.sum(f_cluster1[:, i] < 0) == 0:
                                f_new.append(op_sign(f_cluster1[:, i]))
                                f_new_name.append(add_unary(op_map_r[op], f_names1[i]))
                            elif op == 'reciprocal' and np.sum(f_cluster1[:, i] == 0) == 0:
                                f_new.append(op_sign(f_cluster1[:, i]))
                                f_new_name.append(add_unary(op_map_r[op], f_names1[i]))
                            elif op == 'log' and np.sum(f_cluster1[:, i] <= 0) == 0:
                                f_new.append(op_sign(f_cluster1[:, i]))
                                f_new_name.append(add_unary(op_map_r[op], f_names1[i]))
                            elif op not in ['sqrt', 'reciprocal', 'log']:
                                f_new.append(op_sign(f_cluster1[:, i]))
                                f_new_name.append(add_unary(op_map_r[op], f_names1[i]))
                        except Exception as e:
                            info(f"Error in unary operation {op} on feature {i}: {e}")
                            continue
                    
                    if not f_new:
                        info(f"No valid features generated for unary operation {op}")
                        strategy_performance.append(f'{strategy_used}_failed')
                        per_list.append(old_per)
                        step += 1
                        global_step += 1
                        continue
                        
                    f_generate = np.array(f_new).T
                    final_name = f_new_name

                elif op in O2:  # Binary operations with intelligent limitation
                    if len(selection_result) == 8:  # Binary selection
                        (action_emb_c1, state_emb_c1, f_cluster1, f_names1,
                         action_emb_c2, state_emb_c2, f_cluster2, f_names2) = selection_result
                    else:
                        info("Invalid binary selection result")
                        strategy_performance.append(f'{strategy_used}_invalid')
                        per_list.append(old_per)
                        step += 1
                        global_step += 1
                        continue
                    
                    # Enhanced binary feature limitation
                    potential_combinations = f_cluster1.shape[1] * f_cluster2.shape[1]
                    
                    if potential_combinations > MAX_BINARY_COMBINATIONS:
                        info(f"Limiting binary combinations from {potential_combinations} to {MAX_BINARY_COMBINATIONS}")
                        
                        target = Dg.iloc[:, -1]
                        
                        # Select best features from cluster 1
                        n_samples_c1 = min(10, f_cluster1.shape[1])
                        if f_cluster1.shape[1] > n_samples_c1:
                            try:
                                mi_scores_c1 = mutual_info_regression(f_cluster1, target)
                                top_indices_c1 = np.argsort(mi_scores_c1)[-n_samples_c1:]
                                f_cluster1 = f_cluster1[:, top_indices_c1]
                                f_names1 = [f_names1[i] for i in top_indices_c1]
                            except:
                                f_cluster1 = f_cluster1[:, :n_samples_c1]
                                f_names1 = f_names1[:n_samples_c1]
                        
                        # Select best features from cluster 2
                        n_samples_c2 = min(5, f_cluster2.shape[1])
                        if f_cluster2.shape[1] > n_samples_c2:
                            try:
                                mi_scores_c2 = mutual_info_regression(f_cluster2, target)
                                top_indices_c2 = np.argsort(mi_scores_c2)[-n_samples_c2:]
                                f_cluster2 = f_cluster2[:, top_indices_c2]
                                f_names2 = [f_names2[i] for i in top_indices_c2]
                            except:
                                f_cluster2 = f_cluster2[:, :n_samples_c2]
                                f_names2 = f_names2[:n_samples_c2]
                    
                    info(f"Final selection: {f_cluster1.shape[1]} × {f_cluster2.shape[1]} = {f_cluster1.shape[1] * f_cluster2.shape[1]} combinations")
                    
                    # Enhanced division zero handling
                    if op == '/' and np.sum(f_cluster2 == 0) > 0:
                        non_zero_cols = np.where(np.sum(f_cluster2 == 0, axis=0) == 0)[0]
                        if len(non_zero_cols) == 0:
                            info("No non-zero columns found for division, skipping")
                            strategy_performance.append(f'{strategy_used}_div_zero')
                            per_list.append(old_per)
                            step += 1
                            global_step += 1
                            continue
                        else:
                            f_cluster2 = f_cluster2[:, non_zero_cols]
                            f_names2 = [f_names2[i] for i in non_zero_cols]
                    
                    try:
                        op_func = justify_operation_type(op)
                        f_generate, final_name = operate_two_features_new(f_cluster1, f_cluster2, op, op_func, f_names1, f_names2)
                    except Exception as e:
                        info(f"Error in binary operation {op}: {e}")
                        strategy_performance.append(f'{strategy_used}_op_error')
                        per_list.append(old_per)
                        step += 1
                        global_step += 1
                        continue

                # Post-processing and feature integration
                if np.max(f_generate) > 1000:
                    scaler = MinMaxScaler()
                    f_generate = scaler.fit_transform(f_generate)
                
                f_generate = pd.DataFrame(f_generate)
                f_generate.columns = final_name

                # Only add new features
                new_feature_names = [name for name in final_name if name not in Dg.columns]
                if not new_feature_names:
                    info("No new features generated")
                    strategy_performance.append(f'{strategy_used}_no_new')
                    per_list.append(old_per)
                    step += 1
                    global_step += 1
                    continue

                num_new_features = len(new_feature_names)
                Dg = insert_generated_feature_to_original_feas(Dg, f_generate[new_feature_names])

                # Intelligent feature pruning
                Dg = intelligent_feature_pruning(Dg, FEATURE_LIMIT)

                # Performance evaluation
                new_per = downstream_task_new(Dg, test_indices, train_indices, task_name, measure, 22)
                per_list.append(new_per)
                strategy_performance.append(strategy_used)  # FIX: Add successful strategy
                
                if use_nni:
                    nni.report_intermediate_result(new_per)

                # Enhanced reward calculation
                reward, reward_components = calculate_hybrid_reward(
                    new_per, old_per, list(Dg.columns[:-1]), op, strategy_used,
                    causal_map, feature_group_cache, action_history, num_new_features
                )

                # Early stopping mechanism
                if len(per_list) >= 3:
                    recent_performance = per_list[-3:]
                    if all(p < best_per * 0.99 for p in recent_performance):
                        info(f"Early stopping: performance declining for 3 consecutive steps")
                        wandb.log({'train/early_stop': True, 'train/reason': 'performance_decline'}, step=global_step)
                        break

                # Stop if too many features without improvement
                if len(list(Dg.columns[:-1])) > 500 and new_per <= best_per * 1.01:
                    info(f"Early stopping: too many features ({len(list(Dg.columns[:-1]))}) without significant improvement")
                    wandb.log({'train/early_stop': True, 'train/reason': 'feature_explosion'}, step=global_step)
                    break

                # Scale rewards
                r_c1 = reward * param['a'] / 10
                r_op = reward * param['b'] / 10  
                r_c2 = reward * param['c'] / 10
                
                # Update best performance
                if new_per > best_per:
                    always_best.append((Dg.columns, new_per, episode, step))
                    best_per = new_per
                    D_OPT = Dg.copy()
                    
                all_results.append((Dg.columns, new_per, episode, step))
                all_feature.append(Dg)
                old_per = new_per

                # RL agent updates
                dqn_cluster1.store_transition(state_emb_c1, action_emb_c1, r_c1, state_emb_c1, action_emb_c1)
                if op in O2 and action_emb_c2 is not None:
                    dqn_cluster2.store_transition(state_emb_c2, action_emb_c2, r_c2, state_emb_c2, action_emb_c2)
                dqn_operation.store_transition(state_emb_op, op_index, r_op, state_emb_op)

                # Learning updates
                if steps_done > MEMORY_CAPACITY:
                    loss_c1 = dqn_cluster1.learn()
                    loss_op = dqn_operation.learn()
                    if op in O2:
                        loss_c2 = dqn_cluster2.learn()

                # Enhanced logging
                wandb.log({
                    'train/strategy_used': strategy_used,
                    'train/performance_reward': reward_components['performance_reward'],
                    'train/causal_bonus': reward_components['causal_bonus'],
                    'train/strategy_bonus': reward_components['strategy_bonus'],
                    'train/direct_ratio': reward_components['direct_ratio'],
                    'train/indirect_ratio': reward_components['indirect_ratio'],
                    'train/diversity_bonus': reward_components['diversity_bonus'],
                    'train/feature_efficiency': reward_components['feature_efficiency'],
                    'train/feature_penalty': reward_components['feature_penalty'],
                    'train/num_new_features': num_new_features,
                    'train/total_features': len(list(Dg.columns[:-1])),
                    'train/total_reward': reward,
                    'train/epsilon': eps_threshold,
                    'train/episode': episode,
                    'train/step': step,
                    'train/best_performance': best_per,
                    'train/new_performance': new_per,
                    'train/base_performance': base_per,
                    'train/improvement_from_base': new_per - base_per,
                    'train/improvement_percent': ((new_per - base_per) / base_per * 100) if base_per > 0 else 0,
                    'train/best_improvement_percent': ((best_per - base_per) / base_per * 100) if base_per > 0 else 0,
                    'train/performance_vs_base_ratio': new_per / base_per if base_per > 0 else 1.0
                }, step=global_step)

                info(f'Performance: {new_per:.6f}, Best: {best_per:.6f},  Strategy: {strategy_used}, Features: {len(list(Dg.columns[:-1]))}')
                info(f'Step {step} completed in {time.time() - step_start_time:.1f}s')
                
                step_improvement = ((new_per - old_per) / old_per * 100) if old_per > 0 else 0
                base_improvement = ((new_per - base_per) / base_per * 100) if base_per > 0 else 0
                best_improvement = ((best_per - base_per) / base_per * 100) if base_per > 0 else 0

                info(f'=== Step {step} Results ===')
                info(f'Performance: {new_per:.6f} (step: {step_improvement:+.3f}%, total: {base_improvement:+.3f}%)')
                info(f'Best: {best_per:.6f} (best improvement: {best_improvement:+.3f}%)')
                info(f'Base: {base_per:.6f} | Strategy: {strategy_used} | Features: {len(list(Dg.columns[:-1]))}')
                info(f'Step completed in {time.time() - step_start_time:.1f}s')
                info(f'========================')
                
            except Exception as e:
                info(f"Feature generation failed: {e}")
                strategy_performance.append(f'{strategy_used}_gen_error')
                per_list.append(old_per)
            
            global_step += 1
            step += 1
            
        episode += 1
        info(f'Episode {episode} completed in {time.time() - training_start_time:.1f}s')
    
    if always_best:
        best_entry = max(always_best, key=lambda x: x[1])  # x[1] is new_per
        best_columns, best_per_value, best_episode, best_step = best_entry
        info(f'Best performance {best_per_value:.6f} achieved at episode {best_episode}, step {best_step}')
            
    info(f'Enhanced training completed! Total time: {time.time() - training_start_time:.1f}s')
    info(f'Final best performance: {best_per:.6f}')

    # FIX: Ensure lists are same length before creating DataFrame
    min_length = min(len(per_list), len(strategy_performance))
    per_list_trimmed = per_list[:min_length]
    strategy_performance_trimmed = strategy_performance[:min_length]
    
    info(f"Saving results: {len(per_list_trimmed)} performance entries, {len(strategy_performance_trimmed)} strategy entries")
    
    # Save enhanced results
    result_df = pd.DataFrame({
        'per': per_list_trimmed, 
        'strategy': strategy_performance_trimmed
    })
    result_df = result_df.sort_values(by='per')
    result_df.to_csv(os.path.join(D_OPT_PATH, f'{current_trial_name}_enhanced_results.csv'), index=False)
    D_OPT.to_csv(os.path.join(D_OPT_PATH, f'{current_trial_name}_{best_per}_enhanced_best.csv'))
    
    with open(os.path.join(D_OPT_PATH, f'{current_trial_name}_enhanced_features.pkl'), 'wb') as f:
        pickle.dump(all_feature, f)
    
    info('Enhanced results saved.')
    return best_per

if __name__ == '__main__':
    try:
        args = init_param()
        wandb.login(key="")
        tuner_params = nni.get_next_parameter()
        params = vars(merge_parameter(args, tuner_params))
        wandb.init(
            project="causal_aware_GRFG",
            group="phase1",
            job_type="causal_aware_knowledge_acquisition",
            config=params
        )
        
        tmp_root = os.path.join(BASE_DIR, 'tmp')
        os.makedirs(tmp_root, exist_ok=True)
        D_OPT_PATH = os.path.join(tmp_root, params['file_name'])
        os.makedirs(D_OPT_PATH, exist_ok=True)
        
        info("Starting enhanced causal-aware feature engineering")
        info(f"Configuration: {params}")
        
        final_performance = model_train(params, nni)
        info(f"Final performance achieved: {final_performance}")
        
    except Exception as exception:
        error(exception)
        raise

