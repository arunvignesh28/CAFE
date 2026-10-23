import os
import sys
import psutil
from sklearn.cluster import KMeans
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import torch
from sklearn.manifold import TSNE
from logger import *
from Operation import add_unary, operation_set, O1,  O2, sep_token
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, PowerTransformer
os.environ['NUMEXPR_MAX_THREADS'] = '32'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
torch.set_num_threads(8)
info(torch.get_num_threads())
info(torch.__config__.parallel_info())
import warnings
import math
from nni.utils import merge_parameter
import nni
import matplotlib.pyplot as plt
torch.manual_seed(0)
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from DQN import DQN1, DQN2, DDQN1, DDQN2, DuelingDQN1, DuelingDDQN2, DuelingDQN2, DuelingDDQN1
from tools import *
from task_mapping import task_dict, task_type, base_path, task_measure, state_rep, support_rl_method
import argparse
import pickle
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from tools import estimate_causal_effect
from pgmpy.estimators import PC
import networkx as nx

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def init_param():
    parser = argparse.ArgumentParser(description='PyTorch Experiment')
    parser.add_argument('--file-name', type=str, default='spam_base', help='data name')
    parser.add_argument('--log-level', type=str, default='info', help='log level')
    # parser.add_argument('--task', type=str, default='ng', help='ng/cls/reg/det/rank')
    parser.add_argument('--task', type=str, default='reg', help='ng/cls/reg/det/rank')
    parser.add_argument('--episodes', type=int, default=10, help='episodes for training')
    parser.add_argument('--steps', type=int, default=5, help='steps for each episode')
    parser.add_argument('--memory', type=int, default=8, help='memory capacity')
    parser.add_argument('--eps_start', type=float, default=0.9, help='eps start')
    parser.add_argument('--eps_end', type=float, default=0.5, help='eps end')
    parser.add_argument('--eps_decay', type=int, default=20, help='eps decay')
    parser.add_argument('--a', type=float, default=1, help='a')
    parser.add_argument('--b', type=float, default=1, help='b')
    parser.add_argument('--c', type=float, default=1, help='c')
    parser.add_argument('--rl-method', type=str, default='dqn', help='used reinforcement methods')
    parser.add_argument('--state-method', type=str, default='gcn', help='state representation method')
    parser.add_argument('--default-cuda', type=int, default=1, help='the using cuda')
    args, _ = parser.parse_known_args()
    return args

def cluster_features(X, cluster_num=4):
    # X: (n_samples, n_features)
    X_feat = X.T  # Now (n_features, n_samples)
    pca = PCA(n_components=2, random_state=42)
    X_feat_pca = pca.fit_transform(X_feat)
    gmm = GaussianMixture(n_components=cluster_num, random_state=42)
    feature_clusters = gmm.fit_predict(X_feat_pca)  # (n_features,)
    return feature_clusters

def model_train(param, nni):
    DEVICE = param['default_cuda']
    STATE_METHOD = param['state_method']
    assert STATE_METHOD in state_rep
    use_nni = nni is not None
    current_trial_name = nni.get_trial_id() if use_nni else 'local'
    D_OPT_PATH = os.path.join(BASE_DIR, 'tmp', param['file_name'])
    os.makedirs(D_OPT_PATH, exist_ok=True)
    info('opt path is {}'.format(D_OPT_PATH))

    acc_list, dis_list, per_list, rew_list = [], [], [], []
    always_best, all, all_feature = [], [], []

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

    # 2. Cluster features using PCA+GMM (for RL grouping, not for splitting samples)
    clusters = cluster_features(X_valid, cluster_num=4)
    cluster_dict = {}
    for cluster_id in np.unique(clusters):
        cluster_dict[cluster_id] = np.where(clusters == cluster_id)[0]

    # 3. Train/test split by samples (not by feature clusters)
    train_indices, test_indices = train_test_split(
        np.arange(X_valid.shape[0]),
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    # 4. Prepare DataFrame for RL
    X_train, X_val = X_valid[train_indices], X_valid[test_indices]
    y_train, y_val = y_valid[train_indices], y_valid[test_indices]
    Dg = pd.concat([pd.DataFrame(X_valid), pd.DataFrame(y_valid)], axis=1)
    zero_list, per_list = [], []
    assert param['rl_method'] in support_rl_method
    task_name = task_dict[param['file_name']] if param['task'] == 'ng' else param['task']
    measure = task_measure[task_name]
    info(f'the task is performing {task_name} on dataset {param["file_name"]}')
    info(f'the chosen reinforcement learning method is {param["rl_method"]}')
    info(f'the related measurement is {measure}')
    zero_count = (Dg == 0).sum().sum()
    total_elements = Dg.size
    zero_ratio = zero_count / total_elements
    zero_list.append(zero_ratio)
    old_per = downstream_task_new(Dg, test_indices, train_indices, task_name, measure, 22)
    per_list.append(old_per)
    info('done the base test with performance of {}'.format(old_per))

    feature_names = list(Dg.columns)
    info('initialize the features...')
    X = Dg.values[:, :-1]
    y = Dg.values[:, -1]
    Dg = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)
    Dg.columns = [str(i + len(operation_set) + 5) for i in range(len(feature_names))]
    feature_names = Dg.columns

    # RL agent setup
    one_hot_op = pd.get_dummies(operation_set)
    operation_emb = defaultdict()
    for item in one_hot_op.columns:
        operation_emb[item] = one_hot_op[item].values
    EPISODES = param['episodes']
    STEPS = param['steps']
    STATE_DIM = X.shape[0]
    ACTION_DIM = X.shape[0]
    MEMORY_CAPACITY = param['memory']
    OP_DIM = len(operation_set)
    FEATURE_LIMIT = 1000
    N_ACTIONS = len(operation_set)
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
    else:
        raise ValueError("Unsupported RL method")

    # RL exploration loop
    best_per = old_per
    D_OPT = Dg.copy()
    steps_done = 0
    EPS_START = param['eps_start']
    EPS_END = param['eps_end']
    EPS_DECAY = param['eps_decay']
    CLUSTER_NUM = 4
    a, b, c = param['a'], param['b'], param['c']
    info('initialize the model hyperparameter configure')
    info(f'epsilon start with {EPS_START}, end with {EPS_END}, the decay is {EPS_DECAY}, the cluster num is {CLUSTER_NUM}, the a, b, and c is set to {a}, {b}, and {c}')
    info('the training start...')
    training_start_time = time.time()
    episode = 0
    while episode < EPISODES:
        step = 0
        Dg = D_OPT.copy()
        local_best = -999
        best_per_opt = []
        while step < STEPS:
            step_start_time = time.time()
            steps_done += 1
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)
            clusters = cluster_features(X.astype(np.float64), cluster_num=CLUSTER_NUM)
            feature_cluster_dict = {}
            for cluster_id in np.unique(clusters):
                feature_cluster_dict[cluster_id] = np.where(clusters == cluster_id)[0]
            action_emb_c1, state_emb_c1, f_cluster1, f_names1 = (
                select_meta_cluster1(feature_cluster_dict, Dg.values[:, :-1], feature_names, eps_threshold, dqn_cluster1, STATE_METHOD, DEVICE))
            state_emb_op, op, op_index = select_operation(f_cluster1, operation_set, dqn_operation, steps_done, STATE_METHOD, DEVICE)
            info(f'start operating in step {step}, current op is {op}')
            # Feature generation logic (unary/binary)
            if op in O1:
                op_sign = justify_operation_type(op)
                f_new, f_new_name = [], []
                for i in range(f_cluster1.shape[1]):
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
                if not f_new:
                    step += 1
                    continue
                f_generate = np.array(f_new).T
                final_name = f_new_name
            elif op in O2:
                op_emb = operation_emb[op]
                op_func = justify_operation_type(op)
                action_emb_c2, state_emb_c2, f_cluster2, f_names2 = (
                    select_meta_cluster2(feature_cluster_dict, Dg.values[:, :-1], feature_names, f_cluster1, op_emb, eps_threshold, dqn_cluster2, STATE_METHOD, DEVICE))
                if op == '/' and np.sum(f_cluster2 == 0) > 0:
                    step += 1
                    continue
                f_generate, final_name = operate_two_features_new(f_cluster1, f_cluster2, op, op_func, f_names1, f_names2)
                if f_generate is None or len(final_name) == 0 or f_generate.shape[1] == 0:
                    step += 1
                    continue
            else:
                step += 1
                continue
            # Post-processing and insertion
            if np.max(f_generate) > 1000:
                scaler = MinMaxScaler()
                f_generate = scaler.fit_transform(f_generate)
            f_generate = pd.DataFrame(f_generate)
            f_generate.columns = final_name
            public_name = np.intersect1d(np.array(Dg.columns), final_name)
            if len(public_name) > 0:
                reduns = np.setxor1d(final_name, public_name)
                if len(reduns) > 0:
                    f_generate = f_generate[reduns]
                    Dg = insert_generated_feature_to_original_feas(Dg, f_generate)
                else:
                    step += 1
                    continue
            else:
                Dg = insert_generated_feature_to_original_feas(Dg, f_generate)
            if Dg.shape[1] > FEATURE_LIMIT:
                selector = SelectKBest(mutual_info_regression, k=FEATURE_LIMIT).fit(Dg.iloc[:, :-1], Dg.iloc[:, -1])
                cols = selector.get_support()
                X_new = Dg.iloc[:, :-1].loc[:, cols]
                Dg = pd.concat([X_new, Dg.iloc[:, -1]], axis=1)
            feature_names = list(Dg.columns)
            zero_count = (Dg == 0).sum().sum()
            total_elements = Dg.size
            zero_ratio = zero_count / total_elements
            new_per = downstream_task_new(Dg, test_indices, train_indices, task_name, measure, 22)
            zero_list.append(zero_ratio)
            per_list.append(new_per)
            if use_nni:
                nni.report_intermediate_result(new_per)
            reward = new_per - old_per
            r_c1, r_op, r_c2 = param['a'] / 10 * reward / 3, param['b'] / 10 * reward / 3, param['c'] / 10 * reward / 3
            if new_per > best_per:
                always_best.append((Dg.columns, new_per, episode, step))
                best_per = new_per
                D_OPT = Dg.copy()
            all.append((Dg.columns, new_per, episode, step))
            all_feature.append(Dg)
            old_per = new_per
            # RL agent update
            # (store_transition and learn logic as in your original code)
            info(
                f'New performance is: {new_per:.6f}, Best performance is: {best_per:.6f} (episode {episode} step {step}) Base performance is: {per_list[0]:.6f}'
            )
            info(f'Episode {episode}, Step {step} ends!')
            info(f'Current spend time for step-{step} is: {time.time() - step_start_time:.1f}s')
            best_per_opt.append(best_per)
            step += 1
        episode += 1
        info(f'Current spend time for episode-{episode} is: {time.time() - training_start_time:.1f}s')
        if episode % 5 == 0:
            info(f'Best performance is: {max(best_per_opt):.6f}')
            info(f'Episode {episode} ends!')
    info(f'Total spend time: {time.time() - training_start_time:.1f}s')
    info('Exploration ends!')

    # Save results
    result_df = pd.DataFrame({'per': per_list, 'zero': zero_list})
    result_df = result_df.sort_values(by='per')
    result_df.columns = [str(col) for col in result_df.columns]
    result_df.to_csv(os.path.join(D_OPT_PATH, f'{current_trial_name}_zero_per_data_sorted.csv'), index=False)
    plt.figure(figsize=(10, 6))
    plt.plot(result_df['per'], result_df['zero'], marker='o', label='Zero Count vs Per (Sorted)')
    plt.title('Zero Count vs Per (Sorted)', fontsize=16)
    plt.xlabel('Per', fontsize=14)
    plt.ylabel('Zero Count', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.savefig(os.path.join(D_OPT_PATH, f'{current_trial_name}_zero_vs_per_sorted.png'))
    plt.close()
    D_OPT.to_csv(os.path.join(D_OPT_PATH, f'{current_trial_name}_{best_per}.csv'))
    with open(os.path.join(D_OPT_PATH, f'{current_trial_name}.generated.feature.pkl'), 'wb') as f:
        pickle.dump(all_feature, f)
    info('Results saved.')

if __name__ == '__main__':
    try:
        args = init_param()
        tuner_params = nni.get_next_parameter()
        params = vars(merge_parameter(args, tuner_params))
        tmp_root = os.path.join(BASE_DIR, 'tmp')
        os.makedirs(tmp_root, exist_ok=True)
        D_OPT_PATH = os.path.join(tmp_root, params['file_name'])
        os.makedirs(D_OPT_PATH, exist_ok=True)
        debug(tuner_params)
        info(params)
        model_train(params, nni)
    except Exception as exception:
        error(exception)
        raise