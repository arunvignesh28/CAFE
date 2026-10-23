import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
# === Step 1: 数据加载与分离 ===
data = pd.read_csv('Dow/dow.csv').values
X_all = data[:, 7:]       # 全部特征
y_all = data[:, 1]        # 全部目标值（含 NaN）

# 标准化全部特征
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)

# === Step 2: 全部样本参与 PCA 和 GMM 聚类（无监督）===
pca = PCA(n_components=2, random_state=42)
X_pca_all = pca.fit_transform(X_all_scaled)

n_clusters = 4
gmm = GaussianMixture(n_components=n_clusters, random_state=42)
clusters_all = gmm.fit_predict(X_pca_all)

# === Step 3: 提取有标签样本 ===
mask = ~np.isnan(y_all)
X = X_all_scaled[mask]
y = y_all[mask]
clusters = clusters_all[mask]

# 划分训练 / 测试集
X_train, X_test, y_train, y_test, cluster_train, cluster_test = train_test_split(
    X, y, clusters, test_size=0.2, random_state=42
)

# === Step 4: 定义 Expert MLP ===
class ExpertMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)

# 初始化 experts 和优化器
input_dim = X.shape[1]
experts = [ExpertMLP(input_dim) for _ in range(n_clusters)]
optimizers = [optim.Adam(expert.parameters(), lr=0.001) for expert in experts]
criterion = nn.MSELoss()

# === Step 5: 训练每个 expert ===
for cluster_id in range(n_clusters):
    idx = (cluster_train == cluster_id)
    if idx.sum() == 0:
        continue
    x_cluster = torch.tensor(X_train[idx], dtype=torch.float32)
    y_cluster = torch.tensor(y_train[idx], dtype=torch.float32).unsqueeze(1)

    expert = experts[cluster_id]
    optimizer = optimizers[cluster_id]

    for epoch in range(100):
        expert.train()
        optimizer.zero_grad()
        output = expert(x_cluster)
        loss = criterion(output, y_cluster)
        loss.backward()
        optimizer.step()

    print(f"Expert {cluster_id} trained, final loss: {loss.item():.4f}")

# === Step 6: 推理函数（对新样本预测 cluster 然后用对应 expert）===
def predict(x_input):
    x_scaled = scaler.transform(x_input)
    x_pca = pca.transform(x_scaled)
    predicted_clusters = gmm.predict(x_pca)

    preds = []
    for i, x in enumerate(x_scaled):
        expert = experts[predicted_clusters[i]]
        expert.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            y_pred = expert(x_tensor).item()
        preds.append(y_pred)
    return np.array(preds)

def one_minus_rae(y_true, y_pred):
    y_mean = np.mean(y_true)
    rae = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - y_mean))
    return 1 - rae


# === Baseline 1: 单一 MLP（整体训练）===
mlp = MLPRegressor(hidden_layer_sizes=(64,), max_iter=300, random_state=42)
mlp.fit(X_train, y_train)
mlp_pred = mlp.predict(X_test)
print(f"[Baseline] MLP 1 - RAE: {one_minus_rae(y_test, mlp_pred):.4f}")

# === Baseline 2: 随机森林 ===
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print(f"[Baseline] Random Forest 1 - RAE: {one_minus_rae(y_test, rf_pred):.4f}")

# === Baseline 3: 非 cluster-aware MoE（每个 expert 训练全体数据，router 随机）===
# 初始化 experts
random_experts = [ExpertMLP(input_dim) for _ in range(n_clusters)]
random_opts = [optim.Adam(e.parameters(), lr=0.001) for e in random_experts]

# 训练每个 expert 用全部训练集
X_tensor = torch.tensor(X_train, dtype=torch.float32)
y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

for i in range(n_clusters):
    for epoch in range(100):
        random_experts[i].train()
        random_opts[i].zero_grad()
        output = random_experts[i](X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        random_opts[i].step()

# 推理时随机选一个 expert 预测
def predict_random_expert(x_input):
    x_scaled = scaler.transform(x_input)
    preds = []
    for x in x_scaled:
        expert = np.random.choice(random_experts)
        expert.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            y_pred = expert(x_tensor).item()
        preds.append(y_pred)
    return np.array(preds)

random_pred = predict_random_expert(X_test)
# print(f"[Baseline] Random Expert MoE 1 - RAE: {one_minus_rae(y_test, random_pred):.4f}")

# === 原 cluster-aware MoE 结果 ===
moe_pred = predict(X_test)
print(f"[MoE] Cluster-aware MoE 1 - RAE: {one_minus_rae(y_test, moe_pred):.4f}")

# === Cluster-aware Random Forest MoE ===
rf_experts = [RandomForestRegressor(n_estimators=100, random_state=cluster_id) for cluster_id in range(n_clusters)]

# 用有标签数据训练每个 expert
for cluster_id in range(n_clusters):
    idx = (cluster_train == cluster_id)
    if idx.sum() == 0:
        continue
    x_cluster = X_train[idx]
    y_cluster = y_train[idx]
    rf_experts[cluster_id].fit(x_cluster, y_cluster)
    print(f"Expert {cluster_id} trained on {idx.sum()} samples.")

# 推理函数（根据 GMM 聚类分发给对应 RF expert）
def predict_rf_expert(x_input):
    x_scaled = scaler.transform(x_input)
    x_pca = pca.transform(x_scaled)
    predicted_clusters = gmm.predict(x_pca)

    preds = []
    for i, x in enumerate(x_scaled):
        expert = rf_experts[predicted_clusters[i]]
        y_pred = expert.predict([x])[0]
        preds.append(y_pred)
    return np.array(preds)

# 评估 RF-MoE
rf_moe_pred = predict_rf_expert(X_test)
print(f"[MoE] Cluster-aware Random Forest MoE 1 - RAE: {one_minus_rae(y_test, rf_moe_pred):.4f}")