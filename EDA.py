# =============================================================================
# Exploratory Data Analysis (EDA) & Feature Engineering
# Dataset: Combined high/med/low memory workloads
# Target: IPC
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

# ----------------------------
# 1. Load and Combine Data
# ----------------------------
def load_and_combine_data():
    dfs = []
    for file in ["high_memory.csv", "med_memory.csv", "low_memory.csv"]:
        df_temp = pd.read_csv(file)
        if 'Unnamed: 0' in df_temp.columns:
            df_temp = df_temp.drop('Unnamed: 0', axis=1)
        dfs.append(df_temp)
    return pd.concat(dfs, ignore_index=True)

df_raw = load_and_combine_data()

# ----------------------------
# 2. Define Target and Features
# ----------------------------
TARGET = 'IPC'
FEATURES = [
    'Retiring', 'Frontend', 'Bad_Speculation', 'Backend', 'Memory_Bound',
    'Core_Bound', 'MPKI_L1', 'MPKI_L2', 'MPKC_L2', 'MPKI_L3', 'MPKC_L3',
    'HPKI_L3', 'HPKC_L3', 'Stalls_total', 'Stalls_total_per_cycle',
    'Stalls_other', 'Stalls_mem', 'Stalls_L2', 'Stalls_L3', 'LLCspace',
    'memBW', 'BW_others'
]

# Drop non-feature columns
cols_to_drop = ['interval', 'ways', 'instructions', 'cycles', 'app', 'application', 'time_ms']
df = df_raw.drop(columns=cols_to_drop, errors='ignore')

# Ensure all features and target are present
df = df[[TARGET] + FEATURES].copy()

# ----------------------------
# 3. Basic Statistics & Target Distribution
# ----------------------------
print("=== Basic Statistics ===")
print(df.describe().T)

# Plot target distribution
plt.figure(figsize=(8, 4))
sns.histplot(df[TARGET], kde=True, bins=50)
plt.title(f'Distribution of {TARGET}')
plt.tight_layout()
plt.savefig('ipc_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Q-Q plot for normality
plt.figure(figsize=(8, 4))
stats.probplot(df[TARGET], dist="norm", plot=plt)
plt.title(f'Q-Q Plot for {TARGET}')
plt.tight_layout()
plt.savefig('qq_plot_ipc.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------
# 4. Correlation Analysis
# ----------------------------
plt.figure(figsize=(16, 12))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', 
            annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
plt.title('Pearson Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Top correlated features with IPC
print("\n=== Top Correlated Features with IPC ===")
ipc_corr = corr[TARGET].abs().sort_values(ascending=False)
print(ipc_corr.head(10))

# ----------------------------
# 5. Outlier Detection (Boxplots)
# ----------------------------
plt.figure(figsize=(16, 8))
df[FEATURES + [TARGET]].boxplot(vert=False)
plt.title('Feature Distributions and Outliers')
plt.xlabel('Value')
plt.tight_layout()
plt.savefig('feature_boxplots.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------
# 6. Feature Engineering & Selection
# ----------------------------
df_processed = df.copy()

# 6.1 Transform target (Yeo-Johnson for normality)
pt = PowerTransformer(method='yeo-johnson')
df_processed['IPC_transformed'] = pt.fit_transform(df_processed[[TARGET]])

# 6.2 Remove highly collinear or redundant features
DROP_FEATURES = [
    'Retiring',           # Near-perfect correlation with IPC
    'Stalls_other', 'Stalls_mem', 'Stalls_L2', 'Stalls_L3',  # Redundant with Stalls_total
    'MPKC_L2', 'HPKI_L3', 'HPKC_L3',  # Redundant with MPKI variants
    'Core_Bound', 'LLCspace', 'memBW', 'BW_others'  # Low relevance or collinear
]
features_filtered = [f for f in FEATURES if f not in DROP_FEATURES]

# 6.3 Winsorize stall-related features (1% tails)
stalls_features = [f for f in features_filtered if 'Stalls' in f]
for feat in stalls_features:
    df_processed[feat] = winsorize(df_processed[feat], limits=[0.01, 0.01])

# 6.4 Feature Selection

X = df_processed[features_filtered]
y = df_processed['IPC_transformed']

# Mutual Information (non-linear relevance)
mi_selector = SelectKBest(mutual_info_regression, k=10)
mi_selector.fit(X, y)
selected_mi = X.columns[mi_selector.get_support()].tolist()
print("\nTop 10 Features (Mutual Information):", selected_mi)

# RFE with Random Forest
rfe_selector = RFE(
    estimator=RandomForestRegressor(n_estimators=100, random_state=42),
    n_features_to_select=10
)
rfe_selector.fit(X, y)
selected_rfe = X.columns[rfe_selector.get_support()].tolist()
print("Top 10 Features (RFE):", selected_rfe)

# 6.5 PCA on correlated group: Memory_Bound & Backend
memory_group = ['Memory_Bound', 'Backend']
pca = PCA(n_components=2)
memory_pca = pca.fit_transform(df_processed[memory_group])
df_processed[['Memory_PCA1', 'Memory_PCA2']] = memory_pca

# Plot PCA of memory features
plt.figure(figsize=(10, 8))
plt.scatter(memory_pca[:, 0], memory_pca[:, 1], alpha=0.5)
for i, feature in enumerate(memory_group):
    plt.arrow(0, 0,
              pca.components_[0, i] * 3,
              pca.components_[1, i] * 3,
              color='red', alpha=0.7, width=0.01)
    plt.text(pca.components_[0, i] * 3.2,
             pca.components_[1, i] * 3.2,
             feature, fontsize=12, color='red')
var = pca.explained_variance_ratio_
plt.text(-0.15, -0.15,
         f'Explained Variance:\nPC1: {var[0]:.1%}\nPC2: {var[1]:.1%}',
         fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat"))
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA of Memory-Related Features')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('memory_pca_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------
# 7. Final Feature Set
# ----------------------------
final_features = list(set(selected_mi + ['Memory_PCA1', 'Memory_PCA2']))
print("\n=== Final Feature Set ===")
print(final_features)
