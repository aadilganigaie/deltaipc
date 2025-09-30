import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# LOAD DATA
# -----------------------------
df1 = pd.read_csv("low_mem_newfeatures.csv")
df2 = pd.read_csv("med_mem_newfeatures.csv")
df3 = pd.read_csv("high_mem_newfeatures.csv")

df1['interference_level'] = 'low'
df2['interference_level'] = 'medium'
df3['interference_level'] = 'high'

df = pd.concat([df1, df2, df3], ignore_index=True)

# -----------------------------
# CLEAN DATA & SELECT FEATURES
# -----------------------------
df = df.loc[:, ~df.columns.str.startswith('Unnamed')]

metadata_cols = {'application', 'ways', 'interval', 'IPC', 'interference_level'}

if 'app' in df.columns:
    if df['app'].equals(df['application']):
        print("Removing duplicate 'app' column")
        df = df.drop('app', axis=1)
    else:
        metadata_cols.add('app')

# Use ONLY the hardcoded features
hardcoded_features = ['HPKC_L3', 'HPKI_L3', 'Stalls_L3', 'Frontend', 'Backend', 'MPKI_L1', 'Stalls_total','Bad_Speculation', 
                    'MPKI_L2', 'Stalls_total_per_cycle', 'MPKC_L3', 'Memory_Bound', 'MPKI_L3','memBW', 'BW_others']

# Validate that all hardcoded features exist in the dataframe
missing_features = [f for f in hardcoded_features if f not in df.columns]
if missing_features:
    raise ValueError(f"Missing features in data: {missing_features}")

features = hardcoded_features

print("Using hardware counter features:", features)
print(f"Number of features: {len(features)}")

# -----------------------------
# DATA QUALITY CHECKS
# -----------------------------
print("\n=== DATA QUALITY CHECKS ===")
print(f"Total data points: {len(df)}")
print(f"Applications: {df['application'].nunique()}")
print(f"Ways range: {df['ways'].min()} to {df['ways'].max()}")
print(f"IPC range: {df['IPC'].min():.4f} to {df['IPC'].max():.4f}")
print(f"Missing values: {df.isnull().sum().sum()}")

q1, q3 = df['IPC'].quantile([0.25, 0.75])
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = df[(df['IPC'] < lower_bound) | (df['IPC'] > upper_bound)]
print(f"IPC outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")

# -----------------------------
# PREPARE TRAINING DATA FOR ΔIPC
# -----------------------------
data_pairs = []

for app in df['application'].unique():
    app_df = df[df['application'] == app].copy()
    intervals = app_df['interval'].unique()
    
    for interval in intervals:
        interval_df = app_df[app_df['interval'] == interval]
        if len(interval_df) < 2:
            continue
            
        available_ways = interval_df['ways'].unique()
        
        for baseline_way in available_ways:
            for target_way in available_ways:
                if baseline_way == target_way:
                    continue
                    
                base_row = interval_df[interval_df['ways'] == baseline_way]
                target_row = interval_df[interval_df['ways'] == target_way]
                
                if base_row.empty or target_row.empty:
                    continue
                
                base_row = base_row.iloc[0]
                target_row = target_row.iloc[0]
                
                delta_ipc = target_row['IPC'] - base_row['IPC']
                
                if abs(delta_ipc) > 2.0:
                    continue
                
                row_dict = {
                    'application': app,
                    'baseline_way': baseline_way,
                    'target_way': target_way,
                    'delta_ipc': delta_ipc,
                    'interval': interval,
                    'interference_level': base_row['interference_level'],
                    'baseline_ipc': base_row['IPC']
                }
                
                # Add hardware features from baseline (ONLY hardcoded features)
                for feat in features:
                    row_dict[feat] = base_row[feat]
                
                # Add relative features (target vs baseline features) for hardcoded features only
                for feat in features:
                    if base_row[feat] != 0:
                        row_dict[f'{feat}_rel'] = (target_row[feat] - base_row[feat]) / abs(base_row[feat])
                    else:
                        row_dict[f'{feat}_rel'] = 0
                
                data_pairs.append(row_dict)

pair_df = pd.DataFrame(data_pairs)
print(f"\nGenerated {len(pair_df)} training pairs.")

q1, q3 = pair_df['delta_ipc'].quantile([0.05, 0.95])
pair_df = pair_df[(pair_df['delta_ipc'] >= q1) & (pair_df['delta_ipc'] <= q3)]
print(f"After outlier removal: {len(pair_df)} pairs")

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
pair_df['baseline_way'] = pair_df['baseline_way'].astype(int)
pair_df['target_way'] = pair_df['target_way'].astype(int)

pair_df['way_diff'] = pair_df['target_way'] - pair_df['baseline_way']
pair_df['way_ratio'] = pair_df['target_way'] / pair_df['baseline_way']

le_app = LabelEncoder()
pair_df['application_encoded'] = le_app.fit_transform(pair_df['application'])

le_interf = LabelEncoder()
pair_df['interference_encoded'] = le_interf.fit_transform(pair_df['interference_level'])

pair_df['baseline_way_x_app'] = pair_df['baseline_way'] * pair_df['application_encoded']
pair_df['way_diff_x_interference'] = pair_df['way_diff'] * pair_df['interference_encoded']

relative_features = [f'{feat}_rel' for feat in features if f'{feat}_rel' in pair_df.columns]
model_features = (features + relative_features + 
                 ['baseline_way', 'target_way', 'way_diff', 'way_ratio',
                  'application_encoded', 'interference_encoded', 'baseline_ipc',
                  'baseline_way_x_app', 'way_diff_x_interference'])

model_features = [f for f in model_features if f in pair_df.columns]

print(f"Using {len(model_features)} features for modeling")

X = pair_df[model_features].copy()
y = pair_df['delta_ipc'].copy()

for col in X.columns:
    if not pd.api.types.is_numeric_dtype(X[col]):
        print(f"Converting {col} to numeric")
        X[col] = pd.to_numeric(X[col], errors='coerce')

X = X.fillna(0)
print("All features are numeric. Ready to train.")

# -----------------------------
# TRAIN-TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=pair_df['application']
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# EVALUATION METRICS
# -----------------------------
def calculate_adjusted_r2(r2, n_samples, n_features):
    """Calculate Adjusted R²"""
    if n_samples <= n_features + 1:
        return np.nan
    adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
    return adjusted_r2

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """Symmetric MAPE that handles zero values better"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, use_scaled=False):
    """Evaluate a model and return metrics"""
    X_train_use = X_train_scaled if use_scaled else X_train
    X_test_use = X_test_scaled if use_scaled else X_test
    
    y_train_pred = model.predict(X_train_use)
    y_test_pred = model.predict(X_test_use)
    
    n_features = X_train.shape[1]
    
    # Training metrics
    r2_train = r2_score(y_train, y_train_pred)
    adj_r2_train = calculate_adjusted_r2(r2_train, len(y_train), n_features)
    mse_train = mean_squared_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mse_train)
    mae_train = np.mean(np.abs(y_train - y_train_pred))
    smape_train = symmetric_mean_absolute_percentage_error(y_train, y_train_pred)
    
    # Test metrics
    r2_test = r2_score(y_test, y_test_pred)
    adj_r2_test = calculate_adjusted_r2(r2_test, len(y_test), n_features)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mse_test)
    mae_test = np.mean(np.abs(y_test - y_test_pred))
    smape_test = symmetric_mean_absolute_percentage_error(y_test, y_test_pred)
    
    return {
        'Model': model_name,
        'Train_R2': r2_train,
        'Train_Adj_R2': adj_r2_train,
        'Train_MSE': mse_train,
        'Train_RMSE': rmse_train,
        'Train_MAE': mae_train,
        'Train_SMAPE': smape_train,
        'Test_R2': r2_test,
        'Test_Adj_R2': adj_r2_test,
        'Test_MSE': mse_test,
        'Test_RMSE': rmse_test,
        'Test_MAE': mae_test,
        'Test_SMAPE': smape_test
    }

# -----------------------------
# TRAIN MODELS
# -----------------------------
print("\n" + "="*60)
print("TRAINING AND EVALUATING MODELS")
print("="*60)

results = []

# 1. XGBoost
print("\n[1/5] Training XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    objective='reg:squarederror',
    n_jobs=-1
)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
results.append(evaluate_model(xgb_model, X_train, X_test, y_train, y_test, 'XGBoost'))

# 2. Random Forest
print("[2/5] Training Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
results.append(evaluate_model(rf_model, X_train, X_test, y_train, y_test, 'Random Forest'))


'''
# 3. SVR
print("[3/5] Training SVR...")
svr_model = SVR(
    kernel='rbf',
    C=10.0,
    epsilon=0.1,
    gamma='scale'
)
svr_model.fit(X_train_scaled, y_train)
results.append(evaluate_model(svr_model, X_train, X_test, y_train, y_test, 'SVR', use_scaled=True))

'''

# 4. Polynomial Regression (degree 2)
print("[4/5] Training Polynomial Regression...")
poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('ridge', Ridge(alpha=1.0, random_state=42))
])
poly_model.fit(X_train_scaled, y_train)
results.append(evaluate_model(poly_model, X_train, X_test, y_train, y_test, 'Polynomial Regression', use_scaled=True))

# 5. Lasso
print("[5/5] Training Lasso...")
lasso_model = Lasso(
    alpha=0.001,
    max_iter=10000,
    random_state=42
)
lasso_model.fit(X_train_scaled, y_train)
results.append(evaluate_model(lasso_model, X_train, X_test, y_train, y_test, 'Lasso', use_scaled=True))

# -----------------------------
# FEATURE IMPORTANCE ANALYSIS
# -----------------------------
print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# XGBoost Feature Importance
print("\nXGBoost - Top 10 Most Important Features:")
xgb_importance = pd.DataFrame({
    'feature': model_features,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)
for i, row in xgb_importance.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Random Forest Feature Importance
print("\nRandom Forest - Top 10 Most Important Features:")
rf_importance = pd.DataFrame({
    'feature': model_features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
for i, row in rf_importance.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Lasso Coefficients (non-zero features)
lasso_coef = pd.DataFrame({
    'feature': model_features,
    'coefficient': np.abs(lasso_model.coef_)
}).sort_values('coefficient', ascending=False)
non_zero_features = lasso_coef[lasso_coef['coefficient'] > 0]
print(f"\nLasso - Selected Features (non-zero coefficients): {len(non_zero_features)}/{len(model_features)}")
print("Top 10 Features by Absolute Coefficient:")
for i, row in lasso_coef.head(10).iterrows():
    print(f"  {row['feature']}: {row['coefficient']:.4f}")

# -----------------------------
# DISPLAY RESULTS
# -----------------------------
results_df = pd.DataFrame(results)

print("\n" + "="*60)
print("MODEL COMPARISON - TRAINING SET")
print("="*60)
print(results_df[['Model', 'Train_R2', 'Train_Adj_R2', 'Train_MSE', 'Train_RMSE', 'Train_MAE', 'Train_SMAPE']].to_string(index=False))

print("\n" + "="*60)
print("MODEL COMPARISON - TEST SET")
print("="*60)
print(results_df[['Model', 'Test_R2', 'Test_Adj_R2', 'Test_MSE', 'Test_RMSE', 'Test_MAE', 'Test_SMAPE']].to_string(index=False))

# -----------------------------
# SAVE RESULTS
# -----------------------------
results_df.to_csv('model_comparison_results_hardcoded_features.csv', index=False)
print("\n✅ Results saved to 'model_comparison_results_hardcoded_features.csv'")

# Save feature importance
xgb_importance.to_csv('feature_importance_xgboost.csv', index=False)
rf_importance.to_csv('feature_importance_random_forest.csv', index=False)
lasso_coef.to_csv('feature_coefficients_lasso.csv', index=False)
print("✅ Feature importance saved to separate CSV files")

# Save detailed report
with open("model_comparison_report_hardcoded_features.txt", "w") as f:
    f.write("="*60 + "\n")
    f.write("MULTI-MODEL EVALUATION REPORT\n")
    f.write("Features Used: Stalls_L2, HPKI_L3, MPKC_L2, fractionPF, uselessPF, badSpeculation\n")
    f.write("="*60 + "\n\n")
    
    for idx, row in results_df.iterrows():
        f.write(f"\n{'='*60}\n")
        f.write(f"MODEL: {row['Model']}\n")
        f.write(f"{'='*60}\n")
        f.write(f"\nTRAINING METRICS:\n")
        f.write(f"  R²:           {row['Train_R2']:.6f}\n")
        f.write(f"  Adjusted R²:  {row['Train_Adj_R2']:.6f}\n")
        f.write(f"  MSE:          {row['Train_MSE']:.6f}\n")
        f.write(f"  RMSE:         {row['Train_RMSE']:.6f}\n")
        f.write(f"  MAE:          {row['Train_MAE']:.6f}\n")
        f.write(f"  SMAPE:        {row['Train_SMAPE']:.2f}%\n")
        
        f.write(f"\nTEST METRICS:\n")
        f.write(f"  R²:           {row['Test_R2']:.6f}\n")
        f.write(f"  Adjusted R²:  {row['Test_Adj_R2']:.6f}\n")
        f.write(f"  MSE:          {row['Test_MSE']:.6f}\n")
        f.write(f"  RMSE:         {row['Test_RMSE']:.6f}\n")
        f.write(f"  MAE:          {row['Test_MAE']:.6f}\n")
        f.write(f"  SMAPE:        {row['Test_SMAPE']:.2f}%\n")
        
        overfit = row['Train_R2'] - row['Test_R2']
        f.write(f"\nOVERFITTING INDICATOR:\n")
        f.write(f"  R² Gap (Train - Test): {overfit:.6f}\n")
    
    # Best model summary
    f.write(f"\n\n{'='*60}\n")
    f.write("BEST MODELS BY METRIC\n")
    f.write(f"{'='*60}\n")
    f.write(f"Best Test R²:        {results_df.loc[results_df['Test_R2'].idxmax(), 'Model']} ({results_df['Test_R2'].max():.6f})\n")
    f.write(f"Best Test Adj R²:    {results_df.loc[results_df['Test_Adj_R2'].idxmax(), 'Model']} ({results_df['Test_Adj_R2'].max():.6f})\n")
    f.write(f"Best Test MAE:       {results_df.loc[results_df['Test_MAE'].idxmin(), 'Model']} ({results_df['Test_MAE'].min():.6f})\n")
    f.write(f"Best Test SMAPE:     {results_df.loc[results_df['Test_SMAPE'].idxmin(), 'Model']} ({results_df['Test_SMAPE'].min():.2f}%)\n")
    
    # Feature importance section
    f.write(f"\n\n{'='*60}\n")
    f.write("FEATURE IMPORTANCE (XGBoost)\n")
    f.write(f"{'='*60}\n")
    for i, row in xgb_importance.head(15).iterrows():
        f.write(f"  {row['feature']}: {row['importance']:.4f}\n")
    
    f.write(f"\n\n{'='*60}\n")
    f.write("FEATURE IMPORTANCE (Random Forest)\n")
    f.write(f"{'='*60}\n")
    for i, row in rf_importance.head(15).iterrows():
        f.write(f"  {row['feature']}: {row['importance']:.4f}\n")
    
    f.write(f"\n\n{'='*60}\n")
    f.write("FEATURE SELECTION (Lasso)\n")
    f.write(f"{'='*60}\n")
    f.write(f"Non-zero features: {len(non_zero_features)}/{len(model_features)}\n")
    f.write(f"Top features by absolute coefficient:\n")
    for i, row in lasso_coef.head(15).iterrows():
        f.write(f"  {row['feature']}: {row['coefficient']:.4f}\n")

print("✅ Detailed report saved to 'model_comparison_report_hardcoded_features.txt'")

print("\n" + "="*60)
print("BEST PERFORMING MODELS (TEST SET)")
print("="*60)
print(f"Best R²:        {results_df.loc[results_df['Test_R2'].idxmax(), 'Model']} ({results_df['Test_R2'].max():.6f})")
print(f"Best Adj R²:    {results_df.loc[results_df['Test_Adj_R2'].idxmax(), 'Model']} ({results_df['Test_Adj_R2'].max():.6f})")
print(f"Best MAE:       {results_df.loc[results_df['Test_MAE'].idxmin(), 'Model']} ({results_df['Test_MAE'].min():.6f})")
print(f"Best SMAPE:     {results_df.loc[results_df['Test_SMAPE'].idxmin(), 'Model']} ({results_df['Test_SMAPE'].min():.2f}%)")

print("\n" + "="*60)
print("EXECUTION COMPLETED SUCCESSFULLY")
print("="*60)
