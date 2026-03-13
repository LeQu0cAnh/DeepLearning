import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm
import matplotlib.pyplot as plt
import gc
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import random
import seaborn as sns
from scipy import stats

plt.style.use('default')

# Load data - Wines dataset
data_path = "06032026/wines.csv"  # Update this path if needed
wines = pd.read_csv(data_path)
print(wines.shape)
pd.set_option("display.max_columns", 50)
print(wines.head())

# Preprocessing
# Encode type
wines['type'] = wines['type'].map({'red': 0, 'white': 1})

# Features for sklearn models
features = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 
            'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 
            'pH', 'sulphates', 'alcohol', 'type']
X = wines[features]
y = wines['quality']

# Define the formula for GLM
formula = 'quality ~ fixed_acidity + volatile_acidity + citric_acid + residual_sugar + chlorides + free_sulfur_dioxide + total_sulfur_dioxide + density + pH + sulphates + alcohol + type'

# Function to run CV for a model
def run_cv(model_name, model_func, X, y, n_splits=5, n_repeats=10, stratified=False):
    r2_scores = []
    mse_scores = []
    for repeat in range(n_repeats):
        if stratified:
            # Bin quality for stratification
            y_binned = pd.cut(y, bins=[2, 4, 6, 10], labels=[0, 1, 2])
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random.randint(0, 10000))
            kf_split = kf.split(X, y_binned)
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random.randint(0, 10000))
            kf_split = kf.split(X)
        
        fold_r2 = []
        fold_mse = []
        for train, test in kf_split:
            model = model_func()
            model.fit(X.iloc[train], y.iloc[train])
            preds = model.predict(X.iloc[test])
            r2 = r2_score(y.iloc[test], preds)
            mse = mean_squared_error(y.iloc[test], preds)
            fold_r2.append(r2)
            fold_mse.append(mse)
        r2_scores.append(np.mean(fold_r2))
        mse_scores.append(np.mean(fold_mse))
    
    print(f"{model_name} - Mean R2: {np.mean(r2_scores):.4f}, Std R2: {np.std(r2_scores):.4f}, Mean MSE: {np.mean(mse_scores):.4f}, Std MSE: {np.std(mse_scores):.4f}")
    return r2_scores, mse_scores

# GLM with statsmodels
print("=== GLM (Statsmodels) ===")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
r2_scores_glm = []
mse_scores_glm = []

for train_index, test_index in kf.split(wines):
    X_train_cv, X_test_cv = wines.iloc[train_index], wines.iloc[test_index]
    y_train_cv, y_test_cv = wines['quality'].iloc[train_index], wines['quality'].iloc[test_index]
    
    mod_cv = sm.GLM.from_formula(formula=formula,
                                 data=X_train_cv,
                                 family=sm.families.Gaussian())
    res_cv = mod_cv.fit()
    preds_cv = res_cv.predict(X_test_cv)
    r2 = r2_score(y_test_cv, preds_cv)
    mse = mean_squared_error(y_test_cv, preds_cv)
    r2_scores_glm.append(r2)
    mse_scores_glm.append(mse)

print(f"GLM 5-fold CV R2 scores: {r2_scores_glm}")
print(f"Mean R2: {np.mean(r2_scores_glm):.4f}, Std R2: {np.std(r2_scores_glm):.4f}")
print(f"Mean MSE: {np.mean(mse_scores_glm):.4f}, Std MSE: {np.std(mse_scores_glm):.4f}")

# Repeated 5-fold CV for GLM
print("\n=== GLM Repeated 5-fold CV (10 repeats) ===")
r2_repeated_glm = []
mse_repeated_glm = []
for z in range(10):
    kf = KFold(n_splits=5, shuffle=True, random_state=random.randint(0, 10000))
    fold_r2 = []
    fold_mse = []
    for train, test in kf.split(wines.index.values):
        mod = sm.GLM.from_formula(formula=formula,
                                  data=wines.iloc[train],
                                  family=sm.families.Gaussian()) 
        res = mod.fit()
        preds = res.predict(wines.iloc[test])
        r2 = r2_score(wines.iloc[test].quality, preds)
        mse = mean_squared_error(wines.iloc[test].quality, preds)
        fold_r2.append(r2)
        fold_mse.append(mse)
    r2_repeated_glm.append(np.mean(fold_r2))
    mse_repeated_glm.append(np.mean(fold_mse))
    print(f"Run {z+1}: Mean R2: {np.mean(fold_r2):.4f}, Mean MSE: {np.mean(fold_mse):.4f}")

print(f"Overall Mean R2: {np.mean(r2_repeated_glm):.4f}, Std R2: {np.std(r2_repeated_glm):.4f}")
print(f"Overall Mean MSE: {np.mean(mse_repeated_glm):.4f}, Std MSE: {np.std(mse_repeated_glm):.4f}")

# Stratified vs Non-Stratified
print("\n=== GLM Stratified vs Non-Stratified (10 repeats) ===")
y_binned = pd.cut(wines['quality'], bins=[2, 4, 6, 10], labels=[0, 1, 2])
r2_non_strat = []
r2_strat = []
mse_non_strat = []
mse_strat = []

for z in range(10):
    # Non-stratified
    kf = KFold(n_splits=5, shuffle=True, random_state=random.randint(0, 10000))
    fold_r2 = []
    fold_mse = []
    for train, test in kf.split(wines.index.values):
        mod = sm.GLM.from_formula(formula=formula,
                                  data=wines.iloc[train],
                                  family=sm.families.Gaussian()) 
        res = mod.fit()
        preds = res.predict(wines.iloc[test])
        r2 = r2_score(wines.iloc[test].quality, preds)
        mse = mean_squared_error(wines.iloc[test].quality, preds)
        fold_r2.append(r2)
        fold_mse.append(mse)
    r2_non_strat.append(np.mean(fold_r2))
    mse_non_strat.append(np.mean(fold_mse))
    
    # Stratified
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random.randint(0, 10000))
    fold_r2 = []
    fold_mse = []
    for train, test in kf.split(wines.index.values, y_binned):
        mod = sm.GLM.from_formula(formula=formula,
                                  data=wines.iloc[train],
                                  family=sm.families.Gaussian()) 
        res = mod.fit()
        preds = res.predict(wines.iloc[test])
        r2 = r2_score(wines.iloc[test].quality, preds)
        mse = mean_squared_error(wines.iloc[test].quality, preds)
        fold_r2.append(r2)
        fold_mse.append(mse)
    r2_strat.append(np.mean(fold_r2))
    mse_strat.append(np.mean(fold_mse))

print(f"Non-Stratified: Mean R2 {np.mean(r2_non_strat):.4f}, Std R2 {np.std(r2_non_strat):.4f}, Mean MSE {np.mean(mse_non_strat):.4f}")
print(f"Stratified: Mean R2 {np.mean(r2_strat):.4f}, Std R2 {np.std(r2_strat):.4f}, Mean MSE {np.mean(mse_strat):.4f}")

# Compare with Titanic: In Titanic (classification), stratification had minimal impact on AUC stability.
# Here, for regression, stratification slightly improves R2 stability (lower std).

# Sklearn LinearRegression
print("\n=== Sklearn LinearRegression ===")
r2_lr, mse_lr = run_cv("LinearRegression", lambda: LinearRegression(), X, y, n_repeats=10)

# RandomForestRegressor
print("\n=== RandomForestRegressor ===")
r2_rf, mse_rf = run_cv("RandomForestRegressor", lambda: RandomForestRegressor(random_state=42, n_estimators=100), X, y, n_repeats=10)

# Check overfitting
print("\n=== Check Overfitting (GLM) ===")
for k in range(1, 10):
    X_train, X_test, y_train, y_test = train_test_split(wines,
                                                        wines.quality,
                                                        test_size=0.1 * k,
                                                        random_state=0)
    mod = sm.GLM.from_formula(formula=formula,
                              data=X_train,
                              family=sm.families.Gaussian())
    res = mod.fit()
    predsTrain = res.predict(X_train)
    preds = res.predict(X_test)
    train_r2 = r2_score(y_train, predsTrain)
    val_r2 = r2_score(y_test, preds)
    train_mse = mean_squared_error(y_train, predsTrain)
    val_mse = mean_squared_error(y_test, preds)
    print(f"Train size {int(len(X_train))}: Train R2 {train_r2:.4f}, Valid R2 {val_r2:.4f}, Train MSE {train_mse:.4f}, Valid MSE {val_mse:.4f}")

print(f"\nMean quality: {wines.quality.mean():.2f}, Std quality: {wines.quality.std():.2f}")
