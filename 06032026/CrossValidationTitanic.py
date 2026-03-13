import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm
import matplotlib.pyplot as plt
import gc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import random
import seaborn as sns
from scipy import stats

plt.style.use('default')

# Load data - Titanic dataset
data_path = "06032026/titanic.csv"  # Update this path if needed
titanic = pd.read_csv(data_path)
print(titanic.shape)
pd.set_option("display.max_columns", 50)
print(titanic.head())

# Preprocessing
# Fill missing values
titanic['age'] = titanic['age'].fillna(titanic['age'].median())
titanic['fare'] = titanic['fare'].fillna(titanic['fare'].median())
titanic['embarked'] = titanic['embarked'].fillna(titanic['embarked'].mode()[0])

# Encode categorical variables (sex is already 0/1, embarked to 0/1/2)
titanic['embarked'] = titanic['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Features for sklearn models
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
X = titanic[features]
y = titanic['survived']

# Define the formula for GLM
formula = 'survived ~ pclass + sex + age + sibsp + parch + fare + embarked'

# Function to run CV for a model
def run_cv(model_name, model_func, X, y, n_splits=5, n_repeats=10, stratified=False):
    auc_scores = []
    for repeat in range(n_repeats):
        if stratified:
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random.randint(0, 10000))
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random.randint(0, 10000))
        
        fold_aucs = []
        for train, test in kf.split(X, y):
            model = model_func()
            model.fit(X.iloc[train], y.iloc[train])
            preds = model.predict_proba(X.iloc[test])[:, 1]
            auc = roc_auc_score(y.iloc[test], preds)
            fold_aucs.append(auc)
        auc_scores.append(np.mean(fold_aucs))
    
    print(f"{model_name} - Mean AUC: {np.mean(auc_scores):.4f}, Std: {np.std(auc_scores):.4f}")
    return auc_scores

# GLM with statsmodels
print("=== GLM (Statsmodels) ===")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
auc_scores_glm = []

for train_index, test_index in kf.split(titanic):
    X_train_cv, X_test_cv = titanic.iloc[train_index], titanic.iloc[test_index]
    y_train_cv, y_test_cv = titanic['survived'].iloc[train_index], titanic['survived'].iloc[test_index]
    
    mod_cv = sm.GLM.from_formula(formula=formula,
                                 data=X_train_cv,
                                 family=sm.families.Binomial())
    res_cv = mod_cv.fit()
    preds_cv = res_cv.predict(X_test_cv)
    auc = roc_auc_score(y_test_cv, preds_cv)
    auc_scores_glm.append(auc)

print(f"GLM 5-fold CV AUC scores: {auc_scores_glm}")
print(f"Mean AUC: {np.mean(auc_scores_glm):.4f}, Std: {np.std(auc_scores_glm):.4f}")

# Repeated 5-fold CV for GLM
print("\n=== GLM Repeated 5-fold CV (10 repeats) ===")
auc_repeated_glm = []
for z in range(10):
    kf = KFold(n_splits=5, shuffle=True, random_state=random.randint(0, 10000))
    fold_aucs = []
    for train, test in kf.split(titanic.index.values):
        mod = sm.GLM.from_formula(formula=formula,
                                  data=titanic.iloc[train],
                                  family=sm.families.Binomial()) 
        res = mod.fit()
        preds = res.predict(titanic.iloc[test])
        auc = roc_auc_score(titanic.iloc[test].survived, preds)
        fold_aucs.append(auc)
    auc_repeated_glm.append(np.mean(fold_aucs))
    print(f"Run {z+1}: Mean AUC: {np.mean(fold_aucs):.4f}")

print(f"Overall Mean AUC: {np.mean(auc_repeated_glm):.4f}, Std: {np.std(auc_repeated_glm):.4f}")

# Stratified vs Non-Stratified
print("\n=== GLM Stratified vs Non-Stratified (10 repeats) ===")
auc_non_strat = []
auc_strat = []

for z in range(10):
    # Non-stratified
    kf = KFold(n_splits=5, shuffle=True, random_state=random.randint(0, 10000))
    fold_aucs = []
    for train, test in kf.split(titanic.index.values):
        mod = sm.GLM.from_formula(formula=formula,
                                  data=titanic.iloc[train],
                                  family=sm.families.Binomial()) 
        res = mod.fit()
        preds = res.predict(titanic.iloc[test])
        auc = roc_auc_score(titanic.iloc[test].survived, preds)
        fold_aucs.append(auc)
    auc_non_strat.append(np.mean(fold_aucs))
    
    # Stratified
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random.randint(0, 10000))
    fold_aucs = []
    for train, test in kf.split(titanic.index.values, titanic.survived):
        mod = sm.GLM.from_formula(formula=formula,
                                  data=titanic.iloc[train],
                                  family=sm.families.Binomial()) 
        res = mod.fit()
        preds = res.predict(titanic.iloc[test])
        auc = roc_auc_score(titanic.iloc[test].survived, preds)
        fold_aucs.append(auc)
    auc_strat.append(np.mean(fold_aucs))

print(f"Non-Stratified: Mean AUC {np.mean(auc_non_strat):.4f}, Std {np.std(auc_non_strat):.4f}")
print(f"Stratified: Mean AUC {np.mean(auc_strat):.4f}, Std {np.std(auc_strat):.4f}")

# Sklearn LogisticRegression
print("\n=== Sklearn LogisticRegression ===")
auc_lr = run_cv("LogisticRegression", lambda: LogisticRegression(random_state=42, max_iter=1000), X, y, n_repeats=10)

# RandomForest
print("\n=== RandomForest ===")
auc_rf = run_cv("RandomForest", lambda: RandomForestClassifier(random_state=42, n_estimators=100), X, y, n_repeats=10)

# Check overfitting
print("\n=== Check Overfitting (GLM) ===")
for k in range(1, 10):
    X_train, X_test, y_train, y_test = train_test_split(titanic,
                                                        titanic.survived,
                                                        test_size=0.1 * k,
                                                        random_state=0)
    mod = sm.GLM.from_formula(formula=formula,
                              data=X_train,
                              family=sm.families.Binomial())
    res = mod.fit()
    predsTrain = res.predict(X_train)
    preds = res.predict(X_test)
    train_auc = roc_auc_score(y_train, predsTrain)
    val_auc = roc_auc_score(y_test, preds)
    print(f"Train size {int(len(X_train))}: Train AUC {train_auc:.4f}, Valid AUC {val_auc:.4f}")

print(f"\nOverall survival rate: {round(titanic.survived.sum() / titanic.survived.count() * 100, 2)}%")