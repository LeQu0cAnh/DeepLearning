import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm
import matplotlib.pyplot as plt
import gc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, KFold
import random
import seaborn as sns
from scipy import stats

plt.style.use('default')

# Load data - medical_care dataset
data_path = "06032026/medical_care.csv"  # Update this path if needed
medical = pd.read_csv(data_path)
print(medical.shape)
pd.set_option("display.max_columns", 50)
print(medical.head())

# Encode UCURNINS variable
medical["UCURNINS"] = (medical.UCURNINS == "Yes").astype(int)

# Define the formula
formula = 'UCURNINS ~ UMARSTAT + USATMED + URELATE + REGION + FHOSP + FDENT + FEMER + FDOCT + ' + \
          'UIMMSTAT + UAGE + U_FTPT + U_WKSLY + U_USHRS + HOTHVAL + HRETVAL + HSSVAL + HWSVAL + UBRACE + ' + \
          'UEDUC3 + GENDER'

# Fit GLM model on full data
mod = sm.GLM.from_formula(formula=formula,
                          data=medical,
                          family=sm.families.Binomial())
res = mod.fit()
print(res.summary())

# Cross-validation with KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []

for train_index, test_index in kf.split(medical):
    X_train_cv, X_test_cv = medical.iloc[train_index], medical.iloc[test_index]
    y_train_cv, y_test_cv = medical['UCURNINS'].iloc[train_index], medical['UCURNINS'].iloc[test_index]
    
    # Fit model on training fold
    mod_cv = sm.GLM.from_formula(formula=formula,
                                 data=X_train_cv,
                                 family=sm.families.Binomial())
    res_cv = mod_cv.fit()
    
    # Predict on test fold
    preds_cv = res_cv.predict(X_test_cv)
    
    # Calculate AUC
    auc = roc_auc_score(y_test_cv, preds_cv)
    auc_scores.append(auc)

print(f"Cross-validation AUC scores: {auc_scores}")
print(f"Mean AUC: {np.mean(auc_scores):.4f}")
print(f"Std AUC: {np.std(auc_scores):.4f}")

# Multiple random splits for robustness check
scores = []

for k in range(100):
    X_train, X_test, y_train, y_test = train_test_split(medical,
                                                        medical.UCURNINS,
                                                        stratify=medical.UCURNINS,
                                                        test_size=0.3,
                                                        random_state=random.randint(0, 10000))
    mod = sm.GLM.from_formula(formula=formula,
                              data=X_train,
                              family=sm.families.Binomial())
    res = mod.fit()
    preds = res.predict(X_test)
    auc_score = roc_auc_score(y_test, preds)
    scores.append(auc_score)

print("Value counts for UCURNINS:")
print(medical.UCURNINS.value_counts())

# Diagnostics
df_scores = pd.DataFrame(data=scores, columns=['scores'])
print(df_scores.scores.describe())
print(f'Kurtosis \t {round(df_scores.scores.kurtosis(), 5)}')
print(f'Skewness \t {round(df_scores.scores.skew(), 5)}')

# Normality tests
alpha = 1e-3

k2, p = stats.normaltest(df_scores.scores)
if p < alpha:
    print('Kurtosis, Skewness test: The null hypothesis about normality can be rejected.')
else:
    print('Kurtosis, Skewness test: The null hypothesis about normality can not be rejected.')

ks = stats.kstest(df_scores.scores, 'norm')
if ks[1] < alpha:
    print('Kolmogorov Smirnov test: The null hypothesis about normality can be rejected.')
else:
    print('Kolmogorov Smirnov test: The null hypothesis about normality can not be rejected.')

ks_norm = stats.kstest((df_scores.scores - df_scores.scores.mean()) / df_scores.scores.std(), 'norm')
if ks_norm[1] < alpha:
    print('Kolmogorov Smirnov (normalized sample) test: The null hypothesis about normality can be rejected.')
else:
    print('Kolmogorov Smirnov (normalized sample) test: The null hypothesis about normality can not be rejected.')

# Plot distribution
ax = sns.displot(data=df_scores.scores, kde=True, label='empirical hist')
x0, x1 = ax.ax.get_xlim()
x_pdf = np.linspace(x0, x1, len(df_scores))
y_pdf = stats.norm.pdf(x_pdf, df_scores.scores.mean(), df_scores.scores.std())
ax.ax.plot(x_pdf, y_pdf, 'r', lw=2, label='normal pdf')
ax.ax.legend()
plt.show()

# Check overfitting tendency
for k in range(1, 10):
    X_train, X_test, y_train, y_test = train_test_split(medical,
                                                        medical.UCURNINS,
                                                        test_size=0.1 * k,
                                                        random_state=0)
    mod = sm.GLM.from_formula(formula=formula,
                              data=X_train,
                              family=sm.families.Binomial())
    res = mod.fit()
    predsTrain = res.predict(X_train)
    preds = res.predict(X_test)
    print("Train AUC:", round(roc_auc_score(y_train, predsTrain), 4), "Valid AUC:", round(roc_auc_score(y_test, preds), 4))

# 10-fold CV
kf_10 = KFold(n_splits=10, shuffle=True, random_state=random.randint(0, 10000))

for train, test in kf_10.split(medical.index.values):
    mod = sm.GLM.from_formula(formula=formula,
                              data=medical.iloc[train],
                              family=sm.families.Binomial())
    res = mod.fit()
    predsTrain = res.predict(medical.iloc[train])
    preds = res.predict(medical.iloc[test])
    print("Train AUC:", round(roc_auc_score(medical.iloc[train].UCURNINS, predsTrain), 4), "Valid AUC:",
          round(roc_auc_score(medical.iloc[test].UCURNINS, preds), 4))

# 5-fold CV
kf_5 = KFold(n_splits=5, shuffle=True, random_state=random.randint(0, 10000))

for train, test in kf_5.split(medical.index.values):
    mod = sm.GLM.from_formula(formula=formula,
                              data=medical.iloc[train],
                              family=sm.families.Binomial())    
    res = mod.fit()
    predsTrain = res.predict(medical.iloc[train])
    preds = res.predict(medical.iloc[test])
    print("Train AUC:", round(roc_auc_score(medical.iloc[train].UCURNINS, predsTrain), 4),
          "Valid AUC:", round(roc_auc_score(medical.iloc[test].UCURNINS, preds), 4))

# Run 10-fold CV 10 times
for z in range(10):
    trainRes = []
    valRes = []
    kf = KFold(n_splits=10, shuffle=True, random_state=random.randint(0, 10000))
    
    for train, test in kf.split(medical.index.values):
        mod = sm.GLM.from_formula(formula=formula,
                                  data=medical.iloc[train],
                                  family=sm.families.Binomial())          
        res = mod.fit()
        predsTrain = res.predict(medical.iloc[train])
        preds = res.predict(medical.iloc[test])
        train_auc = roc_auc_score(medical.iloc[train].UCURNINS, predsTrain)
        val_auc = roc_auc_score(medical.iloc[test].UCURNINS, preds)
        trainRes.append(train_auc)
        valRes.append(val_auc)
    
    print(f"Run {z+1}: Mean Train AUC: {np.mean(trainRes):.4f}, Mean Valid AUC: {np.mean(valRes):.4f}")

# Run 5-fold CV 10 times
for z in range(10):
    trainRes = []
    valRes = []
    kf = KFold(n_splits=5, shuffle=True, random_state=random.randint(0, 10000))
    
    for train, test in kf.split(medical.index.values):
        mod = sm.GLM.from_formula(formula=formula,
                                  data=medical.iloc[train],
                                  family=sm.families.Binomial()) 
        res = mod.fit()
        predsTrain = res.predict(medical.iloc[train])
        preds = res.predict(medical.iloc[test])
        train_auc = roc_auc_score(medical.iloc[train].UCURNINS, predsTrain)
        val_auc = roc_auc_score(medical.iloc[test].UCURNINS, preds)
        trainRes.append(train_auc)
        valRes.append(val_auc)
    
    print(f"Run {z+1} (5-fold): Mean Train AUC: {np.mean(trainRes):.4f}, Mean Valid AUC: {np.mean(valRes):.4f}")

# Additional CV with predictions collection
predList = []
indList = []
kf = KFold(n_splits=5, shuffle=True, random_state=random.randint(0, 10000))

for train, test in kf.split(medical.index.values):
    mod = sm.GLM.from_formula(formula=formula,
                              data=medical.iloc[train],
                              family=sm.families.Binomial()) 
    res = mod.fit()
    preds = res.predict(medical.iloc[test])
    
    predList.append(preds.tolist())
    indList.append(medical.iloc[test].index.tolist())

predsSorted = pd.Series(sum(predList, []), index=sum(indList, [])).sort_index()
overall_auc = roc_auc_score(medical.UCURNINS.sort_index(), predsSorted)
print(f"Overall AUC from collected predictions: {overall_auc:.4f}")

# Stratified KFold
from sklearn.model_selection import StratifiedKFold

predList = []
indList = []
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random.randint(0, 10000))

for train, test in kf.split(medical.index.values, medical.UCURNINS):
    mod = sm.GLM.from_formula(formula=formula,
                              data=medical.iloc[train],
                              family=sm.families.Binomial())
    res = mod.fit()
    preds = res.predict(medical.iloc[test])
    
    predList.append(preds.tolist())
    indList.append(medical.iloc[test].index.tolist())

print(f'Overall percent of uninsured persons is {round(medical.UCURNINS.sum() / medical.UCURNINS.count() * 100, 2)}%')