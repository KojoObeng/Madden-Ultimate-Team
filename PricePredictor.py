import pandas as pd
import numpy as np
import math
import copy

import matplotlib.pyplot as plt

from sklearn.linear_model import ElasticNet, Lasso, LassoCV, Ridge, RidgeCV, LassoLarsCV, LinearRegression
from sklearn.model_selection import cross_validate, train_test_split, KFold, cross_val_score, GridSearchCV
import sklearn.metrics as metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# Not Currently Used
import patsy
import matplotlib
from matplotlib.pylab import rcParams
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import *
from sklearn import linear_model
from matplotlib.ticker import MaxNLocator

# Fix Print Settings
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 200)

# Create a copy without Price, NAs and with categorical Variables
MutData = pd.read_csv("MutData.csv")
MutData = MutData[MutData.PRICE != "Unknown"]
MutData.dropna(subset=['SPD', 'STR', 'AGI', 'ACC', 'AWA', 'CTH', 'JMP',
                  'STA', 'INJ', 'TRK', 'ELU', 'BTK', 'BCV', 'SFA', 'SPM', 'JKM', 'CAR', 'SRR', 'MRR', 'DRR', 'CIT',
                  'SPC', 'RLS', 'THP', 'SAC', 'MAC', 'DAC', 'RUN', 'TUP', 'BSK', 'PAC', 'RBK', 'RBP', 'RBF', 'PBK',
                  'PBP', 'PBF', 'LBK', 'IBL', 'TAK', 'POW', 'PMV', 'FMV', 'BSH', 'PUR', 'PRC', 'MCV', 'ZCV', 'PRS',
                  'KPW', 'KAC', 'RET'], inplace=True)
MutData = pd.get_dummies(data=MutData, columns=['POS', 'ARCHETYPE'], drop_first=False)
list_of_pos = list(MutData)
MutDataCopy = copy.deepcopy(MutData)

# Create Empty DataFrame
MutDataCopy = pd.DataFrame()

# Create Categorical Variables
# Add speed to almost every position
# Test and leave the most relevant combinations of other position-skills

# Speed-Position combo except those on this list i.e. QB, P, K....
list_of_pos = list(MutData)
speed_pos = [col for col in list_of_pos if ("POS_" in col) and col not in
             ["POS_QB", "POS_P", "POS_K", "POS_C", "POS_DT", "POS_FB", "POS_LE", "POS_RG", "POS_LG"]]

# SPD Interactions
# Adding Position-Speed combo most positions (not those with high p-values and squaring)
for col in speed_pos:
    MutDataCopy[col.split("_")[1] + "_" + "SPD"] = MutData[col].mul(MutData['SPD'])
    MutDataCopy[col.split("_")[1] + "_" + "SPD " + "SQUARED"] = MutData[col].mul(MutData['SPD'])**2

# QB Interactions
qb_pos = ["POS_QB"]
for col in qb_pos:
    MutDataCopy[col.split("_")[1] + "_" + "RUN"] = MutData[col].mul(MutData['RUN'])

# Lineman Interactions
lineman_pos = ["POS_RT", "POS_RG", "POS_LG", "POS_LT"]
# for col in lineman_pos:
#   MutDataCopy[col.split("_")[1] + "_" + "RBK"] = MutData[col].mul(MutData['RBK'])
#   MutDataCopy[col.split("_")[1] + "_" + "RBP"] = MutData[col].mul(MutData['RBP'])
#   MutDataCopy[col.split("_")[1] + "_" + "RBF"] = MutData[col].mul(MutData['RBF'])
#   MutDataCopy[col.split("_")[1] + "_" + "PBK"] = MutData[col].mul(MutData['PBK'])
#   MutDataCopy[col.split("_")[1] + "_" + "PBP"] = MutData[col].mul(MutData['PBP'])
#   MutDataCopy[col.split("_")[1] + "_" + "PBF"] = MutData[col].mul(MutData['PBF'])

MutDataCopy["RT_PBK"] = MutData["POS_RT"].mul(MutData['PBK'])
MutDataCopy["RG_PBP"] = MutData["POS_RG"].mul(MutData['PBP'])
MutDataCopy["C_RBF"] = MutData["POS_C"].mul(MutData['RBF'])
MutDataCopy["LG_RBK"] = MutData["POS_LG"].mul(MutData['RBK'])
MutDataCopy["LT_PBK"] = MutData["POS_LT"].mul(MutData['PBK'])
MutDataCopy["FB_IBL"] = MutData["POS_FB"].mul(MutData['IBL'])

# Linebacker Interactions
linebacker_pos = ["POS_ROLB", "POS_LOLB", "POS_MLB"]
# for col in linebacker_pos:
#   MutDataCopy[col.split("_")[1] + "_" + "STR"] = MutData[col].mul(MutData['STR'])
#   MutDataCopy[col.split("_")[1] + "_" + "TAK"] = MutData[col].mul(MutData['TAK'])
#   MutDataCopy[col.split("_")[1] + "_" + "POW"] = MutData[col].mul(MutData['POW'])
#   MutDataCopy[col.split("_")[1] + "_" + "PMV"] = MutData[col].mul(MutData['PMV'])
#   MutDataCopy[col.split("_")[1] + "_" + "FMV"] = MutData[col].mul(MutData['FMV'])
#   MutDataCopy[col.split("_")[1] + "_" + "BSH"] = MutData[col].mul(MutData['BSH'])
#   MutDataCopy[col.split("_")[1] + "_" + "PUR"] = MutData[col].mul(MutData['PUR'])
#   MutDataCopy[col.split("_")[1] + "_" + "PRC"] = MutData[col].mul(MutData['PRC'])
#   MutDataCopy[col.split("_")[1] + "_" + "MCV"] = MutData[col].mul(MutData['MCV'])
#   MutDataCopy[col.split("_")[1] + "_" + "ZCV"] = MutData[col].mul(MutData['ZCV'])

MutDataCopy["ROLB_BSH"] = MutData["POS_ROLB"].mul(MutData['BSH'])
MutDataCopy["LOLB_PMV"] = MutData["POS_LOLB"].mul(MutData['PMV'])
MutDataCopy["MLB_BSH"] = MutData["POS_MLB"].mul(MutData['BSH'])

# Pass Rusher Interactions
pass_rush_pos = ["POS_LE", "POS_DT", "POS_RE"]

# for col in pass_rush_pos:
#   MutDataCopy[col.split("_")[1] + "_" + "STR"] = MutData[col].mul(MutData['STR'])
#   MutDataCopy[col.split("_")[1] + "_" + "TAK"] = MutData[col].mul(MutData['TAK'])
#   MutDataCopy[col.split("_")[1] + "_" + "POW"] = MutData[col].mul(MutData['POW'])
#   MutDataCopy[col.split("_")[1] + "_" + "PMV"] = MutData[col].mul(MutData['PMV'])
#   MutDataCopy[col.split("_")[1] + "_" + "FMV"] = MutData[col].mul(MutData['FMV'])
#   MutDataCopy[col.split("_")[1] + "_" + "BSH"] = MutData[col].mul(MutData['BSH'])
#   MutDataCopy[col.split("_")[1] + "_" + "PUR"] = MutData[col].mul(MutData['PUR'])
#   MutDataCopy[col.split("_")[1] + "_" + "PRC"] = MutData[col].mul(MutData['PRC'])

MutDataCopy["LE_ZCV"] = MutData["POS_LE"].mul(MutData['PRC'])
MutDataCopy["DT_BSH"] = MutData["POS_DT"].mul(MutData['BSH'])

# DB Interactions
# db_pos = ["POS_CB", "POS_FS", "POS_SS"]

# for col in db_pos:
#   MutDataCopy[col.split("_")[1] + "_" + "POW"] = MutData[col].mul(MutData['POW'])
#   MutDataCopy[col.split("_")[1] + "_" + "BSH"] = MutData[col].mul(MutData['BSH'])
#   MutDataCopy[col.split("_")[1] + "_" + "PUR"] = MutData[col].mul(MutData['PUR'])
#   MutDataCopy[col.split("_")[1] + "_" + "PRC"] = MutData[col].mul(MutData['PRC'])
#   MutDataCopy[col.split("_")[1] + "_" + "MCV"] = MutData[col].mul(MutData['MCV'])
#   MutDataCopy[col.split("_")[1] + "_" + "ZCV"] = MutData[col].mul(MutData['ZCV'])
#   MutDataCopy[col.split("_")[1] + "_" + "PRS"] = MutData[col].mul(MutData['PRS'])

MutDataCopy["CB_MCV"] = MutData["POS_CB"].mul(MutData['MCV'])
MutDataCopy["FS_MCV"] = MutData["POS_FS"].mul(MutData['ZCV'])
MutDataCopy["SS_PRC"] = MutData["POS_SS"].mul(MutData['PRC'])

# SKill Positions
skill_pos = ["POS_HB", "POS_WR", "POS_TE"]
skill_pos = ["POS_TE"]
# for col in skill_pos:
#   MutDataCopy[col.split("_")[1] + "_" + "STR"] = MutData[col].mul(MutData['STR'])
#   MutDataCopy[col.split("_")[1] + "_" + "AGI"] = MutData[col].mul(MutData['AGI'])
#   MutDataCopy[col.split("_")[1] + "_" + "ACC"] = MutData[col].mul(MutData['ACC'])
#   MutDataCopy[col.split("_")[1] + "_" + "CTH"] = MutData[col].mul(MutData['CTH'])
#   MutDataCopy[col.split("_")[1] + "_" + "JMP"] = MutData[col].mul(MutData['JMP'])
#   MutDataCopy[col.split("_")[1] + "_" + "BTK"] = MutData[col].mul(MutData['BTK'])
#   MutDataCopy[col.split("_")[1] + "_" + "BCV"] = MutData[col].mul(MutData['BCV'])
#   MutDataCopy[col.split("_")[1] + "_" + "SFA"] = MutData[col].mul(MutData['SFA'])
#   MutDataCopy[col.split("_")[1] + "_" + "SPM"] = MutData[col].mul(MutData['SPM'])
#   MutDataCopy[col.split("_")[1] + "_" + "JKM"] = MutData[col].mul(MutData['JKM'])
#   MutDataCopy[col.split("_")[1] + "_" + "CAR"] = MutData[col].mul(MutData['CAR'])
#   MutDataCopy[col.split("_")[1] + "_" + "SRR"] = MutData[col].mul(MutData['SRR'])
#   MutDataCopy[col.split("_")[1] + "_" + "MRR"] = MutData[col].mul(MutData['MRR'])
#   MutDataCopy[col.split("_")[1] + "_" + "DRR"] = MutData[col].mul(MutData['DRR'])
#   MutDataCopy[col.split("_")[1] + "_" + "CIT"] = MutData[col].mul(MutData['CIT'])
#   MutDataCopy[col.split("_")[1] + "_" + "SPC"] = MutData[col].mul(MutData['SPC'])
#   MutDataCopy[col.split("_")[1] + "_" + "RLS"] = MutData[col].mul(MutData['RLS'])
#   MutDataCopy[col.split("_")[1] + "_" + "RBK"] = MutData[col].mul(MutData['RBK'])
#   MutDataCopy[col.split("_")[1] + "_" + "RBP"] = MutData[col].mul(MutData['RBP'])
#   MutDataCopy[col.split("_")[1] + "_" + "RBF"] = MutData[col].mul(MutData['RBF'])
#   MutDataCopy[col.split("_")[1] + "_" + "PBK"] = MutData[col].mul(MutData['PBK'])
#   MutDataCopy[col.split("_")[1] + "_" + "PBP"] = MutData[col].mul(MutData['PBP'])
#   MutDataCopy[col.split("_")[1] + "_" + "PBF"] = MutData[col].mul(MutData['PBF'])

MutDataCopy["HB_BTK"] = MutData["POS_HB"].mul(MutData['BTK'])
MutDataCopy["WR_JKM"] = MutData["POS_WR"].mul(MutData['JKM'])
MutDataCopy["WR_CIT"] = MutData["POS_WR"].mul(MutData['CIT'])
MutDataCopy["WR_RLS"] = MutData["POS_WR"].mul(MutData['RLS'])
MutDataCopy["TE_CTH"] = MutData["POS_TE"].mul(MutData['CTH'])
MutDataCopy["TE_DRR"] = MutData["POS_TE"].mul(MutData['DRR'])

# Kicker/Punter Interactions
kick_pos = ["POS_K", "POS_P"]

# for col in kick_pos:
#     MutDataCopy[col.split("_")[1] + "_" + "KPW"] = MutData[col].mul(MutData['KPW'])
#     MutDataCopy[col.split("_")[1] + "_" + "KAC"] = MutData[col].mul(MutData['KAC'])

MutDataCopy["K_KPW"] = MutData["POS_K"].mul(MutData['KPW'])
MutDataCopy["P_KAC"] = MutData["POS_P"].mul(MutData['KAC'])

# -------------------------------------------------------------------------------
# Setup regression

X = MutDataCopy
y = MutData["PRICE"].astype(float).transform(lambda x: math.log(x))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,)

# Which coefficients were picked in this type of regression?
def coefficient_picks(type_of_reg):
    coeff_df = pd.DataFrame(linear_regr.coef_, X.columns, columns=['Coefficient'])
    num_of_nonzero = coeff_df.loc[coeff_df["Coefficient"] != 0].shape[0]
    num_of_zero = coeff_df.loc[coeff_df["Coefficient"] == 0].shape[0]
    print(type_of_reg + " picked " + str(num_of_nonzero) + " variables and eliminated the other " +
          str(num_of_zero) + " variables")


def regression_results(y_true, y_pred, type_of_reg):
    # Regression metrics
    print("-----------------" + type_of_reg + "-----------------")
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    mean_squared_log_error = metrics.mean_squared_log_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))


# ------------------------Linear-------------------------------------------------------
linear_regr = LinearRegression()
linear_regr.fit(X_train, y_train)
y_pred = linear_regr.predict(X_test)

coeff_df = pd.DataFrame(linear_regr.coef_, X.columns, columns=['Coefficient'])

coefficient_picks("linear_regr")
regression_results(y_test, y_pred, "linear_regr")



df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).apply(np.exp)
print("Linear", df)


model = sm.OLS(y_test, X_test).fit()
print(model.summary())
# model = sm.OLS(y_train, X_train[:]).fit()
# MSEs = cross_val_score(linear_regr, X_train, y_train, scoring='neg_mean_squared_error', cv=5)

# ------------------------Ridge------------------------------------------------------
ridge = Ridge()
parameters = {'alpha': [1e-14, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
ridge_regr = Ridge(alpha=0.001, normalize=True, tol=1)
ridge_regr.fit(X_train, y_train)

y_pred = ridge_regr.predict(X_test)

coefficient_picks("ridge_regr")
regression_results(y_test, y_pred, "ridge_regr")

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).apply(np.exp)
print("Ridge ", df)

#ridge_regr = GridSearchCV(ridge, parameters, scoring = 'mean_squared_error', cv=5)
#print("Ridge best params = ", ridge_regr.best_params_)
#print("Ridge best score = ", ridge_regr.best_score_)

# ------------------------LASSO-------------------------------------------------------
lasso = Lasso()
parameters = {'alpha': [1e-14, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
lasso_regr = Lasso(alpha=0.1, tol=0.0001)
lasso_regr.fit(X_train, y_train)

y_pred = lasso_regr.predict(X_test)

coefficient_picks("lasso_regr")
regression_results(y_test, y_pred, "lasso_regr")

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).apply(np.exp)
print("Lasso ", df)

# -----------------------------------------------------------------------------


def calculate_vif_(X, thresh=5.0):
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["features"] = X.columns
    print(vif)


def clean_up_data():
    # Drop Index Col, NA Rows, and Unknown Price Rows
    MutDataCopy = MutDataCopy.drop(MutDataCopy.columns[0], axis=1)
    MutDataCopy = MutDataCopy.dropna(how='any', axis=0)
    MutDataCopy = MutDataCopy[MutDataCopy.PRICE != 'Unknown']

    # Dropped due to Backward elimination based on P-values
    MutDataCopy.drop(["STR", "JMP", "INJ", "BCV", "SRR", "TAK", "MRR", "DRR", "CIT", "SPD"], 1, inplace=True)
    # MutDataCopy["OVR_SquareRooted"] = MutDataCopy["OVR"].transform(lambda x: np.log(x))
    # MutDataCopy["OVR_Squared"] = MutDataCopy["OVR"].transform(lambda x: x**2)


def final_model():
    # ------------- Final Model------------------
    y_pred = linear_regr.predict(X)
    df = pd.DataFrame({'Actual': y, 'Predicted': y_pred}).apply(np.exp)
    MutDataFinal = MutData.join(df)
    MutDataFinal["Discrepancy"] = MutDataFinal["Predicted"] - MutDataFinal["Actual"]
    MutDataFinal.to_csv("MutData3.csv", index=False)


def plot(feature, target):

    main_figure = plt.figure(1, figsize=(10, 5))
    feature_plot = main_figure.add_subplot(111)
    feature_plot.scatter(MutDataCopy[feature], np.exp(y.astype(float)))
    #feature_plot.plot(MutDataCopy[feature], MutDataCopy[feature].apply(lambda x: x**2))
    x = np.linspace(0, 100)
    line = plt.plot(x, x ** 2, lw=2)
    # feature_plot.xaxis.set_major_locator(MaxNLocator(integer=True))
    # feature_plot.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()

    #my_plot.ylabel(target)
    #my_plot.yaxis.set_major(MaxNLocator(integer=True))
    #my_plot.axis([50, 99, 0, 100])
    #my_plot.show()

    fig_1 = plt.figure(1, figsize=(2, 2))






















