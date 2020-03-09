import pandas as pd
import numpy as np
import patsy
import math
import copy
import matplotlib
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.pylab import rcParams
from sklearn import linear_model
from sklearn.linear_model import ElasticNet, Lasso, LassoCV, Ridge, RidgeCV, LassoLarsCV, LinearRegression
from sklearn.model_selection import cross_validate, train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import *
import sklearn.metrics as metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

rcParams['figure.figsize'] = 12, 10
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 200)

MutData = pd.read_csv("MutData.csv")
MutDataCopy = copy.deepcopy(MutData)

not_used = ["PLAYER_NAME", "HEIGHT", "QUICKSELL", "QS_CURRENCY"]

# All names
names= ["OVR", "POS", "PROGRAM", "TEAM", "WEIGHT", "ARCHETYPE", "PRICE",
        "clutch", "penalty", "lb_style", "dl_swim", "dl_spin", "dl_bull", "big_hitter", "strips_ball",
        "ball_in_air", "high_motor", "covers_ball", "extra_yards", "agg_catch", "rac_catch",
        "poss_catch", "drops_open", "sideline_catch", "qb_style", "tight_spiral", "sense_pressure",
        "throw_away", "force_passes", "SPD", "STR", "AGI", "ACC", "AWA", "CTH", "JMP", "STA", "INJ",
        "TRK", "ELU", "BTK", "BCV", "SFA", "SPM", "JKM", "CAR", "SRR", "MRR", "DRR", "CIT", "SPC", "RLS",
        "THP", "SAC", "MAC", "DAC", "RUN", "TUP", "BSK", "PAC", "RBK", "RBP", "RBF", "PBK", "PBP", "PBF",
        "LBK", "IBL", "TAK", "POW", "PMV", "FMV", "BSH", "PUR", "PRC", "MCV", "ZCV", "PRS", "KPW", "KAC", "RET"]

# Dropped due to domain knowledge
MutDataCopy.drop(["PLAYER_NAME", "HEIGHT", "QUICKSELL", "clutch", "penalty", "lb_style",
              "dl_swim", "dl_spin", "dl_bull", "big_hitter", "strips_ball",  "ball_in_air", "high_motor",
              "covers_ball", "extra_yards", "agg_catch", "rac_catch","poss_catch", "drops_open", "sideline_catch",
              "qb_style", "tight_spiral", "sense_pressure", "throw_away", "force_passes"], 1, inplace=True)



# Create Categorical Variables
MutDataCopy = pd.get_dummies(data=MutDataCopy, columns=['POS', 'PROGRAM', 'TEAM', "ARCHETYPE", "QS_CURRENCY"], drop_first=True)
list_of_pos = list(MutDataCopy)


def create_interactions():
    # SPD Interactions
    speed_pos = [col for col in list_of_pos if "POS_" in col]
    for col in speed_pos:
        MutDataCopy[col.split("_")[1] + "_" + "SPD"] = MutDataCopy[col].mul(MutDataCopy['SPD'])
    # Tackles Interactions
    lineman_pos = ["POS_RT", "POS_LT"]
    for col in lineman_pos:
        MutDataCopy[col.split("_")[1] + "_" + "RBK"] = MutDataCopy[col].mul(MutDataCopy['RBK'])
        MutDataCopy[col.split("_")[1] + "_" + "RBP"] = MutDataCopy[col].mul(MutDataCopy['RBP'])
        MutDataCopy[col.split("_")[1] + "_" + "RBF"] = MutDataCopy[col].mul(MutDataCopy['RBF'])
        MutDataCopy[col.split("_")[1] + "_" + "PBK"] = MutDataCopy[col].mul(MutDataCopy['PBK'])
        MutDataCopy[col.split("_")[1] + "_" + "PBP"] = MutDataCopy[col].mul(MutDataCopy['PBP'])
        MutDataCopy[col.split("_")[1] + "_" + "PBF"] = MutDataCopy[col].mul(MutDataCopy['PBF'])

    # Pass Rushers Interactions
    pass_rush_pos = ["POS_ROLB", "POS_LOLB", "POS_LE", "POS_DT", "POS_RE"]
    for col in pass_rush_pos:
        MutDataCopy[col.split("_")[1] + "_" + "POW"] = MutDataCopy[col].mul(MutDataCopy['POW'])
        MutDataCopy[col.split("_")[1] + "_" + "PMV"] = MutDataCopy[col].mul(MutDataCopy['PMV'])
        MutDataCopy[col.split("_")[1] + "_" + "FMV"] = MutDataCopy[col].mul(MutDataCopy['FMV'])
        MutDataCopy[col.split("_")[1] + "_" + "BSH"] = MutDataCopy[col].mul(MutDataCopy['BSH'])

    # DB Interactions
    db_pos = ["POS_CB", "POS_FS", "POS_SS"]
    for col in db_pos:
        MutDataCopy[col.split("_")[1] + "_" + "MCV"] = MutDataCopy[col].mul(MutDataCopy['MCV'])
        MutDataCopy[col.split("_")[1] + "_" + "ZCV"] = MutDataCopy[col].mul(MutDataCopy['ZCV'])
        MutDataCopy[col.split("_")[1] + "_" + "PRS"] = MutDataCopy[col].mul(MutDataCopy['PRS'])

    # Kicker/Punter Interactions
    kick_pos = ["POS_K", "POS_P"]
    for col in kick_pos:
        MutDataCopy[col.split("_")[1] + "_" + "KPW"] = MutDataCopy[col].mul(MutDataCopy['KPW'])
        MutDataCopy[col.split("_")[1] + "_" + "KAC"] = MutDataCopy[col].mul(MutDataCopy['KAC'])

    # QB Interactions
    qb_pos = ["POS_QB"]
    for col in qb_pos:
        MutDataCopy[col.split("_")[1] + "_" + "RUN"] = MutDataCopy[col].mul(MutDataCopy['RUN'])
        MutDataCopy[col.split("_")[1] + "_" + "TUP"] = MutDataCopy[col].mul(MutDataCopy['TUP'])
        MutDataCopy[col.split("_")[1] + "_" + "THP"] = MutDataCopy[col].mul(MutDataCopy['THP'])
        MutDataCopy[col.split("_")[1] + "_" + "SAC"] = MutDataCopy[col].mul(MutDataCopy['SAC'])
        MutDataCopy[col.split("_")[1] + "_" + "MAC"] = MutDataCopy[col].mul(MutDataCopy['MAC'])
        MutDataCopy[col.split("_")[1] + "_" + "DAC"] = MutDataCopy[col].mul(MutDataCopy['DAC'])


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


def calculate_vif_(X, thresh=5.0):
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["features"] = X.columns
    print(vif)


def coefficient_picks(type_of_reg):
    coeff_df = pd.DataFrame(eval(type_of_reg).coef_, X.columns, columns=['Coefficient'])
    num_of_nonzero = coeff_df.loc[coeff_df["Coefficient"] != 0].shape[0]
    num_of_zero = coeff_df.loc[coeff_df["Coefficient"] == 0].shape[0]
    print(type_of_reg + " picked " + str(num_of_nonzero) + " variables and eliminated the other " +
          str(num_of_zero) + " variables")


create_interactions()
# calculate_vif_()


# Drop Index Col, NA Rows, and Unknown Price Rows
MutDataCopy = MutDataCopy.drop(MutDataCopy.columns[0], axis=1)
MutDataCopy = MutDataCopy.dropna(how='any', axis=0)
MutDataCopy = MutDataCopy[MutDataCopy.PRICE != 'Unknown']
# Dropped due to Backward elimination based on P-values
MutDataCopy.drop(["STR", "JMP", "INJ", "BCV", "SRR", "TAK", "MRR", "DRR", "CIT", "SPD"], 1, inplace=True)
# MutDataCopy["OVR_SquareRooted"] = MutDataCopy["OVR"].transform(lambda x: np.log(x))
# MutDataCopy["OVR_Squared"] = MutDataCopy["OVR"].transform(lambda x: x**2)

# Set up Regression
X = MutDataCopy.drop(['PRICE'], axis=1)
y = MutDataCopy["PRICE"].astype(float).transform(lambda x: math.log(x))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,)

# ------------------------Linear-------------------------------------------------------
linear_regr = LinearRegression()
linear_regr.fit(X_train, y_train)

y_pred = linear_regr.predict(X_test)
regression_results(y_test, y_pred, "linear_regr")
coefficient_picks("linear_regr")

#model = sm.OLS(y_test, X_test).fit()
#print(model.summary())
# model = sm.OLS(y_train, X_train[:]).fit()
# MSEs = cross_val_score(linear_regr, X_train, y_train, scoring='neg_mean_squared_error', cv=5)

# ------------------------Ridge------------------------------------------------------
ridge = Ridge()
parameters = {'alpha': [1e-14, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
ridge_regr = Ridge(alpha=0.001, normalize=True, tol=1)
ridge_regr.fit(X_train, y_train)

y_pred = ridge_regr.predict(X_test)
regression_results(y_test, y_pred, "ridge_regr")
coefficient_picks("ridge_regr")

#ridge_regr = GridSearchCV(ridge, parameters, scoring = 'mean_squared_error', cv=5)
#print("Ridge best params = ", ridge_regr.best_params_)
#print("Ridge best score = ", ridge_regr.best_score_)

# ------------------------LASSO-------------------------------------------------------
lasso = Lasso()
parameters = {'alpha': [1e-14, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
lasso_regr = Lasso(alpha=0.1, tol=0.0001)
lasso_regr.fit(X_train, y_train)

y_pred = lasso_regr.predict(X_test)
regression_results(y_test, y_pred, "lasso_regr")
coefficient_picks("lasso_regr")

# ------------- Final Model------------------
y_pred = linear_regr.predict(X)
df = pd.DataFrame({'Actual': y, 'Predicted': y_pred}).apply(np.exp)
MutDataFinal = MutData.join(df)
MutDataFinal["Discrepancy"] = MutDataFinal["Predicted"] - MutDataFinal["Actual"]
MutDataFinal.to_csv("MutData3.csv", index=False)


def plot(feature, target):
    # fig = plt.figure(figsize=(30, 25))  # create the top-level container
    # gs = gridspec.GridSpec(10, 12)
    # ax = plt.subplot(gs[0:2, 0:2])
    # ax.xaxis.set_ticks(np.arange(10000, 500000, 100000))

    main_figure = plt.figure(1, figsize=(10, 5))
    feature_plot = main_figure.add_subplot(111)
    feature_plot.scatter(MutDataCopy[feature], MutDataCopy[target].astype(float))
    feature_plot.plot(MutDataCopy[feature], MutDataCopy[feature].apply(lambda x: math.exp(x-82.5)))
    feature_plot.xaxis.set_major_locator(MaxNLocator(integer=True))
    feature_plot.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()

    #my_plot.ylabel(target)
    #my_plot.yaxis.set_major(MaxNLocator(integer=True))
    #my_plot.axis([50, 99, 0, 100])
    #my_plot.show()

    fig_1 = plt.figure(1, figsize=(2, 2))

    # scatter_plot("SPD", "PRICE")
    # Plot of Variable
    # Basic
    # plt.plot(X_test, y_pred, color='red', linewidth=2)
    # plt.scatter(X_test, y_test,  color='gray')
    # plt.show()
    #
    # imp_coef = pd.concat([coef.sort_values().head(10),
    #                      coef.sort_values().tail(10)])
    # matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
    # imp_coef.plot(kind = "barh")
    # plt.title("Coefficients in the Lasso Model")

plot("OVR", "PRICE")




# ------- Code I might use Later ---------------

#lasso_regr = GridSearchCV(lasso, parameters, scoring = 'neg_mean_squared_error', cv=5)
# print(lasso_regr.score(X_test, y_test))
# print("Lasso best params = ", lasso_regr.best_params_)
# print("Lasso best score = ", lasso_regr.best_score_)

#lasso_regr = LassoCV(cv=3).fit(X_train, y_train)

#print(coeff_df)
# nonzero_cols = list(coef[coef != 0].index)
# X_train = X_train[nonzero_cols]
# print(X_train)

#X = sm.add_constant(X.to_numpy())
#calculate_vif_(X)

# --------------------------------- VIF--------------
# variables = list(range(X.shape[1]))
    # dropped = True
    # print(X.iloc[:, :].shape[1])
    # while dropped:
    # dropped = False
    # vif = [variance_inflation_factor(X.iloc[:, :].values, col)
    #        for col in range(X.iloc[:, :].shape[1])]
    # print(vif)
    #     maxloc = vif.index(max(vif))
    #     if max(vif) > thresh:
    #         print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
    #               '\' at index: ' + str(maxloc))
    #         del variables[maxloc]
    #         dropped = True
    #
    # print('Remaining variables:')
    # print(X.columns[variables])
    # return X.iloc[:, variables]
    # Xc = add_constant(X)
    # vifs = [vif(Xc.values, i) for i in range(len(Xc.columns))]
    # pd.Series(data=vifs, index=Xc.columns).sort_values(ascending=False)
    # regr = ElasticNet(random_state=0, alpha=0.9).fit(X_train, y_train)  # Do not use fit_intercept = False if you have removed 1 column after