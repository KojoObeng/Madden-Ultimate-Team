import pandas as pd
import numpy as np
import math

from sklearn import linear_model
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
import sklearn.metrics as metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor

import matplotlib.pyplot as plt
from matplotlib import artist


class model:
    def __init__(self):

        # Adjust Print Settings
        pd.set_option('display.max_columns', 100)
        pd.set_option('display.max_rows', 200)

        # Remove NA Price rows
        MutData = pd.read_csv("MutData.csv")
        self.MutData = MutData[MutData.PRICE != "Unknown"]
        self.all_skills = ['STR', 'AGI', 'ACC', 'AWA', 'CTH', 'JMP',
                          'STA', 'INJ', 'TRK', 'ELU', 'BTK', 'BCV', 'SFA', 'SPM', 'JKM', 'CAR', 'SRR', 'MRR', 'DRR', 'CIT',
                          'SPC', 'RLS', 'THP', 'SAC', 'MAC', 'DAC', 'RUN', 'TUP', 'BSK', 'PAC', 'RBK', 'RBP', 'RBF', 'PBK',
                          'PBP', 'PBF', 'LBK', 'IBL', 'TAK', 'POW', 'PMV', 'FMV', 'BSH', 'PUR', 'PRC', 'MCV', 'ZCV', 'PRS',
                          'KPW', 'KAC', 'RET'] #SPD
        self.MutData.dropna(subset=self.all_skills)

        # Removing not used
        not_used = ["Unnamed: 0", "PLAYER_NAME", "TEAM", "HEIGHT", "WEIGHT", "QUICKSELL", "QS_CURRENCY", "clutch",
                    "penalty", "lb_style",
                    "dl_swim", "dl_spin", "dl_bull", "big_hitter", "strips_ball", "ball_in_air", "high_motor",
                    "covers_ball",
                    "extra_yards", "agg_catch", "rac_catch", "poss_catch", "drops_open", "sideline_catch", "qb_style",
                    "tight_spiral", "sense_pressure", "throw_away", "force_passes"]
        for skill in not_used:
            self.MutData.drop(skill, axis=1)

        # Adding categorical features
        self.MutData = pd.get_dummies(data=self.MutData, columns=['POS', 'ARCHETYPE', 'PROGRAM'], drop_first=True)

        self.all_vars = list(self.MutData)
        # Adding quadratic features
        for skill in self.all_skills:
            self.MutData[skill + " SQUARED"] = self.MutData[skill] ** 2

        # Adding Position-Skill Interactions
        pos_skill_dict = [[["POS_QB"], ['RUN']],
                           [["POS_RT", "POS_RG", "POS_LG", "POS_LT"], ["RBK", "RBP", "RBF", "RBK", "PBP", "PBF"]],
                           [["POS_HB", "POS_WR", "POS_TE"], ["STR", "AGI", "ACC", "CTH", "JMP", "BTK", "BCV", "SFA", "SPM",
                                    "JKM", "CAR", "SRR", "MRR", "DRR", "CIT", "SPC", "RLS", "RBK", "RBP", "RBF", "PBK", "PBP",
                                    "PBF"]],
                           [["POS_LE", "POS_DT", "POS_RE"], ["STR", "TAK", "POW", "PMV", "FMV", "BSH", "PUR", "PRC"]],
                           [["POS_ROLB", "POS_LOLB", "POS_MLB"], ["STR", "TAK", "POW", "PMV", "FMV", "BSH", "PUR", "PRC", "MCV",
                                                                  "ZCV"]],
                           [["POS_CB", "POS_FS", "POS_SS"], ["POW", "BSH", "PUR", "PRC", "MCV", "ZCV", "PRS"]],
                           [["POS_K", "POS_P"], ["KAC", "KPW"]]]
        for combo in pos_skill_dict:
            for pos in combo[0]:
                for skill in combo[1]:
                    self.MutData[pos.split("_")[1] + "_" + skill] = self.MutData[pos].mul(self.MutData[skill])
        print(self.MutData)

    def speed_model(self):
        # Adding Speed Squared and Position Interactions to most positions (not those with high p-values and squaring)
        speed_pos = [col for col in self.all_vars if ("POS_" in col) and col not in []]
        # notin = ["POS_QB", "POS_P", "POS_K", "POS_C", "POS_DT", "POS_FB", "POS_LE", "POS_RG", "POS_LG"]]
        for col in speed_pos:
            self.MutData[col.split("_")[1] + "_" + "SPD"] = self.MutData[col].mul(self.MutData['SPD']**2)
            self.MutData[col.split("_")[1] + "_" + "SPD " + "SQUARED"] = self.MutData[col].mul(self.MutData['SPD']**2)

    def do_regression(self):
        def setup_regression():
            # Setup regression
            self.X = self.MutData.drop(columns=["PRICE", "PLAYER_NAME", "TEAM"], inplace=False)
            self.y = self.MutData["PRICE"].astype(float).transform(lambda x: math.log(x))
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=.2,)
            print(self.X)

        def linear_regression():
            self.linear_regr = LinearRegression().fit(self.X_train, self.y_train)

            y_pred = self.linear_regr.predict(self.X_test)
            linear_results = pd.DataFrame({'Actual': self.y_test, 'Predicted': y_pred}).apply(np.exp)

            self.coefficient_picks(self.linear_regr.coef_, "Linear")
            self.regression_results(self.y_test, y_pred, "linear_regr")

            print("linear", self.linear_regr.score(self.X_test, self.y_test))

        def ridge_regression():
            self.ridge_regr = linear_model.Ridge(alpha=.5).fit(self.X_train, self.y_train)

            y_pred = self.ridge_regr.predict(self.X_test)
            ridge_results = pd.DataFrame({'Actual': self.y_test, 'Predicted': y_pred}).apply(np.exp)

            self.coefficient_picks(self.ridge_regr.coef_, "Ridge")
            self.regression_results(self.y_test, y_pred, "ridge_regr")

            print("ridge", self.ridge_regr.score(self.X_test, self.y_test))

        def lasso_regression():
            self.lasso_regr = linear_model.Lasso(alpha=.5).fit(self.X_train, self.y_train)

            y_pred = self.lasso_regr.predict(self.X_test)
            lasso_results = pd.DataFrame({'Actual': self.y_test, 'Predicted': y_pred}).apply(np.exp)

            self.coefficient_picks(self.lasso_regr.coef_, "Lasso")
            self.regression_results(self.y_test, y_pred, "lasso_regr")

            print("lasso", self.lasso_regr.score(self.X_test, self.y_test))

        setup_regression()
        linear_regression()
        ridge_regression()
        lasso_regression()
    # -----------------------------------------------------------------------------

    # Which coefficients were picked in this type of regression?
    def coefficient_picks(self, coeff, type_of_reg):
        num_of_nonzero = np.sum(np.abs(coeff.astype(float) != 0.00))
        num_of_zero = coeff.size - num_of_nonzero
        print(type_of_reg + " picked " + str(num_of_nonzero) + " variables and eliminated the other " +
              str(num_of_zero) + " variables")
        print("It picked these columns ", self.X.columns[coeff.astype(float) != 0.00])

    def regression_results(self, y_true, y_pred, type_of_reg):
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

    def calculate_vif_(self, X, thresh=5.0):
        vif = pd.DataFrame()
        vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif["features"] = X.columns
        print(vif)

    def final_model(self):
        # ------------- Final Model------------------
        y_pred = self.linear_regr.predict(self.X)
        df = pd.DataFrame({'Actual': self.y, 'Predicted': y_pred}).apply(np.exp)
        MutDataFinal = self.MutData.join(df)
        MutDataFinal["Discrepancy"] = MutDataFinal["Predicted"] - MutDataFinal["Actual"]
        MutDataFinal.to_csv("MutDataPred.csv", index=False)
        print("Printed to MutDataPred.csv")

    def plot(self):
        n = 100000
        f = lambda x: x/10

        fig = plt.figure()
        ax = plt.axes()

        ax.scatter(self.X["SPD"], self.y)
        plt.plot(self.X["SPD"], f(self.X["SPD"]), color = "black")
        plt.show()



## Calls
Test = model()
#Test.speed_model()
#Test.do_regression()
#Test.plot()















