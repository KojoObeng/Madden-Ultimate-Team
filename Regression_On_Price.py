import pandas as pd
import numpy as np
import math

from sklearn import linear_model
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
import sklearn.metrics as metrics
from regressors import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib import artist


class model:
    def __init__(self):

        # Adjust Print Settings
        pd.set_option('display.max_columns', 50)
        pd.set_option('display.max_rows', 10)

        # Remove NA Price rows
        MutData = pd.read_csv("Data/MutData.csv")
        self.MutData = MutData.loc[MutData.PRICE != "Unknown", ].copy()

        self.all_skills = ['SPD', 'STR', 'AGI', 'ACC', 'AWA', 'CTH', 'JMP', 'STA', 'INJ', 'TRK', 'ELU', 'BTK', 'BCV',
                           'SFA', 'SPM', 'JKM', 'CAR', 'SRR', 'MRR', 'DRR', 'CIT', 'SPC', 'RLS', 'THP', 'SAC', 'MAC', 'DAC',
                           'RUN', 'TUP', 'BSK', 'PAC', 'RBK', 'RBP', 'RBF', 'PBK', 'PBP', 'PBF', 'LBK', 'IBL', 'TAK', 'POW',
                           'PMV', 'FMV', 'BSH', 'PUR', 'PRC', 'MCV', 'ZCV', 'PRS', 'KPW', 'KAC', 'RET']

        #
        # 'PRC', 'PBK', 'BSK', 'KAC']
        self.skills_picked = ['OVR', 'AWA',
                              'BTK', 'PRC', 'PBK', 'BSK']  # 'KAC'

        self.quadratic_picked = ['AWA', 'BTK', 'PBK',
                                 'TAK', 'AGI', 'SPD']  # 'MCV', 'PRC' 'POW',

        # def add_features():
        #     for skill in self.skills_picked:
        #         self.MutData[skill]

        def add_categorical_features():
            self.MutData = pd.get_dummies(data=self.MutData, columns=[
                                          'POS'], drop_first=True)

        def add_quadratic_features():
            for skill in self.quadratic_picked:
                self.MutData[skill +
                             " SQUARED"] = self.MutData[skill].apply(lambda x: x**2)

        def add_pos_interactions():
            pos_skill_dict = [[["POS_QB"], ['RUN']],
                              [["POS_RT", "POS_RG", "POS_LG", "POS_LT"], [
                                  "RBK", "PBK", ]],  # "RBP", "RBF", "PBP", "PBF"
                              [["POS_HB", "POS_WR", "POS_TE"], ["STR", "AGI", "ACC", "CTH", "JMP", "BTK", "BCV", "SFA", "SPM",
                                                                "JKM", "CAR", "SRR", "MRR", "DRR", "CIT", "SPC", "RLS", "RBK", "PBK", "PBP"]],
                              [["POS_LE", "POS_DT", "POS_RE"], ["STR", "TAK",
                                                                "POW", "PMV", "FMV", "BSH", "PUR", "PRC"]],
                              [["POS_ROLB", "POS_LOLB", "POS_MLB"], ["STR", "TAK", "POW", "PMV", "FMV", "BSH", "PUR", "PRC", "MCV",
                                                                     "ZCV"]],
                              [["POS_CB", "POS_FS", "POS_SS"], [
                                  "POW", "BSH", "PUR", "PRC", "MCV", "ZCV", "PRS"]],
                              [["POS_K", "POS_P"], ["KAC", "KPW"]]]

            pos_skill_combos = [[["POS_RT"], ["PBK"]],  # RBK
                                # SPC "ACC", "AGI"
                                [["POS_HB"], ["PBK", "BTK"]],
                                [["POS_TE"], ["STR"]],  # CAR, RLS, "BCV
                                [["POS_WR"], ["ACC", "CIT"]],  # RLS, "AGI"
                                [["POS_LE"], ["FMV"]],
                                [["POS_DT"], ["BSH", "PRC", "TAK"]],  # FMV
                                [["POS_RE"], ["STR"]],  # FMV
                                [["POS_LOLB"], ["PUR"]],
                                [["POS_MLB"], ["POW", "PUR"]],
                                [["POS_CB"], ["PRC", "MCV"]],  # "BSH"
                                [["POS_FS"], ["ZCV"]],
                                [["POS_K"], ["KAC"]],
                                [["POS_P"], ["KAC"]]]  # "KPW"
            #  [["POS_SS"], ["MCV"]],
            #  [["POS_RG"], ["PBK"]],

            pos_skill_list = []
            for combo in pos_skill_combos:
                for pos in combo[0]:
                    for skill in combo[1]:
                        if skill not in pos_skill_list:
                            pos_skill_list.append(skill)
                        print(pos.split("_")[1] + "_" + skill)
                        self.MutData[pos.split(
                            "_")[1] + "_" + skill] = self.MutData[pos].mul(self.MutData[skill])

            # self.MutData.drop(columns=["POS"])

        def add_speed_interactions():
            # Adding Speed Squared and Position Interactions to most positions (not those with high p-values and squaring)
            speed_pos = [col for col in self.skills_picked if (
                "POS_" in col) and col not in []]
            # notin = ["POS_QB", "POS_P", "POS_K", "POS_C", "POS_DT", "POS_FB", "POS_LE", "POS_RG", "POS_LG"]]
            for col in speed_pos:
                self.MutData[col.split(
                    "_")[1] + "_" + "SPD"] = self.MutData[col].mul(self.MutData['SPD']**2)
                self.MutData[col.split("_")[
                    1] + "_" + "SPD " + "SQUARED"] = self.MutData[col].mul(self.MutData['SPD']**2)

        def drop_skills():
            self.MutData.drop(np.setdiff1d(
                self.all_skills, self.skills_picked), axis=1, inplace=True)

        def drop_useless():
            not_used = ["Unnamed: 0", "HEIGHT", "WEIGHT", "QUICKSELL", "ARCHETYPE", "QS_CURRENCY", "clutch",
                        "penalty", "lb_style",
                        "dl_swim", "dl_spin", "dl_bull", "big_hitter", "strips_ball", "ball_in_air", "high_motor",
                        "covers_ball",
                        "extra_yards", "agg_catch", "rac_catch", "poss_catch", "drops_open", "sideline_catch", "qb_style",
                        "tight_spiral", "sense_pressure", "throw_away", "force_passes"]

            self.MutData.drop(not_used, axis=1, inplace=True)
            self.MutData.dropna(inplace=True)

        add_categorical_features()
        add_quadratic_features()
        add_pos_interactions()
        # add_speed_interactions()
        drop_skills()
        drop_useless()

    def do_regression(self):

        self.regr_dict = {}

        def setup_regression():
            # Setup regression

            self.X = self.MutData.drop(
                columns=["PLAYER_NAME", "TEAM", "PRICE", "PROGRAM"], inplace=False)  # ,"POS", "PROGRAM"
            self.X = sm.add_constant(self.X)
            self.y = self.MutData["PRICE"].astype(
                float).transform(lambda x: math.log(x))
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=.2)

        def linear_regression():
            # Create regression
            self.linear_regr = LinearRegression().fit(self.X_train, self.y_train)

            # Results
            y_pred = self.linear_regr.predict(self.X_test)
            linear_results = pd.DataFrame(
                {'Actual': self.y_test, 'Predicted': y_pred}).apply(np.exp)

            # What coefficients and regression results
            self.regr_dict["Linear"] = self.linear_regr
            self.coefficient_picks(self.linear_regr.coef_, "Linear")
            self.regression_results(self.y_test, y_pred, "linear_regr")

            # print("Print this:", self.linear_regr.coef_)
            # print("linear", self.linear_regr.score(self.X_test, self.y_test))

            model = sm.OLS(self.y, self.X)
            model_fit = model.fit()
            # print(model_fit.summary())
            print(linear_results)

        def ridge_regression():
            # Create regression
            self.ridge_regr = linear_model.Ridge(
                alpha=2.5).fit(self.X_train, self.y_train)

            # Results
            y_pred = self.ridge_regr.predict(self.X_test)
            ridge_results = pd.DataFrame(
                {'Actual': self.y_test, 'Predicted': y_pred}).apply(np.exp)

            # What coefficients and regression results
            self.regr_dict["Ridge"] = self.ridge_regr
            self.coefficient_picks(self.ridge_regr.coef_, "Ridge")
            self.regression_results(self.y_test, y_pred, "ridge_regr")

        def lasso_regression():
            # Create regression
            self.lasso_regr = linear_model.Lasso(
                alpha=2.5).fit(self.X_train, self.y_train)
            # Results
            y_pred = self.lasso_regr.predict(self.X_test)
            lasso_results = pd.DataFrame(
                {'Actual': self.y_test, 'Predicted': y_pred}).apply(np.exp)

            # What coefficients and regression results
            self.regr_dict["Lasso"] = self.lasso_regr
            self.coefficient_picks(self.lasso_regr.coef_, "Lasso")
            self.regression_results(self.y_test, y_pred, "lasso_regr")

            # print("lasso", self.lasso_regr.score(self.X_test, self.y_test), stats.summary(self.lasso_regr, self.X_test, self.y_test))

        setup_regression()
        linear_regression()
        ridge_regression()
        lasso_regression()
    # -----------------------------------------------------------------------------

    # Which coefficients were picked in this type of regression?
    def coefficient_picks(self, coeff, type_of_reg):
        print("-----------------" + type_of_reg + "-----------------")
        num_of_nonzero = np.sum(np.abs(coeff.astype(float) != 0))
        num_of_zero = coeff.size - num_of_nonzero

        print(type_of_reg + " picked " + str(num_of_nonzero) + " variables and eliminated the other " +
              str(num_of_zero) + " variables")

        print("It picked these columns with these p-values \n", pd.concat([pd.DataFrame(self.X.columns[coeff.astype(float) != 0.00]),
                                                                           pd.DataFrame(stats.coef_pval(self.regr_dict[type_of_reg], self.X_test, self.y_test)[1:][coeff.astype(float) != 0])], axis=1))

    def regression_results(self, y_true, y_pred, type_of_reg):
        # Regression metrics
        mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
        mse = metrics.mean_squared_error(y_true, y_pred)
        mean_squared_log_error = metrics.mean_squared_log_error(y_true, y_pred)
        r2 = metrics.r2_score(y_true, y_pred)
        print('mean_squared_log_error: ', round(mean_squared_log_error, 4))
        print('r2: ', round(r2, 4))
        print('MAE: ', round(mean_absolute_error, 4))
        print('MSE: ', round(mse, 4))
        print('RMSE: ', round(np.sqrt(mse), 4))

    # ------------------------------------------------------------------------------------------

    def calculate_vif_(self, X, thresh=5.0):
        vif = pd.DataFrame()
        vif["VIF Factor"] = [variance_inflation_factor(
            X.values, i) for i in range(X.shape[1])]
        vif["features"] = X.columns
        print(vif)

    def final_model(self):
        # ------------- Final Model------------------
        y_pred = self.linear_regr.predict(self.X)
        df = pd.DataFrame(
            {'Actual': self.y, 'Predicted': y_pred}).apply(np.exp)
        MutDataFinal = self.MutData.join(df)
        MutDataFinal["Discrepancy"] = MutDataFinal["Predicted"] - \
            MutDataFinal["Actual"]
        MutDataFinal.to_csv("Data\MutDataPred.csv", index=False)
        print("Printed to MutDataPred.csv")

    def plot(self):
        n = 100000
        def f(x): return x/10

        fig = plt.figure()
        ax = plt.axes()

        ax.scatter(self.X["SPD"], self.y)
        plt.plot(self.X["SPD"], f(self.X["SPD"]), color="black")
        plt.show()


# Calls
Test = model()
Test.do_regression()
Test.final_model()
# Test.plot()
