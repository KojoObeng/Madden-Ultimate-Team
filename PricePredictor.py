import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing, linear_model, metrics
# from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_validate, train_test_split, KFold
from sklearn.feature_selection import SelectKBest, chi2

import statsmodels.api as sm
from statsmodels.formula.api import ols

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 200)


MutData = pd.read_csv("MutData.csv")

not_used = ["PLAYER_NAME", "HEIGHT", "QUICKSELL", "QS_CURRENCY"]
names= ["OVR", "POS", "PROGRAM", "TEAM", "WEIGHT", "ARCHETYPE", "PRICE",
         "clutch", "penalty", "lb_style", "dl_swim", "dl_spin", "dl_bull", "big_hitter", "strips_ball",
        "ball_in_air", "high_motor", "covers_ball", "extra_yards", "agg_catch", "rac_catch",
        "poss_catch", "drops_open", "sideline_catch", "qb_style", "tight_spiral", "sense_pressure",
        "throw_away", "force_passes", "SPD", "STR", "AGI", "ACC", "AWA", "CTH", "JMP", "STA", "INJ",
         "TRK", "ELU", "BTK", "BCV", "SFA", "SPM", "JKM", "CAR", "SRR", "MRR", "DRR", "CIT", "SPC", "RLS",
        "THP", "SAC", "MAC", "DAC", "RUN", "TUP", "BSK", "PAC", "RBK", "RBP", "RBF", "PBK", "PBP", "PBF",
        "LBK", "IBL", "TAK", "POW", "PMV", "FMV", "BSH", "PUR", "PRC", "MCV", "ZCV", "PRS", "KPW", "KAC", "RET"]


MutData.drop(["PLAYER_NAME", "HEIGHT", "QUICKSELL", "clutch", "penalty", "lb_style",
              "dl_swim", "dl_spin", "dl_bull", "big_hitter", "strips_ball",  "ball_in_air", "high_motor",
              "covers_ball", "extra_yards", "agg_catch", "rac_catch","poss_catch", "drops_open", "sideline_catch",
              "qb_style", "tight_spiral", "sense_pressure", "throw_away", "force_passes"], 1, inplace=True)

MutData["OVR_Squared"] = MutData["OVR"].transform(lambda x: x**2)
MutData["OVR_Cubed"] = MutData["OVR"].transform(lambda x: x**3)


MutData = pd.get_dummies(data=MutData, columns=['POS', 'PROGRAM', 'TEAM', "ARCHETYPE", "QS_CURRENCY"])
MutData = MutData.drop(MutData.columns[0], axis=1)
MutData = MutData.dropna(how='any', axis=0)
MutData = MutData[MutData.PRICE != 'Unknown']


X = MutData.drop(['PRICE'], axis=1)
y = MutData["PRICE"].astype(float).transform(lambda x: math.log(x))

print(X,y)

kf = KFold(n_splits=5, shuffle=False).split(range(25))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.10, random_state=40)
regr = linear_model.LinearRegression() # Do not use fit_intercept = False if you have removed 1 column after dummy encoding
regr.fit(X_train, y_train)
print(regr.score(X_test, y_test))

y_pred = regr.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).apply(np.exp)

print(df.head(50))

#poly_reg = PolynomialFeatures(degree=2)
#x_poly = poly_reg.fit_transform((features))

coeff_df = pd.DataFrame(regr.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)

# plt.scatter(X_test, y_test,  color='gray')
# plt.plot(X_test, y_pred, color='red', linewidth=2)
# plt.show()

#X = pd.get_dummies(data=X, drop_first=True)
# target = pd.DataFrame(MutData, columns=["PRICE"])
# for column in MutData.columns:
#     if MutData[column].dtype==object:
#         dummyCols=pd.get_dummies(inputData[column])
#         inputData=inputData.join(dummyCols)
#         del inputData[column]
# program_dummy = pd.get_dummies(MutData['PROGRAM']).iloc[:, 1:]
# dummy = pd.get_dummies(orders['mycol']).iloc[:,1:]
# pd.concat([data, dummy], axis=1)
# model = sm.OLS(Y.astype(float), X.astype(float)).fit()
# predictions = model.predict(X)
# model.summary()
# test_set_r2 = r2_score(y_test, pred)
#MutData.fillna(-99999, inplace=True)
#MutData.dropna(inplace=True)
#X = np.array(MutData.drop([]))

