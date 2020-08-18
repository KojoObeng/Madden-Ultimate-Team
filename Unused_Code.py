# Just for reference
not_used = ["PLAYER_NAME", "HEIGHT", "QUICKSELL", "QS_CURRENCY"]


# MutData = pd.get_dummies(data=MutData, columns=['POS'], drop_first=True)

# Variables
MutDataCopy.drop(['Unnamed: 0', 'OVR', 'PLAYER_NAME', 'PROGRAM', 'TEAM', 'HEIGHT', 'WEIGHT', 'ARCHETYPE',
                  'PRICE', 'QUICKSELL', 'QS_CURRENCY', 'clutch', 'penalty', 'lb_style', 'dl_swim', 'dl_spin',
                  'dl_bull', 'big_hitter', 'strips_ball', 'ball_in_air', 'high_motor', 'covers_ball', 'extra_yards',
                  'agg_catch', 'rac_catch', 'poss_catch', 'drops_open', 'sideline_catch', 'qb_style', 'tight_spiral',
                  'sense_pressure', 'throw_away', 'force_passes', 'SPD', 'STR', 'AGI', 'ACC', 'AWA', 'CTH', 'JMP',
                  'STA', 'INJ', 'TRK', 'ELU', 'BTK', 'BCV', 'SFA', 'SPM', 'JKM', 'CAR', 'SRR', 'MRR', 'DRR', 'CIT',
                  'SPC', 'RLS', 'THP', 'SAC', 'MAC', 'DAC', 'RUN', 'TUP', 'BSK', 'PAC', 'RBK', 'RBP', 'RBF', 'PBK',
                  'PBP', 'PBF', 'LBK', 'IBL', 'TAK', 'POW', 'PMV', 'FMV', 'BSH', 'PUR', 'PRC', 'MCV', 'ZCV', 'PRS',
                 'KPW', 'KAC', 'RET'], 1, inplace=True)

    # speed_pos = ["POS_CB", "POS_HB", "POS_QB", "POS_MLB", "POS_QB", "POS_WR"]

    # OVR Permutations

    MutDataCopy["OVR"] = MutData["OVR"].values
    MutDataCopy["OVR_Squared"] = MutData["OVR"]**2
    MutDataCopy["OVR_Cubed"] = MutData["OVR"]**


# Plot

# fig = plt.figure(figsize=(30, 25))  # create the top-level container
# gs = gridspec.GridSpec(10, 12)
# ax = plt.subplot(gs[0:2, 0:2])
# ax.xaxis.set_ticks(np.arange(10000, 500000, 100000))


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
    # regr = ElasticNet(random_state=0, alpha=0.9).fit(X_train, y_train)  # Do not use fit_intercept = False if

# ---------Linear -------------


# self.coeff_df = pd.DataFrame(self.linear_regr.coef_, self.X.columns, columns=['Coefficient'])
# r_sq = self.linear_regr.score(self.X_train, self.y_train)

# print(self.coeff_df, r_sq)

# df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).apply(np.exp)
# print("Linear", df)
#
# model = sm.OLS(y_test, X_test).fit()
# print(model.summary())
# model = sm.OLS(y_train, X_train[:]).fit()
# MSEs = cross_val_score(linear_regr, X_train, y_train, scoring='neg_mean_squared_error', cv=5)


# ------------- Ridge

# ridge_regr = GridSearchCV(ridge, parameters, scoring = 'mean_squared_error', cv=5)
# print("Ridge best params = ", ridge_regr.best_params_)
# print("Ridge best score = ", ridge_regr.best_score_)
# parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}

# ---------- Lasso

# df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).apply(np.exp)
# print("Lasso ", df)


# QB Interactions
qb_pos = ["POS_QB"]
for col in qb_pos:
    self.MutData[col.split("_")[1] + "_" + "RUN"] = self.MutData[col].mul(self.MutData['RUN'])

# Lineman Interactions
lineman_pos = ["POS_RT", "POS_RG", "POS_C", "POS_LG", "POS_LT"]
for col in lineman_pos:
    self.MutData[col.split("_")[1] + "_" + "RBK"] = self.MutData[col].mul(self.MutData['RBK'])
    self.MutData[col.split("_")[1] + "_" + "RBP"] = self.MutData[col].mul(self.MutData['RBP'])
    self.MutData[col.split("_")[1] + "_" + "RBF"] = self.MutData[col].mul(self.MutData['RBF'])
    self.MutData[col.split("_")[1] + "_" + "PBK"] = self.MutData[col].mul(self.MutData['PBK'])
    self.MutData[col.split("_")[1] + "_" + "PBP"] = self.MutData[col].mul(self.MutData['PBP'])
    self.MutData[col.split("_")[1] + "_" + "PBF"] = self.MutData[col].mul(self.MutData['PBF'])

# SKill Positions
skill_pos = ["POS_HB", "POS_WR", "POS_TE"]

for col in skill_pos:
    self.MutData[col.split("_")[1] + "_" + "STR"] = self.MutData[col].mul(self.MutData['STR'])
    self.MutData[col.split("_")[1] + "_" + "AGI"] = self.MutData[col].mul(self.MutData['AGI'])
    self.MutData[col.split("_")[1] + "_" + "ACC"] = self.MutData[col].mul(self.MutData['ACC'])
    self.MutData[col.split("_")[1] + "_" + "CTH"] = self.MutData[col].mul(self.MutData['CTH'])
    self.MutData[col.split("_")[1] + "_" + "JMP"] = self.MutData[col].mul(self.MutData['JMP'])
    self.MutData[col.split("_")[1] + "_" + "BTK"] = self.MutData[col].mul(self.MutData['BTK'])
    self.MutData[col.split("_")[1] + "_" + "BCV"] = self.MutData[col].mul(self.MutData['BCV'])
    self.MutData[col.split("_")[1] + "_" + "SFA"] = self.MutData[col].mul(self.MutData['SFA'])
    self.MutData[col.split("_")[1] + "_" + "SPM"] = self.MutData[col].mul(self.MutData['SPM'])
    self.MutData[col.split("_")[1] + "_" + "JKM"] = self.MutData[col].mul(self.MutData['JKM'])
    self.MutData[col.split("_")[1] + "_" + "CAR"] = self.MutData[col].mul(self.MutData['CAR'])
    self.MutData[col.split("_")[1] + "_" + "SRR"] = self.MutData[col].mul(self.MutData['SRR'])
    self.MutData[col.split("_")[1] + "_" + "MRR"] = self.MutData[col].mul(self.MutData['MRR'])
    self.MutData[col.split("_")[1] + "_" + "DRR"] = self.MutData[col].mul(self.MutData['DRR'])
    self.MutData[col.split("_")[1] + "_" + "CIT"] = self.MutData[col].mul(self.MutData['CIT'])
    self.MutData[col.split("_")[1] + "_" + "SPC"] = self.MutData[col].mul(self.MutData['SPC'])
    self.MutData[col.split("_")[1] + "_" + "RLS"] = self.MutData[col].mul(self.MutData['RLS'])
    self.MutData[col.split("_")[1] + "_" + "RBK"] = self.MutData[col].mul(self.MutData['RBK'])
    self.MutData[col.split("_")[1] + "_" + "RBP"] = self.MutData[col].mul(self.MutData['RBP'])
    self.MutData[col.split("_")[1] + "_" + "RBF"] = self.MutData[col].mul(self.MutData['RBF'])
    self.MutData[col.split("_")[1] + "_" + "PBK"] = self.MutData[col].mul(self.MutData['PBK'])
    self.MutData[col.split("_")[1] + "_" + "PBP"] = self.MutData[col].mul(self.MutData['PBP'])
    self.MutData[col.split("_")[1] + "_" + "PBF"] = self.MutData[col].mul(self.MutData['PBF'])

# Pass Rusher Interactions
pass_rush_pos = ["POS_LE", "POS_DT", "POS_RE"]

for col in pass_rush_pos:
    self.MutData[col.split("_")[1] + "_" + "STR"] = self.MutData[col].mul(self.MutData['STR'])
    self.MutData[col.split("_")[1] + "_" + "TAK"] = self.MutData[col].mul(self.MutData['TAK'])
    self.MutData[col.split("_")[1] + "_" + "POW"] = self.MutData[col].mul(self.MutData['POW'])
    self.MutData[col.split("_")[1] + "_" + "PMV"] = self.MutData[col].mul(self.MutData['PMV'])
    self.MutData[col.split("_")[1] + "_" + "FMV"] = self.MutData[col].mul(self.MutData['FMV'])
    self.MutData[col.split("_")[1] + "_" + "BSH"] = self.MutData[col].mul(self.MutData['BSH'])
    self.MutData[col.split("_")[1] + "_" + "PUR"] = self.MutData[col].mul(self.MutData['PUR'])
    self.MutData[col.split("_")[1] + "_" + "PRC"] = self.MutData[col].mul(self.MutData['PRC'])

# Linebacker Interactions
linebacker_pos = ["POS_ROLB", "POS_LOLB", "POS_MLB"]
for col in linebacker_pos:
    self.MutData[col.split("_")[1] + "_" + "STR"] = self.MutData[col].mul(self.MutData['STR'])
    self.MutData[col.split("_")[1] + "_" + "TAK"] = self.MutData[col].mul(self.MutData['TAK'])
    self.MutData[col.split("_")[1] + "_" + "POW"] = self.MutData[col].mul(self.MutData['POW'])
    self.MutData[col.split("_")[1] + "_" + "PMV"] = self.MutData[col].mul(self.MutData['PMV'])
    self.MutData[col.split("_")[1] + "_" + "FMV"] = self.MutData[col].mul(self.MutData['FMV'])
    self.MutData[col.split("_")[1] + "_" + "BSH"] = self.MutData[col].mul(self.MutData['BSH'])
    self.MutData[col.split("_")[1] + "_" + "PUR"] = self.MutData[col].mul(self.MutData['PUR'])
    self.MutData[col.split("_")[1] + "_" + "PRC"] = self.MutData[col].mul(self.MutData['PRC'])
    self.MutData[col.split("_")[1] + "_" + "MCV"] = self.MutData[col].mul(self.MutData['MCV'])
    self.MutData[col.split("_")[1] + "_" + "ZCV"] = self.MutData[col].mul(self.MutData['ZCV'])

# DB Interactions
db_pos = ["POS_CB", "POS_FS", "POS_SS"]

for col in db_pos:
    self.MutData[col.split("_")[1] + "_" + "POW"] = self.MutData[col].mul(self.MutData['POW'])
    self.MutData[col.split("_")[1] + "_" + "BSH"] = self.MutData[col].mul(self.MutData['BSH'])
    self.MutData[col.split("_")[1] + "_" + "PUR"] = self.MutData[col].mul(self.MutData['PUR'])
    self.MutData[col.split("_")[1] + "_" + "PRC"] = self.MutData[col].mul(self.MutData['PRC'])
    self.MutData[col.split("_")[1] + "_" + "MCV"] = self.MutData[col].mul(self.MutData['MCV'])
    self.MutData[col.split("_")[1] + "_" + "ZCV"] = self.MutData[col].mul(self.MutData['ZCV'])
    self.MutData[col.split("_")[1] + "_" + "PRS"] = self.MutData[col].mul(self.MutData['PRS'])

# Kicker/Punter Interactions
kick_pos = ["POS_K", "POS_P"]

for col in kick_pos:
    self.MutData[col.split("_")[1] + "_" + "KPW"] = self.MutData[col].mul(self.MutData['KPW'])
    self.MutData[col.split("_")[1] + "_" + "KAC"] = self.MutData[col].mul(self.MutData['KAC'])