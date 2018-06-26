#Author: Brandon Liang
#Version: 6/26/18

import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
import numpy as np
import statsmodels.formula.api as smf
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.svm import SVR
from sklearn.metrics import *
from sklearn import preprocessing, linear_model

X = pd.read_excel("pmad_pva.xlsx", '_TB01')
X = X[X.TargetD.notnull()]

y = pd.Series(X['TargetD'])
X = X.drop(['TargetB', 'ID', 'TargetD'], axis=1)

BIN_DemCluster = [4,8,9,16,19,20,23,25,26,27,
					28,30,32,33,36,38,39,40,41,43,
					45,46,47,48,49,51,52,53]

X['StatusCat96NK'] = (X['StatusCat96NK'] == 'S').astype('int')
X['DemGender'] = (X['DemGender'] == 'F').astype('int')
X['DemHomeOwner'] = (X['DemHomeOwner'] == 'H').astype('int')

X.loc[:, 'DemCluster'] = X.loc[:, 'DemCluster'].isin(BIN_DemCluster).astype(int)

X_preprocess = X
int_types = []

# Label encode non-numeric features
for column in X_preprocess.columns:
    if X_preprocess[column].dtype == type(object):
        le = preprocessing.LabelEncoder()
        X_preprocess[column] = le.fit_transform(X_preprocess[column])
    if X[column].dtype == 'int64':
        int_types.append(column)

# Fill missing values with median 
imp = preprocessing.Imputer(missing_values='NaN', strategy='median', axis=0) 
X_preprocess = pd.DataFrame(imp.fit_transform(X_preprocess), columns = X_preprocess.columns)

# Drop low variance columns
sel = VarianceThreshold(threshold=0.95)	
sel.fit(X_preprocess)
remaining_columns = X_preprocess.iloc[:, sel.variances_ > 0.95].columns
X_preprocess = pd.DataFrame(sel.transform(X_preprocess), columns = remaining_columns)

# Drop highly correlated columns (copied from Chris Albon)
corr_matrix = X_preprocess.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
X_preprocess.drop(X_preprocess.columns[to_drop], axis=1)

# Box-Cox where appropriate, center/scale, then apply spatial sign transform
for column in X_preprocess.columns :
	if X_preprocess[column].all() > 0.0:
		X_preprocess[column], dummyVar = stats.boxcox(X_preprocess[column])
	X_preprocess[column] = pd.Series(preprocessing.scale(X_preprocess[column]))
	X_preprocess[column] = np.linalg.norm(X_preprocess[column])/X_preprocess[column]
 	
for column in X_preprocess.columns:
	smresults = sm.OLS(list(y), X_preprocess[column]).fit()
	X_preprocess[column] = pd.DataFrame(smresults.predict())

X = X_preprocess

# Linear regression and backwards stepwise regression models
# 'forward' and 'both' coefficients in the paper were identical to 'lm' and 'backward'
list_model = dict()
list_model['lm'] = linear_model.LinearRegression()
estimator = list_model['lm']
list_model['lm'].fit(X, y)

# By default takes out half of the features
list_model['backward'] = RFE(estimator) 
X_reduced = pd.DataFrame(list_model['backward'].fit_transform(X, y))
list_model['backward'] = linear_model.LinearRegression()
list_model['backward'].fit(X_reduced, y)

print('\nlm coefficients: \n', list_model['lm'].coef_, '\n')
print('backward coefficients: \n', list_model['backward'].coef_, '\n\n')

# Print training performance metrics
y_hat = list_model['lm'].predict(X)
y_backwards_hat = list_model['backward'].predict(X_reduced)
print("          \t lm              \t backward")
print("TrainRMSE: ", np.sqrt(mean_squared_error(y, y_hat)), " ", np.sqrt(mean_squared_error(y, y_backwards_hat)))
print("TrainRsquared: ", r2_score(y, y_hat), "  ", r2_score(y, y_backwards_hat))
print("TrainMAE: ", mean_absolute_error(y, y_hat), "  ", mean_absolute_error(y, y_backwards_hat), "\n")

# ANOVA tests and and reduced model summary
print("ANOVA: ", stats.f_oneway(list_model['lm'].coef_, list_model['backward'].coef_))
print("\nReduced model summary: \n\n", X_reduced.describe().transpose().head())
