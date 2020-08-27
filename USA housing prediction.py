# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 01:52:48 2020

@author: VIDHI
"""

#Load the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import PolynomialFeatures 
from yellowbrick.regressor import PredictionError
from yellowbrick.regressor import ResidualsPlot
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

#Importing USA Housing Dataset
data = pd.read_excel(r'C:\Users\VIDHI\Desktop\USA_Housing.xlsx')

#Explorartory data analysis
data.head()
data.info()
data.describe()
print(data.shape)

#Defining input and output values
X = data.iloc[:,:5]
y = data.iloc[:,5]

#Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 21)

#Scaling the data
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Fitting the linear regression model and making predictions
LR1 = LinearRegression()
LR1.fit(X_train, y_train)
predictions = LR1.predict(X_test)

#Assessing model performance
score1 = LR1.score(X_test, y_test)
sc = explained_variance_score(y_test, predictions)
mae = MAE(y_test, predictions)
mse = MSE(y_test, predictions)
rmse = mse**(1/2)

#Fitting polynomial regression model 
poly = PolynomialFeatures(degree = 4) 
X_poly = poly.fit_transform(X) 
  
poly.fit(X_poly, y)  
LR1.fit(X_poly, y)

#Applying cross validation
cv_results = cross_val_score(LR1, X, y, cv = 5)
print(cv_results)

#Fitting ridge regression and making predictions
ridge = Ridge(alpha = 0.001, normalize = True)
ridge.fit(X_train, y_train)
predictions2 = ridge.predict(X_test)

#selecting alpha 
alph = np.arange(0.001, 0.1, 0.001)
r2score =[]
for i in alph:
    model = Ridge(alpha = i, normalize = True)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2score.append(model.score(X_test,y_test))

plt.plot(alph, r2score, '-o')
plt.xlabel('alpha value')
plt.ylabel('r2 score')

plt.show()


#assessing ridge regression performance
score2 = ridge.score(X_test, y_test)
mae2 = MAE(y_test, predictions2)
mse2 = MSE(y_test, predictions2)
rmse2 = mse2**(1/2)

#fitting lasso regression and making predictions
lasso = Lasso(alpha = 14)
lasso.fit(X_train, y_train)
predictions3 = lasso.predict(X_test)
score3 = lasso.score(X_test, y_test)

#assessing performance of lasso
mae3 = MAE(y_test, predictions3)
mse3 = MSE(y_test, predictions3)
rmse3 = mse3**(1/2)

#feature importance
lasso_coef = lasso.fit(X, y).coef_

#visualizing regression model
visualizer = PredictionError(lasso)
visualizer.fit(X_train, y_train)  
visualizer.score(X_test, y_test)  
visualizer.show()

#visualizing regression residuals plot
visualizer2 = ResidualsPlot(lasso)
visualizer2.fit(X_train, y_train)  
visualizer2.score(X_test, y_test)  
visualizer2.show()

print('Linear Regression Score: ', score1) 
print('Ridge Regression Score: ', score2)
print('Lasso Regression Score: ', score3)










