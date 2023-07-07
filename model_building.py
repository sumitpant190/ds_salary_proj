# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('eda_data.csv')

#choose relevant column
df.columns
df_model = df[['avg_salary','Rating','Size','Type of ownership','Industry','Sector','Revenue','num_comp','hourly','employer_provided','job_state','same_state','age','python_yn','spark','aws','excel','job_simp','seniority','desc_len']]


#get dummy variables
df_dum = pd.get_dummies(df_model)

#train-test split
from sklearn.model_selection import train_test_split

x = df_dum.drop('avg_salary' ,axis =1)
y = df_dum.avg_salary.values

x_train,x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state =42)


#multiple linear regression
import statsmodels.api as sm
 
x_sm = x = sm.add_constant(x)
model = sm.OLS(y,x_sm)
model.fit().summary()

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

lm = LinearRegression()
lm.fit(x_train, y_train)

np.mean(cross_val_score(lm, x_train, y_train, scoring = 'neg_mean_absolute_error', cv =3))


 

#lasso regression
lm_1 = Lasso(alpha=0.13)
lm_1.fit(x_train,y_train)
np.mean(cross_val_score(lm_1, x_train, y_train, scoring = 'neg_mean_absolute_error', cv =3))

alpha = []
error = []

for i in range(1,100):
    alpha.append(i/100)
    lm1 = Lasso(alpha =(i/100))
    error.append(np.mean(cross_val_score(lm_1, x_train, y_train, scoring = 'neg_mean_absolute_error', cv =3)))

plt.plot(alpha,error)

err = tuple(zip(alpha,error))
df_err = pd.DataFrame(err, columns = ['alphe','error'])
df_err[df_err.error == max(df_err.error)]

# random forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

np.mean(cross_val_score(rf,x_train, y_train, scoring = 'neg_mean_absolute_error' ,cv=3))


# tune models Gridsearch
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators': range(10,300,10), 'criterion':('squared_error','absolute_error'), 'max_features':('sqrt','log2')}

gs = GridSearchCV(rf,parameters,scoring='neg_mean_absolute_error',cv =3)
gs.fit(x_train,y_train)

gs.best_score_
gs.best_estimator_

# test ensembles
tpred_lm = lm.predict(x_test)
tpred_lml = lm_1.predict(x_test)
tpred_rf = gs.best_estimator_.predict(x_test)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,tpred_lm)
mean_absolute_error(y_test,tpred_lml )
mean_absolute_error(y_test,tpred_rf)

mean_absolute_error(y_test,(tpred_lm + tpred_rf)/2)


import pickle
pick1 = {'model' : gs.best_estimator_}
pickle.dump(pick1, open('model_file' + ".p","wb"))

file_name = "model_file.p"
with open(file_name,'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']

model.predict(x_test.iloc[1,:].values.reshape(1,-1))



