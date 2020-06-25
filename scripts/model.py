import time,os,re,csv,sys,uuid,joblib
from datetime import date
from math import floor
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from data_ingestion import fetch_ts, create_lag


data_dir=r'C:\Users\nomic\Desktop\final project ibm\data'
dfs=fetch_ts(data_dir,restart=False)
df=dfs['all']

df=create_lag(df,['revenue','unique_invoices','total_views'],[1,2,3,7,14,28,90,180],['date'],'date')
#create_lag(dfs['all'])



#train, validation and test split
val_perc=0.2
test_perc=0.1
random_state=0

df['months']=df.apply(lambda row: str(row['date'].year)+'-'+str(row['date'].month).zfill(2),axis=1)

#TRAIN and TEST split
months=df['months'].unique()
test_size=floor(len(months)*test_perc)
months_test=months[-test_size:]

mask_test=df['months'].isin(months_test)

train=df[~mask_test]
test=df[mask_test]

#TRAIN and VALIDATION split
months=train['months'].unique()
val_size=floor(len(months)*val_perc)
months_val=months[-val_size:]

mask_val=train['months'].isin(months_val)

validation=train[mask_val]
train=train[~mask_val]

x_col=df.columns.values
y_col='revenue'
x_col=x_col[7:]
x_col

def df_split(train,x_col,ycol):
    X=df[x_col]
    y=df[ycol]
    return X,y
        
X_train,y_train=df_split(train,x_col,y_col)
X_val,y_val=df_split(validation,x_col,y_col)





n_estimators=[100,200,300,400]
max_depth=[10,20,30,40]
param_grid={
            'model__n_estimators': [100,200],
            'model__max_depth':[10,20,30,40]
           }


for estimator in n_stimators:
    for depth in max_depth:
        random_forest= RandomForestRegressor(random_state=random_state,n_jobs=-1,max_depth=depth,n_estimators=estimator)
        pipeline = Pipeline([('scaler', StandardScaler()),('model', random_forest)])
        random_forest.fit(X_train,y_train)
        y_val_pred=random_forest.predict(X_val)
        print('estimator {} and depth {}\n'.format(estimator,depth))
        print(mean_squared_error(y_val,y_val_pred))
    


# Create the GridSearchCV object: gm_cv
random_forest= RandomForestRegressor(random_state=random_state,n_jobs=-1)
pipeline = Pipeline([('scaler', StandardScaler()),('model', random_forest)])
gm_cv = GridSearchCV(pipeline, param_grid)
gm_cv.fit(X_train,y_train)

best=gm_cv.best_estimator_
np.sqrt(mean_squared_error(best.predict(X_val),y_val))

import sklearn.ensemble

dir(sklearn.ensemble)








