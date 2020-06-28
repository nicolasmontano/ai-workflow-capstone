#basics
import pandas as pd
import numpy as np
import math
import re 
import time
import joblib

#OS
from os.path import join

#plots
import matplotlib.pyplot as plt
import seaborn as sns
from math import floor
plt.style.use('seaborn')

#data ingestion package
from data_ingestion import fetch_ts, create_lag

#modeling
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
#models
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
#metrics
from sklearn.metrics import mean_squared_error

#LOGGING
import logging
logger=logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter=logging.Formatter('%(asctime)s------ %(message)s')


file_handler=logging.FileHandler('Info.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


'''
LOAD DATA
'''

def load_data(data_dir,country='all'):
    df=fetch_ts(data_dir,restart=False)[country]
    df=create_lag(df,['revenue','unique_invoices','total_views'],[1,2,3,4,5,6,7,14,28,90,180],['date'],'date')
    if country=='all':
        df['revenue']=df['revenue'].apply(lambda x: 25000 if x>25000 else x)
        df['weekday']=df['date'].apply(lambda x: x.weekday())
    return df


'''
TRAIN, TEST SPLIT
'''
def train_val_test_split(df,val_perc=0.2,test_perc=0.1):
    
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
    print('train: {}\nvalidation: {}\ntest: {}'.format(train.shape[0],validation.shape[0],test.shape[0]))
    
    return train,validation,test
  
def df_split(df,x_col,ycol):
    X=df[x_col]
    y=df[ycol]
    
    return X,y

'''
PREPORCESSING
'''

def preprocessing(df,categorical_features):
  numeric_features=[col for col in df.columns.values if col not in categorical_features]

  numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
  categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

  preprocessor = ColumnTransformer(
      transformers=[
          ('num', numeric_transformer, numeric_features),
          ('cat', categorical_transformer, categorical_features)])
  
  return preprocessor


'''
TRAINING
'''
def train_rf_regressor(X_train,y_train,X_valid,y_valid,X_test,y_test,preprocessor):
  n_estimators=[100,200,300,400]
  max_depth=[10,20,30,40]
  min_mse=9999
  model=None
  for estimator in n_estimators:
      for depth in max_depth:
          rf= RandomForestRegressor(n_jobs=-1,max_depth=depth,n_estimators=estimator)
          rf_regressor = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor',rf )])
          rf_regressor.fit(X_train,y_train)
          y_val_pred=rf_regressor.predict(X_valid)
          if np.sqrt(mean_squared_error(y_valid,y_val_pred))<min_mse:
            min_mse=np.sqrt(mean_squared_error(y_valid,y_val_pred))
            model=rf_regressor
  y_valid_pred=model.predict(X_valid)  
  y_test_pred=model.predict(X_test)         
  print('Validation Error:{}'.format(np.sqrt(mean_squared_error(y_valid,y_val_pred))))
  print('Test Error {}'.format(np.sqrt(mean_squared_error(y_test,y_test_pred))))
  return model



def train_xgboost_model(X_train,y_train,X_valid,y_valid,X_test,y_test,preprocessor):
  max_depth=[15,20]
  n_estimators=[120,150,200,210]
  min_child_weight=[4,5,6]

  min_score=math.inf
  model_xgb=None
  for depth in max_depth:
    for est in n_estimators:
      for min_child in min_child_weight:
              
        xgb_reg= xgb.XGBRegressor( n_jobs=-1,
                                      num_boost_round=400,
                                      max_depth=depth,
                                      min_child_weigh=min_child,
                                      n_estimators=est,
                                      learning_rate=0.01,
                                      early_stopping_rounds=5)
        xgb_regressor = Pipeline(steps=[('preprocessor', preprocessor),('model',xgb_reg )])
        xgb_regressor.fit(X_train,y_train)
        score=mean_squared_error(xgb_regressor.predict(X_valid),y_valid)
        if score<min_score:
          min_score=score
          model_xgb=xgb_regressor

  y_val_pred=model_xgb.predict(X_valid)
  y_test_pred=model_xgb.predict(X_test)
  print('Validation Error:{}'.format(np.sqrt(mean_squared_error(y_valid,y_val_pred))))
  print('Test Error {}'.format(np.sqrt(mean_squared_error(y_test,y_test_pred))))

  return model_xgb
  
 


'''
SAVE MODEL
'''


def save_model(MODEL_DIR,MODEL_VERSION,name,model):
  saved_model=join(MODEL_DIR,'{}_{}.joblib'.format(name,MODEL_VERSION))
  joblib.dump(model,saved_model)

'''
LOAD MODEL
'''
def load_model(MODEL_DIR,name):
  try:
    return joblib.load(join(MODEL_DIR,name))
  except:
    print('{} not found in {}'.format(name,MODEL_DIR))
  




'''
WORKFLOW
'''

def training_models(df,MODEL_VERSION,MODEL_DIR):
    
    run_start = time.time() 
    
    x_col=[col for col in df.columns if col not in ['date','purchases','unique_invoices','unique_streams','total_views','revenue','months']]
    y_col='revenue'
    
    train,validation,test=train_val_test_split(df)
    X_train,y_train=df_split(train,x_col,y_col)
    X_valid,y_valid=df_split(validation,x_col,y_col)
    X_test,y_test=df_split(test,x_col,y_col)
    
    preprocessor= preprocessing(X_train,['weekday'])
    
    #Random forest
    rf_regressor=train_rf_regressor(X_train,y_train,X_valid,y_valid,X_test,y_test,preprocessor)
    #Xgboost
    xgb_regressor=train_xgboost_model(X_train,y_train,X_valid,y_valid,X_test,y_test,preprocessor)
    
    save_model(MODEL_DIR,MODEL_VERSION,'rf_regressor',rf_regressor)
    save_model(MODEL_DIR,MODEL_VERSION,'xgb_regressor',xgb_regressor)
    
    results=pd.DataFrame()    
    results['model']=['rf_regressor','xgb_regressor']
    
    results['val_score']=[np.sqrt(mean_squared_error(y_valid,rf_regressor.predict(X_valid))),
                      np.sqrt(mean_squared_error(y_valid,xgb_regressor.predict(X_valid)))]
    
    results['test_score']=[np.sqrt(mean_squared_error(y_test,rf_regressor.predict(X_test))),
                      np.sqrt(mean_squared_error(y_test,xgb_regressor.predict(X_test)))]
       
    m, s = divmod(time.time()-run_start,60)
    h, m = divmod(m, 60)
    
    logger.info('Training time: {0:.0f}:{1:.0f}:{2:.0f}'.format(h, m, s))
    
    return results


if __name__=='__main__':
    run_start = time.time() 
    
    data_dir=r'C:\Users\nomic\Desktop\final project ibm\data'
    MODEL_VERSION=1
    MODEL_DIR=r'C:\Users\nomic\Desktop\final project ibm\models'
    
    df=load_data(data_dir)
    a=training_models(df,MODEL_VERSION,MODEL_DIR)

    m, s = divmod(time.time()-run_start,60)
    h, m = divmod(m, 60)
    print("load time:", "%d:%02d:%02d"%(h, m, s))



    
