
from os import listdir, mkdir
from os.path import isdir, join,exists
import re
from datetime import date,timedelta
import numpy as np
import shutil
import pandas as pd
#def fetch_data(data_dir):
    


def fetch_data(data_dir):
    '''   
    Fetch the json files 
    Parameters
    ----------
    data_dir : file name (str) importnat to have a folder called 'raw' within this folder
    Raises
    ------
    Exception: Directory does not exist,  no files, files do not have the expectect number of colmns
    Returns
    -------
    data : Data Frame
    '''
    
    data_dir=join(data_dir,'raw')
    print('Reading raw data from:{}\n'.format(data_dir))
    #input testing
    if not isdir(data_dir):
        raise Exception('does not exist')
    if not len(listdir(data_dir))>0:
        raise Exception('No files')
        
    files=[f for f in listdir(data_dir) if re.findall('.json$',f)]
    columns=['country','customer_id','invoice','price','stream_id','times_viewed', 'year','month', 'day']
    
    data={}
    for num,file in enumerate(files):
        # JSON file 
        _=pd.read_json(join(data_dir,file))
        
        #Check number of columns
        if len(_.columns)!=len(columns):
            raise Exception ('File {} has different columns'.format(file))
            
        _.columns=columns
        _['file']=num
        data[num]=_
    
    data=pd.concat(data.values())
    
    data['date']=data.apply(lambda row: date(row['year'],row['month'],row['day']),axis=1)
    data['date']=pd.to_datetime(data['date'],format='%Y-%m-%d')
    
    
    #remove characters in invoice
    data['invoice']=data['invoice'].apply(lambda x:re.sub("\D",'', x)).astype('int32')
    #a=data[data['stream_id']=='gift_0001_90'].loc[:,'stream_id'].apply(lambda x:''.join(re.findall("[^0-9]",x)))   not sure if needed
    #drop redundant columns
    data.drop(columns=['year','month','day'],inplace=True)
    
    #delete negative prices
    before=data.shape[0]
    data=data[data['price']>0]
    after=data.shape[0]
    print('Deleted rows (negative price):{}'.format(before-after))
    return data
    
def convert_to_ts(df_orig, country=None):
    '''
    Create a df filtered by country and return a grouped df by day
    '''
    
    if country:
        if country not in df_orig['country'].unique():
            raise Exception("country not found")
    
        mask = df_orig['country'] == country
        df = df_orig[mask]
        
    else:
        df = df_orig
        
    #grouping by day
    days = np.arange( df['date'].min(),df['date'].max(),dtype='datetime64[D]')
    purchases = [df[df['date']==day].loc[:,'price'].count() for day in days]
    invoices= [len(df[df['date']==day].loc[:,'invoice'].unique()) for day in days]
    streams= [len(df[df['date']==day].loc[:,'stream_id'].unique()) for day in days]
    views= [df[df['date']==day].loc[:,'times_viewed'].sum() for day in days]
    revenue= [df[df['date']==day].loc[:,'price'].sum() for day in days]
    
    df_time = pd.DataFrame({'date':days,
                            'purchases':purchases,
                            'unique_invoices':invoices,
                            'unique_streams':streams,
                            'total_views':views,
                            'revenue':revenue})
    return df_time

def fetch_ts(data_dir,restart=False):
    '''Given the directory it creates the ts data  and the csv files  '''  
      
    ts_data_dir=join(data_dir,'processed')
    
    if restart:
        shutil.rmtree(ts_data_dir)
    
    if not exists(ts_data_dir):
        mkdir(ts_data_dir)

    #Load ALL csv files if they exist ina dictionary
    csvs=[i for i in listdir(ts_data_dir) if re.search('.csv$',i)]
    
    
    if len(csvs)>0:
        print('Reading processeded data from: {}'.format(ts_data_dir))
        return {re.sub('.csv','',file):pd.read_csv(join(ts_data_dir,file))for file in csvs}
    

    #Load the original data in case there is no processed data
    df=fetch_data(data_dir)
    
    #analyse <=98% of the revenue 
    countries=df.pivot_table(index='country', values='price', aggfunc=np.sum).sort_values(by='price',ascending=False)
    countries['perc']=countries['price']/countries['price'].sum()
    countries['perc_acum']=countries['perc'].cumsum()
    countries=countries[countries['perc_acum']<=0.98].index.values
    countries=np.insert(countries,0,'all')
    

    
    print('Saving processeded data to: {}'.format(ts_data_dir))
    
    dfs={}
    for country in countries:

        if country=='all':
            df_country=convert_to_ts(df)
        else:
            df_country=convert_to_ts(df,country=country)
        dfs[country]=df_country
        
        #Save csv files
        df_country.to_csv(join(ts_data_dir,country+'.csv'),index=False)
            

    return dfs


'''
FEATURE ENGINEERING
'''
def create_lag(df,cols,lags,join_keys,date_col):
    for col in cols:
        for lag in lags:
            db1=df.copy()
            db1=db1[join_keys+[col]]
            db1[date_col]+=timedelta(lag)
            db1=db1.rename(columns={col:col+'_lag_'+str(lag)})
            df = pd.merge(df, db1, on=join_keys, how='left')
    max_lag=max(lags)
    df=df[(max_lag):]
    df=df.fillna(0)    
    return df


if __name__=='__main__':
    data_dir=r'C:\Users\nomic\Desktop\final project ibm\data'
    dfs=fetch_ts(data_dir,restart=True)

    
    
    


