
import pandas as pd
import time as tm
import os


class frozenCleaner():

    '''Class that contains multiple methods to detect frozen data'''
    
    def __init__(self,file_name:str='',source:pd.DataFrame=pd.DataFrame()):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if file_name:
            self.source_data = pd.read_csv(os.path.join(self.base_dir,'data',file_name), index_col=0)
        else:
            self.source_data = source

    def frozen_by_threshold(self,target:str,df:pd.DataFrame,ts_column:str,threshold:float=0.00001):
        '''Detects Frozen data using a threshold of change. Use the sensor sensitivity as the threshold'''
        t = tm.time()
        df=df.sort_values(by=ts_column,ascending=True)
        # Detect frozen values
        df['change'] = df[target].diff().abs()
        df['is_frozen'] = df['change'].le(threshold)
        outliers = df[df.is_frozen == True]
        execution_time = tm.time() - t
        return outliers,execution_time
        
    def frozen_by_window(self,target:str,ts_column:str,window_size:int=3):
        '''Detects Frozen data using a threshold of change. Use the sensor sensitivity as the threshold'''
        t = tm.time()
        df = self.source_data
        # Detect frozen values using a rolling window
        df['rolling_std'] = df[target].rolling(window=window_size).std()
        df['is_frozen'] = df['rolling_std'].eq(0)
        outliers = df[df.is_frozen == True]
        execution_time = tm.time() - t
        return outliers,execution_time