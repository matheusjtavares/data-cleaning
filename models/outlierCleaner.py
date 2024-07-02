
import pandas as pd
import numpy as np
import seaborn as sns
import time as tm
import os
from scipy.stats import t
from scipy.stats import kstest


class outlierCleaner():
    '''Class that contains multiple methods to clean outlier data'''
    def __init__(self,file_name:str='',source:pd.DataFrame=pd.DataFrame()):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if file_name:
            self.source_data = pd.read_csv(os.path.join(self.base_dir,'data',file_name))
        else:
            self.source_data = source
            
    def test_for_normal(self,target:str):
        df = self.source_data
        stat, p_value = kstest(df[target], 'norm', args=(df[target].mean(), df[target].std()))
        print(f'Statistic: {stat}, p-value: {p_value}')

        if p_value > 0.05:
            print('The column follows a normal distribution (fail to reject H0)')
            return True
        else:
            print('The column does not follow a normal distribution (reject H0)')
            return False
 
    def z_score(self,target:str,abs_limit:int =3):
        '''Applies the z-score method for outlier detection'''
        t = tm.time()
        df = self.source_data
        df['Z_Score'] = (df[target] - df[target].mean()) / df[target].std()
        outliers = df[df['Z_Score'].abs() > abs_limit]
        execution_time = tm.time() - t
        return outliers,execution_time
    
    def modified_z_score(self,target:str,limit:int =3.5):
        '''Applies the modified z-score method for outlier detection (using MAD)'''
        t = tm.time()
        df = self.source_data
        median = df[target].median()
        mad = (np.abs(df[target] - median)).median()
        df['Modified_Z_Score'] = 0.6745 * (df[target] - median) / mad
        outliers = df[df['Modified_Z_Score'].abs() > limit]
        execution_time = tm.time() - t
        return outliers,execution_time
    
    def iqr(self,target:str,limit:int =3.5):
        '''Applies the interquantile method for outlier detection'''
        t = tm.time()
        df = self.source_data
        Q1 = df[target].quantile(0.25)
        Q3 = df[target].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[target] < (Q1 - 1.5 * IQR)) | (df[target] > (Q3 + 1.5 * IQR))]
        execution_time = tm.time() - t
        return outliers,execution_time
    
    def grubbs_test(self,target:str,alpha:int =3.5):
        '''Applies the grubbs testfor outlier detection'''
        t = tm.time()
        df = self.source_data
        def grubbs_method(data, alpha=0.05):
            mean = np.mean(data)
            std_dev = np.std(data)
            N = len(data)
            G = max(abs(data - mean)) / std_dev
            t_value = t.ppf(1 - alpha / (2 * N), N - 2)
            critical_value = ((N - 1) * np.sqrt(t_value**2)) / (np.sqrt(N) * np.sqrt(N - 2 + t_value**2))
            return G > critical_value
        outliers = df[df[target].apply(lambda x: grubbs_method(df[target]))]
        execution_time = tm.time() - t

        return outliers,execution_time