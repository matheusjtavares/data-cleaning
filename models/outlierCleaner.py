
import pandas as pd
import numpy as np
import seaborn as sns
import time as tm
import os
from scipy.stats import t
from scipy.stats import kstest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses, Model
from random import randint

class outlierCleaner():
    '''Class that contains multiple methods to clean outlier data'''
    def __init__(self,file_name:str='',source:pd.DataFrame=pd.DataFrame()):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.file_name = file_name
        if file_name:
            self.source_data = pd.read_csv(os.path.join(self.base_dir,'data',file_name), index_col=0)
            
        else:
            self.source_data = source
        self.npz_path = f"{os.path.join(self.base_dir,'data',file_name)}"
        np.savez(file_name,self.source_data.to_numpy())
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
 
    def z_score(self,target:str,df:pd.DataFrame,abs_limit:int =3):
        '''Applies the z-score method for outlier detection'''
        t = tm.time()
        df['Z_Score'] = (df[target] - df[target].mean()) / df[target].std()
        outliers = df[df['Z_Score'].abs() > abs_limit]
        df = df.drop(columns=['Z_Score'])
        execution_time = tm.time() - t
        return outliers,execution_time
    
    def _train_tensor_flow_model(self):
        '''Function to train the ML model in which the prediction of values will run'''
        (x_train, _), (x_test, _)=tf.keras.datasets.mnist.load_data(f'{self.file_name}.npz')
        hidden_size = 100
        latent_size = 20
        input_layer = layers.Input(shape = x_train.shape[1:])
        flattened = layers.Flatten()(input_layer)
        hidden = layers.Dense(hidden_size, activation = 'relu')(flattened)
        latent = layers.Dense(latent_size, activation = 'relu')(hidden)
        encoder = Model(inputs = input_layer, outputs = latent, name = 'encoder')
        encoder.summary()
        input_layer_decoder = layers.Input(shape = encoder.output.shape)
        upsampled = layers.Dense(hidden_size, activation = 'relu')(input_layer_decoder)
        upsampled = layers.Dense(encoder.layers[1].output_shape[-1], activation = 'relu')(upsampled)
        constructed = layers.Reshape(x_train.shape[1:])(upsampled)
        decoder = Model(inputs = input_layer_decoder, outputs = constructed, name= 'decoder')
        decoder.summary()
        return
    def modified_z_score(self,target:str,df:pd.DataFrame,limit:int = 3.5):
        '''Applies the modified z-score method for outlier detection (using MAD)'''
        t = tm.time()
        median = df[target].median()
        mad = (np.abs(df[target] - median)).median()
        df['Modified_Z_Score'] = 0.6745 * (df[target] - median) / mad
        outliers = df[df['Modified_Z_Score'].abs() > limit]
        df = df.drop(columns=['Modified_Z_Score'])
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
    
    def grubbs_test(self,target:str,alpha:int =0.05):
        '''Applies the grubbs testfor outlier detection'''
        df = self.source_data

        def grubbs_method(data, alpha=alpha):
            mean = np.mean(data)
            std_dev = np.std(data)
            N = len(data)
            G = max(abs(data - mean)) / std_dev
            t_value = t.ppf(1 - alpha / (2 * N), N - 2)
            critical_value = ((N - 1) * np.sqrt(t_value**2)) / (np.sqrt(N) * np.sqrt(N - 2 + t_value**2))
            return G > critical_value
        outliers = df[df[target].apply(lambda x: grubbs_method(df[target]))]

        return outliers

if __name__== '__main__':
    oc = outlierCleaner(file_name='T1.csv')
    oc._train_tensor_flow_model()