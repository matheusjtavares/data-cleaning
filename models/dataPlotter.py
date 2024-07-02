
import pandas as pd
import numpy as np
import seaborn as sns
import time as tm
import os
from scipy.stats import t
import matplotlib.pyplot as plt

class dataPlotter():
    '''Class that contains multiple methods to clean outlier data'''

    def __init__(self,file_name:str='',source:pd.DataFrame=pd.DataFrame()):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        print(self.base_dir)
        if file_name:
            self.source_data = pd.read_csv(os.path.join(self.base_dir,'data',file_name))
        else:
            self.source_data = source
        self.chart_output_path = os.path.join(self.base_dir,'output/charts')
     
    def make_histogram(self,target:str,bins:int=30,chart_name:str='histogram.png'):
        df = self.source_data
        df[target].hist(bins=bins, edgecolor='k')
        plt.title(f'Histogram - {target}')
        plt.xlabel(target)
        plt.ylabel('Frequency')
        plt.show()
        plt.savefig(os.path.join(self.chart_output_path,chart_name))
        return True
    
    def make_box_plot(self,target:str,chart_name:str='box_plot.png'):
        df = self.source_data
        sns.boxplot(x=df[target])
        plt.title(f'Box Plot - {target}')
        plt.show()
        plt.savefig(os.path.join(self.chart_output_path,chart_name))
        return True