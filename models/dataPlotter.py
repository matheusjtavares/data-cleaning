
import pandas as pd
import numpy as np
import seaborn as sns
import time as tm
import os
import scipy.stats as stats
from scipy.stats import t
from matplotlib import pyplot as plt
import random
class dataPlotter():
    '''Class that contains multiple methods to clean outlier data'''

    def __init__(self,file_name:str='',source:pd.DataFrame=pd.DataFrame()):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        print(self.base_dir)
        if file_name:
            self.source_data = pd.read_csv(os.path.join(self.base_dir,'data',file_name), index_col=0)
            self.source_data.index = pd.to_datetime(self.source_data.index,format = '%d %m %Y %H:%M')
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
        plt.close()
        return True
    
    def make_all_histogram(self,targets:str=[],bins:int=30,width:int=12,height:int=10,chart_name:str='histogram.png'):
        df = self.source_data
        if targets == []:
            targets = list(df.columns)
            # plot the data with subplots and assign the returned array

        axes = df.plot.density(subplots=True,figsize=(width,height),sharex=False,sharey=False)

        # flatten the array
        axes = axes.flat  # .ravel() and .flatten() also work

        # extract the figure object to use figure level methods
        fig = axes[0].get_figure()

        # iterate through each axes to use axes level methods
        for ax in axes:
            ax.legend(loc='upper center')
        fig.suptitle('TimeSeries', fontsize=22, y=0.95)
        plt.show()
        return True
        
    def make_qq_plot(self,target:str,chart_name:str='box_plot.png'):
        df = self.source_data
        stats.probplot(df[target], dist="norm", plot=plt)
        plt.title('QQ Plot')
        plt.show()
        plt.savefig(os.path.join(self.chart_output_path,chart_name))
        plt.close()
        return True
    
    def make_box_plot(self,target:str,chart_name:str='box_plot.png'):
        df = self.source_data
        sns.boxplot(x=df[target])
        plt.title(f'Box Plot - {target}')
        plt.show()
        plt.savefig(os.path.join(self.chart_output_path,chart_name))
        plt.close()
        return True
    
    def plot_time_series(self,targets:list = [],width:int=12,height:int=10,chart_name:str='time_series.png'):
        df = self.source_data
        if targets == []:
            targets = list(df.columns)
            # plot the data with subplots and assign the returned array

        axes = df.plot(subplots=True,figsize=(width,height),sharex=True)

        # flatten the array
        axes = axes.flat  # .ravel() and .flatten() also work

        # extract the figure object to use figure level methods
        fig = axes[0].get_figure()

        # iterate through each axes to use axes level methods
        for ax in axes:
            ax.legend(loc='upper center')
        fig.suptitle('TimeSeries', fontsize=22, y=0.95)
        plt.show()

        return True
    
    def generate_random_colors(self,base_list):
        num_colors = len(base_list)
        colors = []
        for _ in range(num_colors):
            color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
            colors.append(color)
        return colors