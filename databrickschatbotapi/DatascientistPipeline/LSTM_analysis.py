import opendatasets as od
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import os
#visualize data
from scipy.stats import norm
from scipy import stats
# Import missingno as msno
import missingno as msno
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

import sweetviz as sv

class LSTMAnalysis:
    def __init__(self, data, file_name):
        self.data = data
        self.file_name = file_name
    
    def most_corr_data(self, most_corr_features):
        most_corr_data = self.data[most_corr_features].copy()
        return most_corr_data 
    
    def pairplot(self, data):
        print("____________Pair Plot______________ \n")
        sns.pairplot(data)
        plt.savefig(f'LSTM_graphs/{self.file_name}_pairplot.png')
        plt.show
        
    def time_indexing(self, data, index_col):
        indexed = data.set_index(index_col)
        return indexed
    
    def df_plot(self, data):
        print("____________Dataframe Plot______________ \n")
        data.plot()
        plt.savefig(f'LSTM_graphs/{self.file_name}_dataframe_plot.png')
        plt.show()
        
    def select_numeric(self, data):
        numeric_columns = data.select_dtypes(include=['int', 'float']).columns
        df_numeric = data[numeric_columns]
        
        return df_numeric
        
        
        
    