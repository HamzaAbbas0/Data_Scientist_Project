
import pandas as pd
import numpy as np

def analysis(df):
    # Define stabilized gain as the average of each patient's BG
    df['Stabilized_Gain'] = df.groupby('Patient')['BG'].transform(lambda x: np.mean(x))

    # Return the stabilized gains of each patient
    return df['Stabilized_Gain'].values

# Read your data добавить
data = pd.read_csv('path_to_your_file.csv')
print(analysis(data))
