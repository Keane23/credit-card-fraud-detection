import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler

path = Path(__file__).resolve().parent.joinpath('dataset')
dataset = pd.read_csv(path.joinpath(Path('credit.csv')))

def check(df):
    temp = []
    columns = df.columns
    for col in columns:
        dtypes = df[col].dtypes
        nunique = df[col].nunique()
        sum_null = df[col].isnull().sum()
        temp.append([col, dtypes, nunique, sum_null])
    df_check = pd.DataFrame(temp)
    df_check.columns=['column', 'dtypes', 'nunique', 'sum_null']
    return df_check 

print(check(dataset))