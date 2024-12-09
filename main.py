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
    '''Checks content of dataset'''

    temp = []
    columns = df.columns

    for col in columns:

        # checks type of data
        dtypes = df[col].dtypes

        # calculates the number of unique data
        nunique = df[col].nunique()

        # checks for any missing values

        sum_null = df[col].isnull().sum()
        temp.append([col, dtypes, nunique, sum_null])

    df_check = pd.DataFrame(temp)
    df_check.columns=['column', 'dtypes', 'nunique', 'sum_null']
    return df_check 

print(check(dataset), '\n')
print(dataset.describe(), '\n')
print(dataset.columns, '\n')

def boxplots_custom(dataset, columns_list, rows, cols, title):
    '''Creating boxplots to visualize data'''

    fig, axs = plt.subplots(rows, cols, sharey=True, figsize=(14, 6))
    fig.suptitle(title, y=0.97, size=23)
    fig.subplots_adjust(top=0.92)
    axs = axs.flatten()

    # for every column in the dataset, it is outputted as a boxplot
    for i, data in enumerate(columns_list):
        sns.boxplot(data=dataset[data], orient='h', ax=axs[i])
        axs[i].set_title(data + ', skewness is: '+str(round(dataset[data].skew(axis = 0, skipna = True), 2)))

boxplots_custom(dataset=dataset, columns_list=dataset.columns, rows=6, cols=3, title='Boxplots for all variables in dataset')
plt.tight_layout(pad=1.0)
plt.show()