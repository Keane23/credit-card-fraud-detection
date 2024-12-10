import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from pathlib import Path

path = Path(__file__).resolve().parent.parent
dataset = pd.read_csv(path.joinpath(Path('creditcard.csv')))

# read the first 5 and last 5 rows of the data
print(dataset, '\n')

# reading the data
fraud = len(dataset[dataset['Class'] == 1])
valid = len(dataset[dataset['Class'] == 0])
print('Fraudulent Transactions: ' + str(fraud))
print('Valid Transactions: ' + str(valid))
print('Proportion of Fraudulent Cases: ' + str(fraud/dataset.shape[0]))
print('Non-Missing Values: ' + str(dataset.isnull().shape[0]))
print('Missing Values: ' + str(dataset.shape[0] - dataset.isnull().shape[0]) + '\n')

def displots_custom(list, suptitle):
    '''Display custom displots'''

    fig, axes = plt.subplots(1, 2, figsize=(15,7))

    for i, data in enumerate(list):
        sns.kdeplot(data, color='m', fill=True, ax=axes[i]).set_title(suptitle[i])

    plt.tight_layout()
    plt.show()

# shows data spread of two named values
amount_value = dataset['Amount'].values
time_value = dataset['Time'].values
displots_custom([amount_value, time_value], ['Distribution of Amount', 'Distribution of Time'])

# checks to see any notable differences in valid and fraudulent
print('Average Amount in a Fraudulent Transaction: ' + str(dataset[dataset['Class'] == 1]['Amount'].mean()))
print('Average Amount in a Valid Transaction: ' + str(dataset[dataset['Class'] == 0]['Amount'].mean()))

def displots_custom_class(dataset, variable):
    '''Display custom displots'''

    ax = plt.subplot()
    sns.kdeplot(dataset[variable][dataset.Class == 0], fill=True, ax=ax)
    sns.kdeplot(dataset[variable][dataset.Class == 1], fill=True, ax=ax)
    ax.set_title('Distribution of ' + variable)
    
    plt.tight_layout()
    plt.show()

displots_custom_class(dataset, 'Time')