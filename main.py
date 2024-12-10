import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
    '''Display custom displots, where blue is valid, orange is fraudulent'''

    ax = plt.subplot()
    sns.kdeplot(dataset[variable][dataset.Class == 0], fill=True, ax=ax)
    sns.kdeplot(dataset[variable][dataset.Class == 1], fill=True, ax=ax)
    ax.set_title('Distribution of ' + variable)
    
    plt.tight_layout()
    plt.show()

displots_custom_class(dataset, 'Time')
displots_custom_class(dataset, 'Amount')

# displots for all variables showing fraudulent and valid transactions
columns = dataset.iloc[:,1:29].columns
for col in columns:
    displots_custom_class(dataset, col)

# removed outliers as seen from graph
scaler = RobustScaler().fit(dataset[['Time', 'Amount']])
dataset[['Time', 'Amount']] = scaler.transform(dataset[['Time', 'Amount']])

# splitting dataset into training and testing sets
y = dataset['Class']
x = dataset.iloc[:,0:30]

x_train, x_test, y_train, y_test = train_test_split( 
        x, y, test_size = 0.2, random_state = 42)

# random forest classifier model
rfc = RandomForestClassifier() 
rfc.fit(x_train, y_train) 
y_pred = rfc.predict(x_test)

# checking model's accuracy, precision, recall, and f1 score
print('Accuracy Score:', accuracy_score(y_test, y_pred)) 
print('Precision Score:', precision_score(y_test, y_pred))
print('Recall Score:', recall_score(y_test, y_pred))
print('F1 Score: ', f1_score(y_test, y_pred))