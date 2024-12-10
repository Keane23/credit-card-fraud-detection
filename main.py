import pandas as pd
from pathlib import Path

# reading csv file (stored in another folder because git cannot hold more than 100 MB)
path = Path(__file__).resolve().parent.parent
dataset = pd.read_csv(path.joinpath(Path('creditcard.csv')))

# read the first 5 and last 5 rows of the data
print(dataset)

fraud = len(dataset[dataset['Class'] == 1])
valid = len(dataset[dataset['Class'] == 0])
print("Fraudulent Transactions: " + str(fraud))
print("Valid Transactions: " + str(valid))
print("Proportion of Fraudulent Cases: " + str(fraud/dataset.shape[0]))